from typing import List, Dict, Any
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from langchain_core.tools import tool
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank
from langchain_tavily import TavilySearch
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from app.utilities import dc_logger
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.singletons_factory import DcSingleton
from typing import List, Dict, Any, Union



logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"tools": "V1"})


PROJECT_ID = EnvironmentVariableRetriever.get_env_variable("PROJECT_ID")
REGION = EnvironmentVariableRetriever.get_env_variable("REGION")
BUCKET = EnvironmentVariableRetriever.get_env_variable("BUCKET")
INDEX_NAME = EnvironmentVariableRetriever.get_env_variable("VERTEX_INDEX_NAME")
ENDPOINT_NAME = EnvironmentVariableRetriever.get_env_variable("VERTEX_INDEX_ENDPOINT")
BUCKET_URI = f"gs://{BUCKET}"
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


class RAGRetriever(metaclass=DcSingleton):
    def __init__(self, index_name: str, endpoint_name: str):
        self.index, self.index_endpoint = self._init_index(index_name, endpoint_name)
        self.vector_store = self._get_vectorstore()
        self.reranker = VertexAIRank(
            project_id=PROJECT_ID,
            location_id="global",
            ranking_config="default_ranking_config",
            title_field="source",
            top_n=3,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        self.retriever_with_reranker = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.retriever
        )

    def _init_index(self, index_name, endpoint_name):
        my_index = aiplatform.MatchingEngineIndex(index_name=index_name)
        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
        return my_index, my_index_endpoint

    def _get_vectorstore(self) -> VectorSearchVectorStore:
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        return VectorSearchVectorStore.from_components(
            project_id=PROJECT_ID,
            region=REGION,
            gcs_bucket_name=BUCKET,
            index_id=self.index.name if self.index else None,
            endpoint_id=self.index_endpoint.name if self.index_endpoint else None,
            embedding=embeddings,
            stream_update=False,
        )
    def retrieve(self, query: str, filters: List[Namespace] = None) -> List[Any]:

        return self.retriever_with_reranker.invoke(query, filter=filters)


@tool
def retrieve_by_standards(query: Union[str, List[str]], 
                          standard: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Retrieve compliance-related documents from the vector database (RAG pipeline).

    Arguments:
        query (str | List[str]): One or more queries describing what compliance information to search.
        standard (str | List[str]): One or more compliance standards to restrict retrieval to.
            Must be "FDA", "IEC-62304", ISO-27001, ISO-13485.

    Returns:
        List[Dict[str, Any]]: Each dictionary contains the page content of a relevant document.
    """
    try:
        rag_retriever = RAGRetriever(INDEX_NAME, ENDPOINT_NAME)
        # Normalize to lists
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query

        if isinstance(standard, str):
            standards = [standard]
        else:
            standards = standard

        # Validate standards
        valid_standards = {"FDA", "IEC-62304"}
        invalid = set(standards) - valid_standards
        if invalid:
            raise ValueError(f"Unsupported standards: {invalid}. Allowed: {valid_standards}")

        all_docs = []

        # Run retrieval per query
        for q in queries:
            filters = [Namespace(name="doc_name", allow_tokens=standards)]
            docs = rag_retriever.retrieve(q, filters)

            for d in docs:
                all_docs.append({
                    "query": q,
                    "standard": standards,
                    "content": getattr(d, "page_content", "")
                })

        logger.info(f"RAG Tool Retrieved {len(all_docs)} documents")
        return all_docs

    except Exception as e:
        logger.error(f"Error in retrieve_by_standards: {e}")
        return []



def web_search_tool(top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a web search using TavilySearch and returns the top_k results.
    """
    return TavilySearch(max_results=top_k)
