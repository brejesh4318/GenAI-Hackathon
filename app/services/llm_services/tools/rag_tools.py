from typing import List, Dict, Any
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore
from langchain_core.tools import tool
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank
from langchain_tavily import TavilySearch
from app.utilities import dc_logger
from app.utilities.env_util import EnvironmentVariableRetriever


logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"tools": "V1"})


PROJECT_ID = EnvironmentVariableRetriever.get_env_variable("PROJECT_ID")
REGION = EnvironmentVariableRetriever.get_env_variable("REGION")
BUCKET = EnvironmentVariableRetriever.get_env_variable("BUCKET")
INDEX_NAME = EnvironmentVariableRetriever.get_env_variable("VERTEX_INDEX_NAME")
ENDPOINT_NAME = EnvironmentVariableRetriever.get_env_variable("VERTEX_INDEX_ENDPOINT")

BUCKET_URI = f"gs://{BUCKET}"
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)


my_index = aiplatform.MatchingEngineIndex(index_name=INDEX_NAME)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=ENDPOINT_NAME)

def get_vectorstore() -> VectorSearchVectorStore:
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    return VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=BUCKET,
        index_id=my_index.name if my_index else None,
        endpoint_id=my_index_endpoint.name if my_index_endpoint else None,
        embedding=embeddings,
        stream_update=False,
    )

vector_store = get_vectorstore()


reranker = VertexAIRank(
    project_id=PROJECT_ID,
    location_id="global",
    ranking_config="default_ranking_config",
    title_field="source",
    top_n=5,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

retriever_with_reranker = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)


@tool
def retrieve_by_standards(query: str, standard: str) -> List[Dict[str, Any]]:
    """
    Retrieves documents related to a specific compliance standard using RAG.
    Supports FDA and IEC 62304.
    """
    logger.info(f"RAG Tool Invoked with query: {query} and standard: {standard}")

    filters = [Namespace(name="doc_name", allow_tokens=["IEC-62304"])]
    docs = retriever_with_reranker.invoke(query, config={"filter": filters})

    all_docs = [
        {
            "query": query,
            "standard": standard,
            "contents": getattr(d, "page_content", ""),
        }
        for d in docs
    ]

    logger.info(f"RAG Tool Retrieved {len(all_docs)} documents")
    return all_docs


def web_search_tool(top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a web search using TavilySearch and returns the top_k results.
    """
    return TavilySearch(max_results=top_k)
