# Copilot Instructions for GenAI-Hackathon

## Project Overview

**GenAI-Hackathon** is a compliance-driven QA test case generation system built with **LangGraph** and **FastAPI**. It processes product requirement documents (PRDs) and generates comprehensive test cases aligned with compliance standards (FDA, IEC 62304, etc.). The system uses multi-agent agentic workflows orchestrated through LangGraph state graphs.

### Core Purpose

Convert technical documents into actionable test cases with compliance traceability, enabling teams to auto-generate QA coverage for regulated industries (healthcare, finance).

---

## Architecture Overview

### Key Components

#### 1. **FastAPI Application Layer** (`app/main.py`, `app/routers/`)
- REST API exposing document upload, project management, and test case generation
- **Modular router architecture**: Separate routers for `auth`, `projects`, `versions`, `testcases`, `dashboard`
- Endpoints follow pattern: `/v1/dash-test/{resource}`
- All routes mounted at `/v1/dash-test` via subapi pattern
- Real-time middleware: process time logging, CORS (allow all), GZIP compression, Trusted Host

#### 2. **LangGraph Agentic Workflow** (`app/services/llm_services/graph_pipeline.py`)
- **GraphPipe** singleton orchestrates multi-stage pipeline with LangGraph `StateGraph`
- Workflow stages:
  1. **file_parser** → reads document (PDF/DOCX/TXT/MD)
  2. **brain** → routes workflow (orchestrator, decides next action)
  3. **context_builder** → extracts requirements, builds context with tools
  4. **tools** → ToolNode handling interrupt_tool, web_search, RAG retrieval
  5. **test_generator** → generates initial test cases
  6. **validator** → validates and structures output using Pydantic
- **Conditional routing**: Brain agent determines next node via `should_continue()` router
- **Interrupt mechanism**: Uses LangGraph `interrupt()` for user input during workflow
- **Memory/Checkpointing**: SQLite checkpointer (`langchain.db`) enables resumable workflows with thread-based state

#### 3. **Document Parsing** (`app/utilities/document_parser.py`)
- **Singleton pattern**: Single instance across app
- Supports:
  - **Docling formats**: PDF, DOCX (extracts images as base64 data URIs)
  - **Text formats**: TXT, MD (returns content as-is)
- Returns tuple: `(markdown_string, list_of_base64_images)`
- Used by both file parsing and RAG ingestion

#### 4. **LLM Abstraction** (`app/services/llm_services/`)
- **LLMInterface** abstract base with sync/async support (`generate()`, `agenerate()`)
- **LlmFactory** creates instances based on type enum (Gemini 2.5 Flash / Lite)
- Factory pattern allows swapping implementations without router changes
- Both models accessed via **langchain-google-genai** bindings

#### 5. **Hybrid Database Architecture**

**SQLite + SQLAlchemy ORM** (`app/utilities/db_utilities/sqlite_implementation.py`, `models.py`)
- **SQLiteImplement** singleton with connection pooling (StaticPool, thread-safe)
- **ORM Models**: `Project`, `Version`, `User`, `ProjectPermission`, `AuditLog`
- UUID-based primary keys: `id = Column(String(36), primary_key=True, default=generate_uuid)`
- Automatic table creation: `Base.metadata.create_all(bind=self.engine)`
- Methods: `create()`, `get_by_id()`, `get_all()`, `update()`, `delete()`
- Context manager for session handling:
  ```python
  with sqlite_client.get_session() as session:
      project = session.query(Project).filter_by(id=project_id).first()
  ```

**MongoDB (PyMongo)** (`app/utilities/db_utilities/mongo_implementation.py`)
- **MongoImplement** singleton for test cases and prompts
- Collections: `test_cases`, `prompts` (from constants.yaml)
- Methods: `read()`, `insert_one()`, `insert_many()`, `update_one()`
- TestCase schema: `test_case_id`, `feature`, `title`, `steps`, `expected_result`, `compliance_reference_standard`, `compliance_reference_clause`, etc.

**Why Hybrid?**
- SQLite: Relational data (projects, versions, users, permissions) with ACID guarantees
- MongoDB: Schema-flexible test cases with nested arrays, variable compliance fields

#### 6. **Authentication & Authorization** (`app/utilities/auth_helper.py`, `app/routers/auth_router.py`)
- **AuthManager** singleton handles JWT tokens, password hashing (bcrypt)
- JWT config from constants.yaml: `secret_key`, `algorithm` (HS256), `access_token_expire_minutes` (1440 = 24h)
- Token structure: `{"sub": user_id, "email": email, "exp": timestamp}`
- FastAPI dependency injection: `current_user: User = Depends(AuthManager.get_current_user)`
- Permission checks: Query `ProjectPermission` model to verify user access (`viewer`, `editor`, `owner`)
- Endpoints: `/auth/register`, `/auth/login` (returns access token)

#### 7. **Configuration System** (`app/utilities/constants.py`, `app/resources/constants.yaml`)
- **Constants** singleton loads YAML config at module level
- Accessed via `Constants.fetch_constant("key_name")`
- Config includes LLM settings, MongoDB params, SQLite params, JWT settings, prompt templates, collection names

---

## Key Design Patterns

### Pattern 1: Singleton Metaclass (Dependency Management)
```python
from app.utilities.singletons_factory import DcSingleton

class MyService(metaclass=DcSingleton):
    def __init__(self):
        # Single instance guaranteed across app lifecycle
        pass
```
**Used by**: DocumentParser, Constants, MongoImplement, LLMInterface implementations, Helper, TestCaseGenerator, GraphPipe

**When adding services**: Use `DcSingleton` to ensure app-wide sharing of expensive resources (DB connections, LLM clients, document parsers).

### Pattern 2: Factory Pattern for LLM Selection
```python
# Get LLM type from config, instantiate via factory
llm = LlmFactory.get_llm(type=Constants.fetch_constant("llm_model")["model_name"])
llm_tools = LlmFactory.get_llm(type="gemini_2.5_flash")
```
**Location**: `app/services/llm_services/llm_factory.py`

**To add new model**: Create new class inheriting `LLMInterface`, add case in factory's `get_llm()`.

### Pattern 3: LangGraph StateGraph for Agentic Workflows
```python
# State management via TypedDict
class AgentState(TypedDict):
    file_path: str
    document: str
    messages: Annotated[List[AnyMessage], add_messages]
    test_cases_final: List[dict]
    next_action: str  # Routes workflow
    user_interrupt: Optional[str]  # For user input prompts
    context_agent_messages: Annotated[list, add_messages]

# Conditional edges for routing
workflow_graph.add_conditional_edges("brain", self.should_continue, 
    {"context_builder": "context_builder", "test_generator": "test_generator"})
```
**Location**: `app/services/llm_services/graph_pipeline.py`, `app/services/llm_services/graph_state.py`

**Key technique**: Use `next_action` field in state to control routing (like a lookup table for edges).

### Pattern 4: Prompt Templating + Output Parsing
```python
from app.services.prompt_fetch import PromptFetcher
from langchain.output_parsers import PydanticOutputParser

fetch_prompt = PromptFetcher()
brain_prompt = fetch_prompt.fetch("spec2test-brain-agent")  # from database
output_parser = PydanticOutputParser(pydantic_object=AgentFormat)

# Use in LLM call
validated = output_parser.parse(llm.generate(prompt))
```
**Location**: Prompts stored in constants.yaml or MongoDB; parsing uses Pydantic models (`app/services/reponse_format.py`)

### Pattern 5: Logging with Context
```python
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
logger.info("Operation completed")  # Adds context to all logs
```
**Location**: `app/utilities/dc_logger.py`

**Usage**: Add context dict as second argument to LoggerAdap for request tracing.

---

## Data Flow: Document → Test Cases

1. **Upload Document** (`POST /testcases/generate`)
   - File saved to `/tmp/` with UUID prefix
   - Project ID validated in SQLite, permissions checked

2. **Document Parsing** → `GraphPipe.file_parser()` node
   - DocumentParser extracts markdown + images
   - State populated: `file_path`, `document`, `images`

3. **Brain Agent Decision** → `brain` node
   - Decides: need more context? compliance research? proceed to generation?
   - Sets `next_action` field to route workflow

4. **Context Building** → `context_builder` node (if needed)
   - LLM identifies requirements, scope, compliance keywords
   - May call tools (interrupt for user input, web search, RAG)
   - Returns enriched document context

5. **Test Case Generation** → `test_generator` node
   - LLM generates test cases from document + context
   - Output format: JSON with `test_case_id`, `steps`, `expected_result`, etc.

6. **Validation** → `validator` node
   - Pydantic parser enforces schema
   - Retries on parse failure (max 2 attempts, 2sec delay via `@retry`)
   - Returns `FinalOutput` with validated test cases list

7. **Persistence**
   - Insert test cases to `test_cases` collection (MongoDB)
   - Update project metadata (`no_test_cases`, `updated_at`) in SQLite

---

## Critical Developer Workflows

### Setting Up Environment
```powershell
# Install dependencies (Python 3.12+)
pip install -r requirements.txt
# or with poetry (recommended)
poetry install --no-interaction --no-ansi
```

### Running Locally
```powershell
# Set environment variables first (see .env requirements below)
# Start FastAPI dev server
uvicorn app.main:app --reload --port 8000
# Test endpoint
Invoke-WebRequest http://localhost:8000/v1/dash-test/
```

### Running Tests
```powershell
# If tests exist (currently minimal)
pytest app/
```

### Docker Build & Deploy
```bash
docker build -t genai-hackathon:latest .
docker run -p 8080:8080 --env GOOGLE_CRED='<base64_gcp_key>' genai-hackathon
```
**Note**: Dockerfile uses Python 3.12 from deadsnakes PPA, Poetry package manager, Gunicorn with UvicornWorker.

### Environment Variables Required
- **GOOGLE_CRED**: Base64-encoded GCP service account JSON (for Gemini API)
- **MONGO_DB_URI**: MongoDB connection string
- All config in `app/resources/constants.yaml`

### Debugging Graph Workflow
```python
# In graph_pipeline.py, uncomment to generate PNG:
png = workflow_graph.get_graph().draw_mermaid_png()
with open("workflow_graph.png", "wb") as f:
    f.write(png)
# Visualize workflow state at any node:
logger.debug(f"State: {state}")
```

---

## Common Patterns & Conventions

### 1. **Async/Await for Long-Running Operations**
```python
# In router, wrap heavy LLM calls in thread pool
test_cases = await asyncio.to_thread(
    testcase_generator.generate_testcase, 
    document_path=path, 
    project_id=project_id
)
```
**Why**: Keep FastAPI event loop responsive; avoid blocking.

### 2. **Error Handling with HTTPException**
```python
if not project:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"status": "Failed", "message": "Project not found"}
    )
```
**Convention**: Always return `{"status": "Success/Failed", "message": "..."}` in detail.

### 3. **MongoDB ObjectId Conversion**
```python
from bson import ObjectId

# Convert string to ObjectId for queries
test_case = mongo_client.read("test_cases", {"_id": ObjectId(testcase_id)}, max_count=1)
# Convert ObjectId back to string for responses
return {"testcase_id": str(test_case["_id"])}
```
**Note**: SQLite uses UUID strings; no ObjectId conversion needed for projects/versions/users.

### 4. **Temp File Cleanup Pattern**
```python
try:
    path = Helper.save_file("/tmp", content=content, filename=filename)
    # process file
finally:
    if os.path.exists(path):
        os.remove(path)
```
**Location**: `app/routers/router.py` in `testcaseGenerator` endpoint.

### 5. **Batching External API Calls**
```python
BATCH_SIZE = 45
for i in range(0, len(issues), BATCH_SIZE):
    batch = issues[i:i+BATCH_SIZE]
    # send to Jira
    await asyncio.sleep(1)  # Rate limiting
```
**Location**: `/jiraPush` endpoint for Jira bulk issue creation.

---

## Adding New Features

### Add a New LLM Model
1. Create class in `app/services/llm_services/llm_implementation/`
2. Inherit `LLMInterface`, implement `generate()`, `agenerate()`, `get_llm()`
3. Add case in `LlmFactory.get_llm()`
4. Update `app/utilities/dc_enums.py` with new enum value

### Add a New Graph Node
1. Define node function taking `state: AgentState` and returning updated state dict
2. Add to graph: `workflow_graph.add_node("node_name", self.node_function)`
3. Add edges: `workflow_graph.add_edge()` or `add_conditional_edges()`
4. Update `AgentState` TypedDict if new fields needed (in `graph_state.py`)
5. Use `state["next_action"] = "node_name"` to control routing from brain agent

### Add a New API Endpoint
1. Choose appropriate router file (`projects_router.py`, `testcases_router.py`, etc.)
2. Create async function with auth dependency: `current_user: User = Depends(AuthManager.get_current_user)`
3. Use SQLiteImplement for projects/versions/permissions; MongoImplement for test cases
4. Check permissions: Query `ProjectPermission` model with `user_id` and `project_id`
5. Follow error handling convention (HTTPException with detail dict)

Example:
```python
@router.get("/projects/{project_id}")
async def get_project(project_id: str, current_user: User = Depends(AuthManager.get_current_user)):
    # Check permission
    perms = sqlite_client.get_all(ProjectPermission, filters={"project_id": project_id, "user_id": current_user.id})
    if not perms:
        raise HTTPException(status_code=403, detail={"status": "Failed", "message": "Access denied"})
    # Get project
    project = sqlite_client.get_by_id(Project, project_id)
    return {"status": "Success", "data": project.to_dict()}
```

### Add a New MongoDB Collection
1. Add collection name to `app/resources/constants.yaml` under `mongo_collections`
2. Reference via `Constants.fetch_constant("mongo_collections")["collection_name"]`
3. Use MongoImplement methods: `read()`, `insert_one()`, `insert_many()`, `update_one()`

---

## Known Limitations & Future Work

- **Single file upload**: `files[0]` hardcoded in testcaseGenerator (TODO: handle multiple files)
- **No test suite**: Minimal unit/integration tests
- **Interrupt handling**: Currently basic; resume logic could be more robust
- **Prompt fetching**: Mix of YAML and MongoDB storage (consolidate to one source)
- **Compliance standards**: Currently RAG/web search; could pre-ingest standard documents

---

## References & Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app initialization, middleware setup |
| `app/routers/router.py` | All API endpoints (project, test case, export, Jira) |
| `app/routers/projects_router.py` | Project CRUD operations with auth |
| `app/routers/testcases_router.py` | Test case generation and management |
| `app/routers/auth_router.py` | User registration, login, token management |
| `app/services/llm_services/graph_pipeline.py` | LangGraph workflow orchestration |
| `app/utilities/document_parser.py` | Multi-format document ingestion |
| `app/utilities/auth_helper.py` | JWT token creation/validation, password hashing |
| `app/utilities/db_utilities/models.py` | SQLAlchemy ORM models (Project, User, Version, etc.) |
| `app/utilities/constants.py` | YAML-based configuration |
| `app/utilities/helper.py` | Utility functions for LLM calls, file ops |
| `app/resources/constants.yaml` | Configuration (LLM, MongoDB, prompts) |
| `pyproject.toml` | Dependencies (LangChain, LangGraph, FastAPI, etc.) |
| `Dockerfile` | Containerization with Python 3.12, Gunicorn |

---

## Questions to Clarify Before Major Changes

1. **Prompt Storage**: Should all prompts move to MongoDB or stay in YAML?
2. **Compliance Standards**: Pre-load standards into vector DB for faster retrieval?
3. **Test Generation Logic**: Should validation use a separate LLM from generation for quality?
4. **User Interrupts**: Should interrupt responses be streamed or returned as batch?
5. **Multi-file Support**: How should document relationships be modeled?

