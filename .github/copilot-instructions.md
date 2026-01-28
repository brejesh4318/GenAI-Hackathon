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
- **Modular router architecture**: 7 separate routers mounted at `/v1/dash-test`
  - `auth_router`: `/auth/*` (login, register)
  - `dashboard_router`: Root endpoints (/, /dashboardData)
  - `projects_router`: `/projects/*` (CRUD operations)
  - `versions_router`: `/versions/*` (version management)
  - `testcases_router`: `/testcases/*` (test case generation/CRUD)
  - `requirements_router`: `/requirements/*` (document requirement extraction)
  - `utilities_router`: `/utils/*` (file upload, export, Jira integration)
- **Subapi pattern**: All routers included in `subapi` FastAPI instance, then mounted on main app
- Real-time middleware: process time logging (X-Process-Time header), CORS (allow all), GZIP compression, Trusted Host

#### 2. **LangGraph Agentic Workflow** (`app/services/testcase_service/graph_pipeline.py`)
- **GraphPipe** singleton orchestrates multi-stage pipeline with LangGraph `StateGraph`
- Workflow stages:
  1. **file_parser** → reads document using docling (PDF/DOCX/TXT/MD)
  2. **brain** → orchestrator agent, routes workflow based on document analysis
  3. **context_builder** → extracts requirements, builds context with tools
  4. **tools** → ToolNode handling interrupt_tool (user input), web_search (Tavily), RAG retrieval
  5. **test_generator** → generates test cases using LLM
  6. **validator** → validates/structures output using Pydantic with retry logic
- **Conditional routing**: Brain agent sets `state["next_action"]` to control flow
- **Interrupt mechanism**: Uses LangGraph `interrupt()` for user input during workflow
- **Memory/Checkpointing**: SQLite checkpointer (`langchain.db`) enables resumable workflows with thread-based state
- **Tool binding**: LLM with tools uses `llm.bind_tools([interrupt_tool, web_search_tool, retrieve_by_standards])`

#### 3. **Requirements Extraction Service** (`app/services/req_extract_service/requirements_extractor.py`)
- **RequirementsExtractor** singleton handles document → requirements → test cases flow
- **Deep extraction**: Configurable `pages_per_chunk` (default: 10) for batch LLM processing
- **Version-aware diffing**: Compares requirements across document versions using SHA-256 hashing
- Stores raw document pages and extracted requirements in MongoDB
- Links to SQLite projects/versions via `project_id` and `version_id`
- Returns diff summary: `{"added": [], "modified": [], "removed": [], "unchanged": []}`
- **Integration point**: Used by `requirements_router.py` for `/requirements/extract` endpoint

#### 4. **Prompt Management** (`app/services/prompt_service.py`)
- **PromptService** singleton with **Langfuse integration**
- **Startup loading**: Fetches all prompts from Langfuse at initialization (defined in `constants.yaml` → `prompt_names`)
- **In-memory caching**: `_prompt_cache` dict for fast access
- Methods: `get(name)`, `compile(name, variables)` for template substitution
- Prompts: brain-orchestrator-agent, compliance-researcher-agent, context-builder-agent, validator-agent, test-case-generator, requirement-extractor
- Replaces older database-based prompt fetching for better performance

#### 5. **LLM Abstraction** (`app/services/llm_services/`)
- **LLMInterface** abstract base with sync/async support (`generate()`, `agenerate()`)
- **LlmFactory** creates instances based on type enum
- Available models (via `gemini_models.py`):
  - `gemini_2.5_flash`: High-performance model for complex tasks
  - `gemini-2.5-flash-lite`: Lightweight model for simple operations
- Factory pattern allows swapping implementations without router changes
- All models accessed via **langchain-google-genai** bindings

#### 6. **Hybrid Database Architecture**

**SQLite + SQLAlchemy ORM** (`app/utilities/db_utilities/sqlite_implementation.py`, `models.py`)
- **SQLiteImplement** singleton with connection pooling (StaticPool, thread-safe)
- **ORM Models**: `Project`, `Version`, `User`, `ProjectPermission`, `AuditLog`
- **Integer autoincrement IDs**: `id = Column(Integer, primary_key=True, autoincrement=True)`
- **ID reuse prevention**: `__table_args__ = {'sqlite_autoincrement': True}`
- Automatic table creation: `Base.metadata.create_all(bind=self.engine)`
- Methods: `create()`, `get_by_id()`, `get_all()`, `update()`, `delete()`
- **Relationships**: Cascade deletes (e.g., deleting project removes versions and permissions)

**MongoDB (PyMongo)** (`app/utilities/db_utilities/mongo_implementation.py`)
- **MongoImplement** singleton for document-oriented data
- Collections (from `constants.yaml`):
  - `test_cases`: Test case documents with nested arrays
  - `requirements`: Extracted requirements with version tracking
  - `document_pages`: Raw page-level document content with SHA-256 hashes
- Methods: `read()`, `insert_one()`, `insert_many()`, `update_one()`
- TestCase schema: `test_case_id`, `feature`, `title`, `steps`, `expected_result`, `compliance_reference_standard`, `compliance_reference_clause`, etc.

**Why Hybrid?**
- SQLite: Relational data (projects, versions, users, permissions) with ACID guarantees
- MongoDB: Schema-flexible test cases, requirements with nested arrays, variable compliance fields

#### 7. **Authentication & Authorization** (`app/services/auth_service.py`, `app/routers/auth_router.py`)
- **AuthService** singleton (consolidated from old `auth_helper.py`)
- JWT token management with **python-jose**, password hashing with **bcrypt 4.0.1**
- JWT config from constants.yaml: `secret_key`, `algorithm` (HS256), `access_token_expire_minutes` (1440 = 24h)
- Token structure: `{"sub": user_id, "email": email, "exp": timestamp}`
- FastAPI dependency injection: `current_user: User = Depends(AuthService.get_current_user)`
- Permission checks: Query `ProjectPermission` model to verify user access (viewer, editor, owner)
- Endpoints: `/auth/register`, `/auth/login` (returns access token)

#### 8. **Configuration System** (`app/utilities/constants.py`, `app/resources/constants.yaml`)
- **Constants** singleton loads YAML config at module level
- Accessed via `Constants.fetch_constant("key_name")`
- Config includes: LLM settings, MongoDB params, SQLite params, JWT settings, prompt names, collection names, feature flags (`deep_file_extractor`)

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
**Used by**: Constants, MongoImplement, SQLiteImplement, LLMInterface implementations, Helper, TestCaseGenerator, GraphPipe, RequirementsExtractor, PromptService, AuthService

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
**Location**: `app/services/testcase_service/graph_pipeline.py`, `graph_state.py`

**Key technique**: Use `next_action` field in state to control routing. Brain agent analyzes document and sets next node.

### Pattern 4: Prompt Management with Langfuse
```python
from app.services.prompt_service import PromptService

prompt_service = PromptService()  # Singleton, loads all prompts at startup
brain_prompt = prompt_service.get("brain-orchestrator-agent")
# For templates with variables
compiled = prompt_service.compile("test-case-generator", {"document": doc_text})
```
**Location**: `app/services/prompt_service.py`, prompts defined in Langfuse

**Migration note**: Old code may use `PromptFetcher()` - new code should use `PromptService()`.

### Pattern 5: FastAPI Dependency Injection for Auth
```python
from app.services.auth_service import AuthService
from app.utilities.db_utilities.models import User

@router.get("/protected")
async def protected_route(current_user: User = Depends(AuthService.get_current_user)):
    # current_user is validated User object from database
    # Raises 401 if token invalid/expired
    return {"user_id": current_user.id, "email": current_user.email}
```
**Location**: All protected endpoints use this pattern

**Permission check**: After getting `current_user`, query `ProjectPermission` for resource access.

### Pattern 6: Logging with Context
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
   - Uses docling for PDF/DOCX extraction (markdown + base64 images)
   - State populated: `file_path`, `document`, `images`

3. **Brain Agent Decision** → `brain` node
   - Analyzes document structure and complexity
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

## Data Flow: Requirements Extraction

1. **Extract Requirements** (`POST /requirements/extract`)
   - Document uploaded with `project_id`, `version_id`, optional `previous_version_id`
   
2. **Document Parsing** → `RequirementsExtractor.extract_and_store()`
   - Parse document into pages using docling (PDF/DOCX/TXT/MD support)
   - Hash each page with SHA-256 for change detection
   - Store raw pages to `document_pages` collection

3. **Batch LLM Processing**
   - Process document in chunks (configurable `pages_per_chunk`, default: 10)
   - LLM extracts requirements from each chunk using `requirement-extractor` prompt
   - Returns structured requirements with IDs, descriptions, priority, etc.

4. **Version Diffing** (if `previous_version_id` provided)
   - Load previous version's requirements from MongoDB
   - Compare using SHA-256 hashes: `{"added": [], "modified": [], "removed": [], "unchanged": []}`
   - Identify which requirements need new test case generation

5. **Persistence**
   - Store requirements to `requirements` collection (MongoDB)
   - Link to SQLite project/version via `project_id` and `version_id`
   - Return diff summary + `generate_tests_for` list

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
# Required environment variables:
# - GOOGLE_CRED: Base64-encoded GCP service account JSON (for Gemini API)
# - MONGO_DB_URI: MongoDB connection string  
# - LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST: For prompt management

# Start FastAPI dev server
uvicorn app.main:app --reload --port 8000

# Test endpoints
Invoke-WebRequest http://localhost:8000/  # Health check
Invoke-WebRequest http://localhost:8000/v1/dash-test/  # Subapi health
```

### Running Tests
```powershell
# If tests exist (currently minimal)
pytest app/
```

### Docker Build & Deploy
```bash
docker build -t genai-hackathon:latest .
docker run -p 8080:8080 \
  --env GOOGLE_CRED='<base64_gcp_key>' \
  --env MONGO_DB_URI='<connection_string>' \
  genai-hackathon
```
**Note**: Dockerfile uses Python 3.12 from deadsnakes PPA, Poetry package manager, Gunicorn with UvicornWorker (4 workers).

### Debugging LangGraph Workflows

**Visualize graph structure:**
```python
# In graph_pipeline.py
png = workflow_graph.get_graph().draw_mermaid_png()
with open("workflow_graph.png", "wb") as f:
    f.write(png)
```

**Inspect state at any node:**
```python
# Add to node functions
logger.debug(f"Current state: {state}")
logger.debug(f"Next action: {state.get('next_action')}")
```

**Resume interrupted workflows:**
```python
# Use thread_id to resume from checkpointer
config = {"configurable": {"thread_id": thread_id}}
result = graph_pipeline.graph.invoke(state, config)
```

**Check stored checkpoints:**
```bash
# SQLite database: langchain.db
sqlite3 langchain.db
.tables  # Shows checkpoint-related tables
SELECT * FROM checkpoints LIMIT 10;
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
**Note**: SQLite uses integer autoincrement IDs; no ObjectId conversion needed for projects/versions/users.

### 4. **Temp File Cleanup Pattern**
```python
try:
    path = Helper.save_file("/tmp", content=content, filename=filename)
    # process file
finally:
    if os.path.exists(path):
        os.remove(path)
```
**Location**: Used in `testcases_router.py` for document uploads.

### 5. **Batching External API Calls**
```python
BATCH_SIZE = 45
for i in range(0, len(issues), BATCH_SIZE):
    batch = issues[i:i+BATCH_SIZE]
    # send to Jira
    await asyncio.sleep(1)  # Rate limiting
```
**Location**: `/jiraPush` endpoint in `utilities_router.py` for Jira bulk issue creation.

### 6. **Router Registration Pattern**
```python
# In main.py - all routers mounted on subapi
subapi = FastAPI(title="Dash-Test API", version="v1")
subapi.include_router(auth_router.router)  # No prefix conflicts
subapi.include_router(projects_router.router)
app.mount("/v1/dash-test", subapi)
```
**Why**: Clean namespace separation, versioning support, easier testing.

### 7. **Permission Checking Pattern**
```python
# In protected endpoint
user_perms = sqlite_client.get_all(
    ProjectPermission,
    filters={"project_id": project_id, "user_id": current_user.id}
)
if not user_perms:
    raise HTTPException(status_code=403, detail={...})

# Check role if needed
if user_perms[0].role not in ["editor", "owner"]:
    raise HTTPException(status_code=403, detail={"message": "Insufficient permissions"})
```
**Roles**: `viewer` (read-only), `editor` (modify), `owner` (full control, delete).

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
2. Create async function with auth dependency: `current_user: User = Depends(AuthService.get_current_user)`
3. Use SQLiteImplement for projects/versions/permissions; MongoImplement for test cases
4. Check permissions: Query `ProjectPermission` model with `user_id` and `project_id`
5. Follow error handling convention (HTTPException with detail dict)

Example:
```python
@router.get("/projects/{project_id}")
async def get_project(project_id: int, current_user: User = Depends(AuthService.get_current_user)):
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

### Add a New Prompt to Langfuse
1. Create prompt in Langfuse UI with specific name (e.g., "my-new-prompt")
2. Add prompt name to `constants.yaml` under `prompt_names` list
3. Restart app to load prompt into cache
4. Access via `PromptService().get("my-new-prompt")`
5. For templates: Use `{{variable_name}}` syntax, compile with `PromptService().compile("name", {"variable_name": "value"})`

---

## Known Limitations & Future Work

- **Single file upload**: `files[0]` hardcoded in testcaseGenerator (TODO: handle multiple files)
- **No test suite**: Minimal unit/integration tests
- **Interrupt handling**: Currently basic; resume logic could be more robust
- **Compliance standards**: Currently RAG/web search; could pre-ingest standard documents

---

## References & Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app initialization, middleware setup |
| `app/routers/projects_router.py` | Project CRUD operations with auth |
| `app/routers/testcases_router.py` | Test case generation and management |
| `app/routers/requirements_router.py` | Requirements extraction and version diffing |
| `app/routers/auth_router.py` | User registration, login, token management |
| `app/services/testcase_service/graph_pipeline.py` | LangGraph workflow orchestration |
| `app/services/req_extract_service/requirements_extractor.py` | Deep document parsing and requirement extraction |
| `app/services/prompt_service.py` | Langfuse-based prompt management with caching |
| `app/services/auth_service.py` | JWT token creation/validation, password hashing |
| `app/utilities/db_utilities/models.py` | SQLAlchemy ORM models (Project, User, Version, etc.) |
| `app/utilities/constants.py` | YAML-based configuration |
| `app/utilities/helper.py` | Utility functions for LLM calls, file ops |
| `app/resources/constants.yaml` | Configuration (LLM, MongoDB, prompts) |
| `pyproject.toml` | Dependencies (LangChain, LangGraph, FastAPI, etc.) |
| `Dockerfile` | Containerization with Python 3.12, Gunicorn |

---

## Questions to Clarify Before Major Changes

1. **Compliance Standards**: Pre-load standards into vector DB for faster retrieval?
2. **Test Generation Logic**: Should validation use a separate LLM from generation for quality?
3. **User Interrupts**: Should interrupt responses be streamed or returned as batch?
4. **Multi-file Support**: How should document relationships be modeled?
5. **Performance Optimization**: Should we add Redis for caching frequently accessed data?

