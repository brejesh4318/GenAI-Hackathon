"""
Utility endpoints for file operations and integrations
"""
import asyncio
import io
import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
import httpx
from httpx import BasicAuth

from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project
from app.utilities.helper import Helper
from app.routers.datamodel import JiraPushRequest

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    prefix="/utils",
    tags=["Utilities"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "not found"}}
)

# Initialize database clients
mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("MONGO_DB_URI"),
    db_name=Constants.fetch_constant("mongo_db")["db_name"],
    max_pool=Constants.fetch_constant("mongo_db")["max_pool_size"],
    server_selection_timeout=Constants.fetch_constant("mongo_db")["server_selection_timeout"]
)

sqlite_config = Constants.fetch_constant("sqlite_db")
sqlite_client = SQLiteImplement(
    db_path=sqlite_config["db_path"],
    max_pool_size=sqlite_config["max_pool_size"]
)

testcase_collection = Constants.fetch_constant("mongo_collections")["testcases"]


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file"""
    try:
        content = await file.read()
        filename = file.filename

        # Save the uploaded file
        path = Helper.save_file("/tmp", content=content, filename=filename)
        logger.info(f"Saved uploaded file to: {path}")
        file_content, images = Helper.read_file(file_path=path)
        logger.info(f"Read file content: {len(file_content.split())} words, {len(images)} images")
        return {"status": "success", "file_path": path}
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": str(e)}
        )


@router.get("/export/{project_id}")
async def export_test_cases(project_id: int):
    """Export test cases to CSV format"""
    try:
        # Get project from SQLite using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": "Project not found"}
            )

        # Get test cases from MongoDB
        test_cases = list(mongo_client.read(testcase_collection, {"project_id": project_id}))
        if not test_cases:
            logger.warning(f"No test cases found for project id '{project_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": "No test cases found for this project"}
            )

        # Normalize data for CSV
        data = []
        for tc in test_cases:
            data.append({
                "Test Case ID": tc.get("test_case_id"),
                "Feature": tc.get("feature"),
                "Title": tc.get("title"),
                "Type": tc.get("type"),
                "Priority": tc.get("priority"),
                "Preconditions": "\n".join(tc.get("preconditions", [])) if tc.get("preconditions") else "",
                "Test Data": str(tc.get("test_data")) if tc.get("test_data") else "",
                "Steps": "\n".join(tc.get("steps", [])) if tc.get("steps") else "",
                "Expected Result": "\n".join(tc.get("expected_result", [])) if tc.get("expected_result") else "",
                "Postconditions": tc.get("postconditions") or "",
                "Compliance Standard": tc.get("compliance_reference_standard") or "",
                "Compliance Clause": tc.get("compliance_reference_clause") or "",
                "Compliance Requirement": tc.get("compliance_reference_requirement_text") or ""
            })

        # Create DataFrame and export to CSV
        df = pd.DataFrame(data)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        filename = f"{project.project_name}_test_cases.csv"
        logger.info(f"Exporting {len(test_cases)} test cases for project '{project.project_name}'")
        
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.error(f"Failed to export test cases: {str(exe)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/jira/push")
async def push_to_jira(request: JiraPushRequest):
    """
    Push selected test cases to Jira in bulk mode with batching.
    Supports rate limiting and handles large batches.
    """
    try:
        test_case_ids = request.selected_ids
        project_id = request.project_id
        domain_name = request.domain_name
        jira_api = request.jira_api
        
        # Fetch project info from SQLite
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": "Project not found"}
            )

        # Jira API setup
        jira_url = f"https://{domain_name}.atlassian.net/rest/api/3/issue/bulk"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        all_issues = []
        logger.info(f"Fetching {len(test_case_ids)} test cases from MongoDB")
        
        # Build issues payload for each test case
        for test_case_id in test_case_ids:
            test_case = mongo_client.read(testcase_collection, {"_id": ObjectId(test_case_id)})
            if not test_case:
                logger.warning(f"Test case {test_case_id} not found, skipping")
                continue
            test_case = test_case[0]

            description_blocks = []
            
            def add_block(label, value):
                """Helper to format rich text paragraphs"""
                if value:
                    if isinstance(value, list):
                        content_text = "\n".join(value)
                    elif isinstance(value, dict):
                        content_text = "\n".join(f"{k}: {v}" for k, v in value.items())
                    else:
                        content_text = str(value)
                    description_blocks.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": f"{label}: {content_text}"}]
                    })

            add_block("Feature", test_case.get("feature"))
            add_block("Type", test_case.get("type"))
            add_block("Priority", test_case.get("priority"))
            add_block("Preconditions", test_case.get("preconditions"))
            add_block("Test Data", test_case.get("test_data"))
            add_block("Steps", test_case.get("steps"))
            add_block("Expected Result", test_case.get("expected_result"))
            add_block("Postconditions", test_case.get("postconditions"))
            add_block("Compliance Ref Standard", test_case.get("compliance_reference_standard"))
            add_block("Compliance Ref Clause", test_case.get("compliance_reference_clause"))
            add_block("Compliance Ref Requirement", test_case.get("compliance_reference_requirement_text"))

            issue = {
                "fields": {
                    "project": {"key": getattr(project, "project_key", "HS")},
                    "summary": test_case.get("title", "Untitled Test Case"),
                    "issuetype": {"name": "Task"},
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": description_blocks or [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "No description provided"}]
                        }],
                    },
                    "labels": ["spec2test", "auto-generated"],
                }
            }
            all_issues.append(issue)

        # Push to Jira in batches to respect API rate limits
        BATCH_SIZE = 45
        created_issues = []
        
        logger.info(f"Pushing {len(all_issues)} test cases to Jira in batches of {BATCH_SIZE}")
        
        for i in range(0, len(all_issues), BATCH_SIZE):
            batch = all_issues[i:i + BATCH_SIZE]
            payload = {"issueUpdates": batch}
            
            logger.info(f"Pushing batch {i // BATCH_SIZE + 1}: {len(batch)} issues")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    jira_url,
                    json=payload,
                    headers=headers,
                    auth=BasicAuth(request.jira_mail_id, jira_api),
                    timeout=30.0,
                )

            if response.status_code not in (200, 201):
                logger.error(f"Jira API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail={"status": "Failed", "message": response.text}
                )

            created_issues.extend(response.json().get("issues", []))

            # Rate limiting: 1 request per second
            await asyncio.sleep(1)

        logger.info(f"Successfully pushed {len(created_issues)} issues to Jira")
        
        return {
            "status": "success",
            "total_pushed": len(created_issues),
            "created_issues": created_issues,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to push to Jira: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(e)}
        )
