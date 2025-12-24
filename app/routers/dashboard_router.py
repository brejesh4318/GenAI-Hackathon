"""
Dashboard and miscellaneous endpoints
"""
import io
import asyncio
import pandas as pd
from typing import Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import BasicAuth
import httpx
from app.utilities.auth_helper import AuthManager

from app.routers.datamodel import JiraPushRequest
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, ProjectPermission, User
from app.utilities.helper import Helper
from bson import ObjectId
from datetime import datetime

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    tags=["Dashboard & Utility"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "not found"}}
)

# Initialize database clients
mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("FIRESTORE_DB_URI"),
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


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy"}


@router.get("/dashboardData")
async def get_dashboard_data(current_user: User = Depends(AuthManager.get_current_user)):
    """Get dashboard statistics (requires authentication)"""
    try:
        compliance_coverage = 0
        testcases_generated = 0
        compliance_covered = 0
        timesaved = 0
        recent_projects = []
        
        # Get recent projects from SQLite
        all_projects = sqlite_client.get_all(Project, order_by=Project.created_at.desc())
        
        # Filter by user permissions
        user_id = current_user.id
        user_perms = sqlite_client.get_all(ProjectPermission, filters={"user_id": user_id})
        project_ids = [perm.project_id for perm in user_perms]
        projects = [p for p in all_projects if p.id in project_ids]
        if projects:
            # Get first 3 projects
            projects = projects[:3]
            testcases = mongo_client.read("test_cases", {}) 
            logger.info(f"Fetched {len(projects)} recent projects for dashboard")
            logger.info(f"Len Testcases {len(testcases)}")
            
            if testcases:
                for testcase in testcases:
                    if testcase["compliance_reference_standard"]:
                        compliance_covered += 1
                compliance_coverage = int((compliance_covered / len(testcases)) * 100)
                testcases_generated = len(testcases)
                timesaved += round((len(testcases) * 5) / 60, 2)  # convert minutes to hours
                
            for project in projects:
                recent_projects.append({
                    "projectName": project.project_name,
                    "projectId": project.id,
                    "TestCasesGenerated": project.no_test_cases,
                    "description": project.description or "",
                    "UpdatedTime": Helper.time_saved_format(project.updated_at if project.updated_at else datetime.now()),
                    "status": "active" if project.no_test_cases > 0 else "review" if project.no_documents > 0 else "completed"
                })
        
        return JSONResponse(content={
            "TotalTestCasesGenerated": testcases_generated,
            "complianceCoveredTestCases": compliance_covered,
            "complianceCoverage": compliance_coverage,
            "timeSaved": timesaved,
            "recentProject": recent_projects
        })
                
    except Exception as exe:
        logger.error(f"Failed to fetch dashboard data: {exe}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/uploadFile")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """Upload and parse a file (requires authentication)"""
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
        return {"status": "error", "message": str(e)}


@router.get("/export/{project_id}")
async def export_test_cases(
    project_id: str,
    current_user: User = Depends(AuthManager.get_current_user)
):
    """Export test cases to CSV (requires authentication)"""
    try:
        # 1. Get project from SQLite using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Check user has permission
        user_id = current_user.id
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        if not user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "You don't have permission to export test cases for this project"}
            )

        # 2. Get test cases
        test_cases = list(mongo_client.read(testcase_collection, {"project_id": project_id}))
        if not test_cases:
            logger.warning(f"No test cases found for project id '{project_id}'")
            raise HTTPException(status_code=404, detail="No test cases found for this project")

        # 3. Normalize data for CSV
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

        # 4. Create DataFrame
        df = pd.DataFrame(data)

        # 5. Save to in-memory CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # 6. Stream CSV response
        filename = f"{project.project_name}_test_cases.csv"
        logger.info(f"Exporting {len(test_cases)} test cases for project '{project.project_name}'")
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException as he:
        raise he
    except Exception as exe:
        logger.error(f"Failed to export test cases for project id '{project_id}'")
        raise HTTPException(
            status_code=500,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/jiraPush")
async def push_jira(
    request: JiraPushRequest,
    current_user: User = Depends(AuthManager.get_current_user)
):
    """
    Push selected test cases from MongoDB to Jira (bulk mode with batching).
    Requires owner/editor permission for the project.
    """
    try:
        test_case_ids = request.selected_ids
        project_id = request.project_id
        domain_name = request.domain_name
        jira_api = request.jira_api
        user_id = current_user.id
        
        # Fetch project info from SQLite using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")
        
        # Check user has permission (owner or editor)
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        
        if not user_perms or user_perms[0].permission_level not in ['owner', 'editor']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "You don't have permission to push test cases to Jira for this project"}
            )

        # Jira API setup
        jira_url = f"https://{domain_name}.atlassian.net/rest/api/3/issue/bulk"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        all_issues = []
        logger.info("Fetching Test cases from DB")
        
        # Build issues payload for each test case
        for test_case_id in test_case_ids:
            test_case = mongo_client.read(testcase_collection, {"_id": ObjectId(test_case_id)})
            if not test_case:
                continue  # skip if not found
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
                    "project": {"key": getattr(project, "project_key", "HS")},  # fallback project key
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

        BATCH_SIZE = 45
        created_issues = []
        logger.info("Pushing Test cases to Jira")
        
        for i in range(0, len(all_issues), BATCH_SIZE):
            batch = all_issues[i:i+BATCH_SIZE]
            payload = {"issueUpdates": batch}
            logger.info(f"Pushing test cases batch to Jira batch size: {len(batch)}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    jira_url,
                    json=payload,
                    headers=headers,
                    auth=BasicAuth(request.jira_mail_id, jira_api),
                    timeout=30.0,
                )

            if response.status_code not in (200, 201):
                raise HTTPException(status_code=response.status_code, detail=response.text)

            created_issues.extend(response.json().get("issues", []))

            # Respect Jira API rate limits (1 request/sec safe zone)
            await asyncio.sleep(1)

        return {
            "status": "success",
            "total_pushed": len(created_issues),
            "created_issues": created_issues,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to push to Jira")
        raise HTTPException(status_code=500, detail=str(e))
