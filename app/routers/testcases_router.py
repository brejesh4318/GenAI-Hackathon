"""
Test case management endpoints
"""
import asyncio
import os
import uuid
from typing import List, Optional
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, Request, Form, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from app.services.testcase_generator import TestCaseGenerator
from app.utilities.auth_helper import AuthManager
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, Version, ProjectPermission, User
from app.utilities.helper import Helper
from app.services.llm_services.llm_factory import LlmFactory
from app.services.llm_services.graph_pipeline import GraphPipe, memory

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    prefix="/testcases",
    tags=["TestCases"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "not found"}}
)

# Initialize LLM and graph pipeline
llm = LlmFactory.get_llm(type=Constants.fetch_constant("llm_model")["model_name"])
llm_tools = LlmFactory.get_llm(type="gemini_2.5_flash")
graph_pipeline = GraphPipe(llm=llm, llm_tools=llm_tools)
testcase_generator = TestCaseGenerator(graph_pipe=graph_pipeline)

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


@router.get("/project/{project_id}")
async def get_testcases(
    project_id: str,
    version_id: Optional[str] = None,
    current_user: User = Depends(AuthManager.get_current_user)
):
    """Get all test cases for a project (requires authentication)"""
    try:
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        
        # Check user has permission
        user_id = current_user.id
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        if not user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "Failed",
                    "message": "You don't have permission to view test cases for this project"
                }
            )
        
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project with id '{project_id}' not found"
                }
            )
        
        # Build MongoDB query based on whether version_id is provided
        query = {"project_id": project_id}
        if version_id:
            query["version_id"] = version_id
            # Get version details using ORM
            version = sqlite_client.get_by_id(Version, version_id)
            version_name = version.version_name if version else None
        else:
            version_name = None
        
        response_testcases = []
        # Get test cases from MongoDB
        testcases = mongo_client.read(testcase_collection, query)
        if testcases:
            for testcase in testcases:
                data = {
                    "testUniqueID": str(testcase.get("_id", "")),
                    "testCaseId": testcase.get("test_case_id"),
                    "testCaseUniqueId": str(testcase["_id"]),
                    "priority": testcase.get("priority"),
                    "testCaseName": testcase.get("title"),
                    "requirement": testcase.get("feature"),
                    "steps": testcase.get("steps"),
                    "complianceTags": [testcase.get("compliance_reference_standard")]
                }
                response_testcases.append(data)
        else:
            logger.info(f"No test cases found for project id: {project_id}" + (f", version id: {version_id}" if version_id else ""))
        
        response = {
            "projectId": project_id, 
            "projectName": project.project_name,
            "test_cases": response_testcases
        }
        
        if version_name:
            response["versionName"] = version_name
            response["versionId"] = version_id
        
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.error(f"Failed to fetch test cases {exe}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/{testcase_id}")
async def get_testcase_detail(
    testcase_id: str,
    current_user: User = Depends(AuthManager.get_current_user)
):
    """Get detailed information about a specific test case (requires authentication)"""
    user_id = current_user.id
    try:
        testcase = mongo_client.read(testcase_collection, {"_id": ObjectId(testcase_id)}, max_count=1)
        if not testcase:
            logger.warning(f"Test case with id '{testcase_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Test case with id '{testcase_id}' not found"
                }
            )
        
        testcase = testcase[0]
        
        # Check user has permission to view project
        project_id = testcase.get("project_id")
        if project_id:
            user_perms = sqlite_client.get_all(
                ProjectPermission,
                filters={"project_id": project_id, "user_id": user_id}
            )
            if not user_perms:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "status": "Failed",
                        "message": "You don't have permission to view this test case"
                    }
                )
        
        if testcase:
            data = {
                "id": testcase["test_case_id"],
                "featureModule": testcase["feature"],
                "title": testcase["title"], 
                "type": testcase["type"],
                "reqId": testcase["requirement_id"],
                "reqDesc": testcase["requirement_description"],
                "priority": testcase["priority"],
                "status": "active",
                "preconditions": testcase["preconditions"],
                "testData": [{"field": i[0], "value": i[1]} for i in testcase["test_data"].items()] if testcase["test_data"] else [],
                "stepsToExecute": testcase["steps"],
                "expectedResults": testcase["expected_result"],
                "postconditions": testcase["postconditions"],
                "complianceStandard": testcase["compliance_reference_standard"],
                "complianceClause": testcase["compliance_reference_clause"],
                "complianceRequirementText": testcase["compliance_reference_requirement_text"]
            }
            return JSONResponse(content={"test_case": data}, status_code=status.HTTP_200_OK)
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch test case details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/generate/{request_type}/{project_type}")
async def generate_testcases(
    request: Request,
    project_type: str,
    request_type: str,
    project_id: str = Form(...),
    version_id: Optional[str] = Form(None),
    command: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    current_user: User = Depends(AuthManager.get_current_user)
):
    """Generate test cases from document upload (requires owner/editor permission)"""
    filename = None
    user_id = current_user.id
    
    # Fetch project using ORM
    project = sqlite_client.get_by_id(Project, project_id)
    
    # Check user has permission (owner or editor)
    user_perms = sqlite_client.get_all(
        ProjectPermission,
        filters={"project_id": project_id, "user_id": user_id}
    )
    
    if not user_perms or user_perms[0].permission_level not in ['owner', 'editor']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "Failed",
                "message": "You don't have permission to generate test cases for this project"
            }
        )

    if not project:
        logger.warning(f"Project '{project_id}' not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "status": "Failed",
                "message": f"Project '{project_id}' not found"
            }
        )

    logger.info(f"Found existing project '{project.project_name}' with UUID: {project_id}")
    
    # Validate version if provided
    if version_id:
        version = sqlite_client.get_by_id(Version, version_id)
        if not version or version.project_id != project_id:
            logger.warning(f"Version '{version_id}' not found for project '{project_id}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Version not found for this project"
                }
            )
        logger.info(f"Using version: {version.version_name}")
    
    if request_type == "resume" and not command:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "Failed",
                "message": "Missing 'command' form data for resume request type"
            }
        )
    
    if request_type == "resume":
        logger.info(f"Resuming testcase generation for project: {project_id}")
        test_cases = await asyncio.to_thread(
            testcase_generator.generate_testcase, 
            document_path=None, 
            project_id=project_id, 
            invoke_type=request_type, 
            invoke_command=command
        )
    else:
        logger.info(f"Starting new testcase generation for project: {project_id}")
        memory.delete_thread(thread_id=project_id)
        filename = None
        file = files[0]  # TODO: handle multiple files
        
        try:
            logger.info(
                f"Request from: {getattr(request.client, 'host', 'unknown')}, "
                f"project: {project_id}, incoming file: {file.filename}"
            )

            # Read uploaded file
            content_bytes = await file.read()
            filename = f"{uuid.uuid4()}_{file.filename}"
            logger.info(f"Read uploaded file bytes: {len(content_bytes)} bytes, saving as: {filename}")

            path = Helper.save_file("/tmp", content=content_bytes, filename=filename)
            logger.info(f"Saved uploaded file to temp path: {path}")

            # Generate test cases
            test_cases = await asyncio.to_thread(
                testcase_generator.generate_testcase, 
                document_path=path, 
                project_id=project_id, 
                invoke_type=request_type, 
                invoke_command=command, 
                project_type=project_type
            )
        except Exception as exe:
            logger.exception("Failed to process testcase generation request")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "Failed",
                    "message": str(exe),
                }
            )
    
    try:
        if test_cases["type"] == "interrupt":
            logger.info("Test case generation interrupted for user input")
            return JSONResponse(
                content={
                    "status": "interrupt",
                    "message": test_cases["response"]
                },
                status_code=status.HTTP_200_OK  
            )
        
        test_cases = test_cases["response"]
        logger.info(f"Generated {len(test_cases)} test cases for project: {project_id}")
        
        # Add project_id and version_id to test cases
        test_cases = [{"project_id": str(project_id), "version_id": str(version_id) if version_id else None, **tc} for tc in test_cases]
        
        # Insert test cases into MongoDB
        testcases_result = mongo_client.insert_many(testcase_collection, test_cases)
        logger.info(f"Inserted {len(test_cases)} testcases into MongoDB")
        
        # Update project counts using ORM
        update_data = {
            "no_test_cases": project.no_test_cases + len(test_cases),
            "no_documents": project.no_documents + 1
        }
        sqlite_client.update(Project, project_id, update_data)
        
        # Update version counts if version_id provided
        if version_id:
            version_update_data = {
                "no_test_cases": version.no_test_cases + len(test_cases),
                "no_documents": version.no_documents + 1
            }
            sqlite_client.update(Version, version_id, version_update_data)
        
        logger.info("Updated Project/Version details in database")
        return JSONResponse(content={"status": "success"}, status_code=status.HTTP_200_OK)

    except Exception as exe:
        logger.exception("Failed to process testcase generation request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "Failed",
                "message": str(exe),
            }
        )
    finally:
        # Ensure temp file cleanup
        try:
            if filename:
                tmp_path = os.path.join("tmp", filename)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    logger.info(f"Removed temp file: {tmp_path}")
        except Exception:
            logger.exception("Failed to remove temp file in finally block")
