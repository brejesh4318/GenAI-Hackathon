import asyncio
import base64
import io
import json
import os
import uuid
import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, Request, Form, UploadFile, File
from typing import List, Optional
from fastapi.responses import JSONResponse, StreamingResponse
from requests.auth import HTTPBasicAuth
import httpx
from httpx import BasicAuth

from datetime import datetime
from app.routers.datamodel import ProjectCreateRequest, VersionCreateRequest
from app.services.testcase_generator import TestCaseGenerator
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, Version
from app.utilities.helper import Helper
from app.services.llm_services.llm_factory import LlmFactory
from app.services.llm_services.graph_pipeline import GraphPipe, memory
from app.routers.datamodel import JiraPushRequest

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
router = APIRouter(
    tags= ["Inference"],
    responses={status.HTTP_404_NOT_FOUND: {"description":"notfound"}}
)
logger.info("router is initialized")

# def create_gcp_admin():
#     gcp_config = EnvironmentVariableRetriever.get_env_variable("GOOGLE_CRED")
#     gcp_dict = json.loads(base64.b64decode(gcp_config))
#     with open("gcp_admin.json", "w") as f:
#         json.dump(gcp_dict, f)
#     logger.info("GCP admin credentials written to gcp_admin.json")
#     return gcp_dict
# create_gcp_admin()

llm = LlmFactory.get_llm(type=Constants.fetch_constant("llm_model")["model_name"])
llm_tools = LlmFactory.get_llm(type="gemini_2.5_flash")
graph_pipeline = GraphPipe(llm=llm, llm_tools=llm_tools)
testcase_generator =  TestCaseGenerator(graph_pipe = graph_pipeline)

# MongoDB for test cases only
mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("FIRESTORE_DB_URI"),
    db_name=Constants.fetch_constant("mongo_db")["db_name"],
    max_pool=Constants.fetch_constant("mongo_db")["max_pool_size"],
    server_selection_timeout=Constants.fetch_constant("mongo_db")["server_selection_timeout"]
)
logger.info("MongoDB client initialized")

# SQLite for projects and versions
sqlite_config = Constants.fetch_constant("sqlite_db")
sqlite_client = SQLiteImplement(
    db_path=sqlite_config["db_path"],
    max_pool_size=sqlite_config["max_pool_size"]
)
logger.info("SQLite client initialized")

testcase_collection = Constants.fetch_constant("mongo_collections")["testcases"]


@router.get("/")
async def root():
    return {"status":"healthy"}


@router.get("/dashboardData")
async def get_dashboard_data():
    try:
        compliance_coverage = 0
        testcases_generated = 0
        compliance_covered = 0
        timesaved =0
        rescent_projects = []
        projects = mongo_client.read("projects", {}, max_count=3, sort=[("created_at", -1)])
        if projects:
            testcases = mongo_client.read("test_cases", {}) 
            logger.info(f"Fetched {len(projects)} recent projects for dashboard")
            logger.info(f"Len Testcases {len(testcases)}")
            if testcases:
                for testcase in testcases:
                    if testcase["compliance_reference_standard"]:
                        compliance_covered+=1
                compliance_coverage = int((compliance_covered / len(testcases)) * 100)
                testcases_generated = len(testcases)
                timesaved += round((len(testcases) * 5) / 60, 2)  # convert minutes to hours, rounded to 2 decimals
                
            for project in projects:
                rescent_projects.append({
                    "projectName": project["project_name"],
                    "projectId": str(project["_id"]),
                    "TestCasesGenerated": project.get("no_test_cases", 0),
                    "description": project.get("description", ""),
                    "UpdatedTime": Helper.time_saved_format(project.get("updated_at")),
                    "status": "active" if project.get("no_test_cases", 0) > 0 else "review" if project.get("no_documents", 0) > 0 else "completed"
                })
        return JSONResponse(content= {
                "TotalTestCasesGenerated": testcases_generated,
                "complianceCoveredTestCases": compliance_covered,
                "complianceCoverage": compliance_coverage,
                "timeSaved": timesaved,
                "recentProject": rescent_projects})
                
    except Exception as exe:
        logger.error(f"Failed to fetch dashboard data: {exe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})


@router.post("/createProject")
async def createProject(request: ProjectCreateRequest):
    logger.info(f"Creating project: {request.project_name}")
    try:
        # Check if project name already exists using ORM
        existing = sqlite_client.get_all(Project, filters={"project_name": request.project_name})
        
        if existing:
            logger.warning(f"Project with name '{request.project_name}' already exists")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail={
                                    "status": "Failed",
                                    "message": f"Project with name '{request.project_name}' already exists"
                                })
        
        # Create new project using ORM model
        new_project = Project(
            project_name=request.project_name,
            description=request.description,
            organization_id=request.organization_id,
            no_test_cases=0,
            no_documents=0
        )
        
        project_id = sqlite_client.create(new_project)
        
        logger.info(f"Created project in SQLite with UUID: {project_id}")
        return {"status": "Success", "project_id": project_id}

    except Exception as exe:
        logger.exception("Failed to create project")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})


@router.post("/createVersion")
async def create_version(request: VersionCreateRequest):
    """Create a new version under a project"""
    logger.info(f"Creating version: {request.version_name} for project: {request.project_id}")
    try:
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, request.project_id)
        
        if not project:
            logger.warning(f"Project with id '{request.project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project with id '{request.project_id}' not found"
                }
            )
        
        # Check if version name already exists for this project
        existing_versions = sqlite_client.get_all(
            Version, 
            filters={"project_id": request.project_id, "version_name": request.version_name}
        )
        
        if existing_versions:
            logger.warning(f"Version '{request.version_name}' already exists for project '{request.project_id}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "Failed",
                    "message": f"Version '{request.version_name}' already exists for this project"
                }
            )
        
        # Create new version using ORM model
        new_version = Version(
            project_id=request.project_id,
            version_name=request.version_name,
            description=request.description,
            is_active=request.is_active if request.is_active is not None else True,
            no_documents=0,
            no_test_cases=0
        )
        
        version_id = sqlite_client.create(new_version)
        
        logger.info(f"Created version in SQLite with UUID: {version_id}")
        return {
            "status": "Success",
            "version_id": version_id,
            "message": f"Version '{request.version_name}' created successfully"
        }
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to create version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/getVersions/{project_id}")
async def get_versions(project_id: str):
    """Get all versions for a project"""
    logger.info(f"Fetching versions for project: {project_id}")
    try:
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project with id '{project_id}' not found"
                }
            )
        
        # Get all versions for this project using ORM
        versions = sqlite_client.get_all(Version, filters={"project_id": project_id}, order_by=Version.created_at.desc())
        
        response_versions = []
        for version in versions:
            version_data = {
                "versionId": version.id,
                "versionName": version.version_name,
                "description": version.description or "",
                "isActive": version.is_active,
                "noDocuments": version.no_documents,
                "noTestCases": version.no_test_cases,
                "createdAt": version.created_at.isoformat() if version.created_at else None,
                "updatedAt": version.updated_at.isoformat() if version.updated_at else None
            }
            response_versions.append(version_data)
        
        logger.info(f"Fetched {len(response_versions)} versions for project {project_id}")
        return JSONResponse(
            content={
                "status": "Success",
                "projectId": project_id,
                "projectName": project.project_name,
                "versions": response_versions
            },
            status_code=status.HTTP_200_OK
        )
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch versions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/getVersionDetail/{version_id}")
async def get_version_detail(version_id: str):
    """Get detailed information about a specific version"""
    logger.info(f"Fetching details for version: {version_id}")
    try:
        # Get version with project info using JOIN
        query = """
            SELECT v.*, p.project_name 
            FROM versions v 
            JOIN projects p ON v.project_id = p.id 
            WHERE v.id = ?
        """
        version = sqlite_client.fetch_one(query, (version_id,))
        
        if not version:
            logger.warning(f"Version with id '{version_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Version with id '{version_id}' not found"
                }
            )
        
        version_detail = {
            "versionId": version["id"],
            "projectId": version["project_id"],
            "projectName": version["project_name"],
            "versionName": version["version_name"],
            "description": version.get("description", ""),
            "isActive": bool(version.get("is_active", 1)),
            "noDocuments": version.get("no_documents", 0),
            "noTestCases": version.get("no_test_cases", 0),
            "createdAt": version["created_at"],
            "updatedAt": version["updated_at"]
        }
        
        return JSONResponse(
            content={"status": "Success", "version": version_detail},
            status_code=status.HTTP_200_OK
        )
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch version details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.put("/updateVersion/{version_id}")
async def update_version(version_id: str, request: VersionCreateRequest):
    """Update version details"""
    logger.info(f"Updating version: {version_id}")
    try:
        # Check if version exists using ORM
        version = sqlite_client.get_by_id(Version, version_id)
        
        if not version:
            logger.warning(f"Version with id '{version_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Version with id '{version_id}' not found"
                }
            )
        
        # Update version using ORM
        update_data = {
            "version_name": request.version_name,
            "description": request.description,
            "is_active": request.is_active if request.is_active is not None else True
        }
        sqlite_client.update(Version, version_id, update_data)
        
        logger.info(f"Updated version {version_id}")
        return {
            "status": "Success",
            "message": f"Version updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to update version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )

@router.get("/getProjects")
async def getProjects():
    try:
        # Get all projects from SQLite using ORM
        projects = sqlite_client.get_all(Project, order_by=Project.created_at.desc())
        
        response_projects = []
        for project in projects:
            compliances = []
            # Get test cases from MongoDB using SQL project_id (UUID)
            test_cases = mongo_client.read("test_cases", {"project_id": project.id})
            for test_case in test_cases:
                if test_case.get("compliance_reference_standard") and test_case.get("compliance_reference_standard") not in compliances:
                    compliances.append(test_case.get("compliance_reference_standard"))
            
            # Get version count using ORM
            version_count = sqlite_client.count(Version, filters={"project_id": project.id})
            
            data = {
                "projectName": project.project_name,
                "projectId": project.id,
                "description": project.description or "",
                "TestCasesGenerated": project.no_test_cases,
                "documents": project.no_documents,
                "versions": version_count,
                "ComplianceReferenceStandards": compliances,
                "status": "active" if project.no_test_cases > 0 else "review" if project.no_documents > 0 else "completed",
                "UpdatedTime": Helper.time_saved_format(project.updated_at if project.updated_at else datetime.now())
            }
            response_projects.append(data)
        
        logger.info(f"Fetched {len(response_projects)} projects")
        return JSONResponse(content=response_projects, status_code=status.HTTP_200_OK)
    except Exception as exe:
        logger.error(f"Failed to fetch projects: {exe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed", "message": str(exe)})
@router.get("/getTestCases/{project_id}")
async def getTestCases(project_id: str, version_id: Optional[str] = None):
    try:
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail={
                                    "status": "Failed",
                                    "message": f"Project with id '{project_id}' not found"
                                })
        
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
        # Get test cases from MongoDB (still using MongoDB for flexible test case storage)
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
    except Exception as exe:
        logger.error(f"Failed to fetch test cases {exe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})

@router.get("/getTestCaseDetail/{testcase_id}")
async def getTestCaseDetail(testcase_id: str):
    try:
        testcase = mongo_client.read(testcase_collection, {"_id": ObjectId(testcase_id)}, max_count=1)
        if not testcase:
            logger.warning(f"Test case with id '{testcase_id}' not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail={
                                    "status": "Failed",
                                    "message": f"Test case with id '{testcase_id}' not found"
                                })
        else:
            testcase = testcase[0]
            data ={
                    "id": testcase["test_case_id"],
                    "featureModule": testcase["feature"],
                    "title": testcase["title"], 
                    "type": testcase["type"],
                    "reqId": testcase["requirement_id"],
                    "reqDesc": testcase["requirement_description"],
                    "priority": testcase["priority"],
                    "status": "active",
                    "preconditions": testcase["preconditions"],
                    "testData": [{"field": i[0], 
                                  "value": i[1]} for i in testcase["test_data"].items()] if testcase["test_data"] else [],
                    "stepsToExecute": testcase["steps"],
                    "expectedResults": testcase["expected_result"],
                    "postconditions": testcase["postconditions"],
                    "complianceStandard": testcase["compliance_reference_standard"],
                    "complianceClause": testcase["compliance_reference_clause"],
                    "complianceRequirementText": testcase["compliance_reference_requirement_text"]
                    }
            return JSONResponse(content={"test_case": data}, status_code=status.HTTP_200_OK)
    except Exception as exe:
        logger.exception("Failed to fetch test case details")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})

@router.post("/testcaseGenerator/{request_type}/{project_type}")
async def generate_testcases(
    request: Request,
    project_type: str,
    request_type: str,
    project_id: str = Form(...),
    version_id: Optional[str] = Form(None),
    command: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    # Fetch project using ORM
    filename=None
    project = sqlite_client.get_by_id(Project, project_id)

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
    if request_type =="resume":
        logger.info(f"Resuming testcase generation for project: {project_id}")
        test_cases = await asyncio.to_thread(testcase_generator.generate_testcase, document_path=None, project_id=project_id, invoke_type=request_type, invoke_command=command)
    else:
        logger.info(f"Starting new testcase generation for project: {project_id}")
        memory.delete_thread(thread_id=project_id)
        filename = None
        file = files[0] ##TODO handle multiple files
        try:
            logger.info(
                f"Request from: {getattr(request.client, 'host', 'unknown')}, "
                f"project: {project_id}, incoming file: {file.filename}"
            )

            # Read uploaded file
            content_bytes = await file.read()
            filename = f"{uuid.uuid4()}_{file.filename}"
            logger.info(
                f"Read uploaded file bytes: {len(content_bytes)} bytes, saving as: {filename}"
            )

            path = Helper.save_file("/tmp", content=content_bytes, filename=filename)
            logger.info(f"Saved uploaded file to temp path: {path}")


            # Generate test cases
            test_cases = await asyncio.to_thread(testcase_generator.generate_testcase, document_path=path, project_id=project_id, invoke_type=request_type, invoke_command=command, project_type=project_type)
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
        logger.info( f"Generated {len(test_cases)} test cases for project: {project_id}")
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
        # ensure temp file cleanup
        try:
            if filename:
                tmp_path = os.path.join("tmp", filename)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    logger.info(f"Removed temp file: {tmp_path}")
        except Exception:
            logger.exception("Failed to remove temp file in finally block")

@router.post("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
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
async def export_test_cases(project_id: str):
    try:
        # 1. Get project from SQLite using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(status_code=404, detail="Project not found")

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
async def push_jira(request: JiraPushRequest):
    """
    Push selected test cases from MongoDB to Jira (bulk mode with batching).
    Each test case is fetched from the database and formatted before sending.
    """
    try:
        test_case_ids = request.selected_ids
        project_id = request.project_id
        domain_name = request.domain_name
        jira_api = request.jira_api
        
        # Fetch project info from SQLite using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")

        # Jira API setup
        jira_url = f"https://{domain_name}.atlassian.net/rest/api/3/issue/bulk"
        auth = HTTPBasicAuth(request.jira_mail_id, jira_api)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        all_issues = []
        logger.info("Fetching Test cases frm Db")
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
        logger.info("Pushing Test cases to jira")
        for i in range(0, len(all_issues), BATCH_SIZE):
            batch = all_issues[i:i+BATCH_SIZE]
            payload = {"issueUpdates": batch}
            logger.info(f"Pushing test cases batch to Jira batch size: {len(batch)}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    jira_url,
                    json=payload,
                    headers=headers,
                    auth=BasicAuth("fundauraap@gmail.com", jira_api),
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
