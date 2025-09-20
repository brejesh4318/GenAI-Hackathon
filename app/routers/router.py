import os
import uuid
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, Request, Form, UploadFile, File
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt
# from passlib.context import CryptContext
from datetime import datetime, timedelta, UTC
from app.routers.datamodel import ProjectCreateRequest
from app.services.testcase_generator import TestCaseGenerator
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.helper import Helper
from app.services.llm_services.llm_factory import LlmFactory
from app.services.llm_services.graph_pipeline import GraphPipe
logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
llm = LlmFactory.get_llm(type=Constants.fetch_constant("llm_model")["model_name"])
llm_tools = LlmFactory.get_llm(type="gemini_2.5_flash")
graph_pipeline = GraphPipe(llm=llm, llm_tools=llm_tools)
testcase_generator =  TestCaseGenerator(graph_pipe = graph_pipeline)

mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("FIRESTORE_DB_URI"),
    db_name=Constants.fetch_constant("mongo_db")["db_name"],
    max_pool=Constants.fetch_constant("mongo_db")["max_pool_size"],
    server_selection_timeout=Constants.fetch_constant("mongo_db")["server_selection_timeout"]
)
logger.info("MongoDB client initialized")
project_collection = Constants.fetch_constant("mongo_collections")["projects_collection"]
testcase_collection = Constants.fetch_constant("mongo_collections")["testcases"]
router = APIRouter(
    tags= ["Inference"],
    responses={status.HTTP_404_NOT_FOUND: {"description":"notfound"}}
)
logger.info("vectordb router is initialized")


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
                compliance_coverage = (compliance_covered / len(testcases) * 100)
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
def createProject(request: ProjectCreateRequest):
    logger.info(f"Creating project: {request.project_name}")
    try:
        existing_projects = mongo_client.read(project_collection, {"project_name": request.project_name}, max_count=1)
        if existing_projects:
            logger.warning(f"Project with name '{request.project_name}' already exists")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail={
                                    "status": "Failed",
                                    "message": f"Project with name '{request.project_name}' already exists"
                                })
        project_doc = {
            "project_name": request.project_name,
            "description": request.description,
            "no_test_cases":0,
            "no_documents":0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        project_id = mongo_client.insert_one(project_collection, project_doc)
        logger.info(f"Inserted project doc into collection '{project_collection}' result: {project_id}")
        return {"status": "Success", "project_id": project_id}

    except Exception as exe:
        logger.exception("Failed to create project")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})
    
@router.get("/listProjects")
def listProjects():
    try:
        pass
    except:
        pass

@router.get("/getProjects")
def getProjects():
    try:
        projects = mongo_client.read("projects",query= {})
        response_projects = []
        for project in projects:
            compliances = []
            test_cases = mongo_client.read("test_cases", {"project_id": ObjectId(str(project["_id"]))})
            for test_case in test_cases:
                if test_case.get("compliance_reference_standard") and test_case.get("compliance_reference_standard") not in compliances:
                    compliances.append(test_case.get("compliance_reference_standard"))
            
            data = {
                "projectName": project["project_name"],
                "projectId": str(project["_id"]),
                "description": project.get("description", ""),
                "TestCasesGenerated": project.get("no_test_cases", 0),

                "documents": project.get("no_documents", 0),
                "ComplianceReferenceStandards": compliances,
                "status": "active" if project.get("no_test_cases", 0) > 0 else "review" if project.get("no_documents", 0) > 0 else "completed",
                "UpdatedTime": Helper.time_saved_format(project.get("updated_at", datetime.now()))
            }
            response_projects.append(data)
        logger.info(f"Fetched {len(response_projects)} projects")
        return JSONResponse(content={"projects": response_projects}, status_code=status.HTTP_200_OK)
    except Exception as exe:
        logger.exception("Failed to fetch project details")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})

@router.get("/getTestCases/{project_id}")
def getTestCases(project_id: str):
    try:
        project_details = mongo_client.read(project_collection, {"_id": ObjectId(project_id)}, max_count=1)
        if not project_details:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail={
                                    "status": "Failed",
                                    "message": f"Project with id '{project_id}' not found"
                                })
        else:
            response_testcases = []
            testcases = mongo_client.read(testcase_collection, {"project_id": project_id})
            if testcases:
                for testcase in testcases:
                    data = {
                            "testCaseId":testcase.get("test_case_id"),
                            "testCaseUniqueId": str(testcase["_id"]),
                            "priority": testcase.get("priority"),
                            "testCaseName": testcase.get("title"),
                            "requirement": testcase.get("feature"),
                            "steps": testcase.get("steps"),
                            "complianceTags": [testcase.get("compliance_reference_standard")]},
                    response_testcases.append(data)
            else:
                logger.info(f"No test cases found for project id: {project_id}")
            return JSONResponse(content={"projectId": project_id, 
                                         "projectName": project_details[0].get("project_name"),
                                         "test_cases": response_testcases}, status_code=status.HTTP_200_OK)
    except Exception as exe:
        logger.error(f"Failed to fetch test cases {exe}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"status": "Failed","message": str(exe),})

@router.get("/getTestCaseDetail/{testcase_id}")
def getTestCaseDetail(testcase_id: str):
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
                    "priority": testcase["priority"],
                    "status": "active",
                    "preconditions": testcase["preconditions"],
                    "testData": [{"field": i[0], 
                                  "value": i[1]} for i in testcase["test_data"].items()],
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

# @router.get("/compliancePage")
# def getCompliancePage():
#     try:
#         total_test_cases_generated = 0
#         compliance_test_cases= 0
#         compliance_coverage = 0
#         test_saved = 0
#         test_cases_standards = {}
#         testcases = mongo_client.read("testcases", {})
#         compliance_data = []
#         if testcases:
#             total_test_cases_generated = len(testcases)
#             for testcase in testcases:
#                 if testcase["compliance_reference_standard"]:
#                     data = {"project_id": str(testcase["project_id"]),
#                             "standard": testcase["compliance_reference_standard"],
#                             "clause": testcase["compliance_reference_clause"],
#                             "requirement": testcase["compliance_reference_clause"],
#                             "linkedTestCases":}
#             compliance_coverage = (compliance_test_cases / len(testcases) * 100)
#             test_saved = round((len(testcases) * 5) / 60, 2)  # convert minutes to hours, rounded to 2 decimals
#     except:
#         pass

@router.post("/testcaseGenerator")
async def process_testcases(
    request: Request,
    project_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
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

        path = Helper.save_pdf("tmp", content=content_bytes, filename=filename)
        logger.info(f"Saved uploaded file to temp path: {path}")

        # Fetch project by name
        existing_projects = mongo_client.read(
            project_collection,
            {"_id": ObjectId(project_id)},
            max_count=1
        )

        if not existing_projects:
            logger.warning(f"Project '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project '{project_id}' not found"
                }
            )

        project_id = str(existing_projects[0]["_id"])
        logger.info(f"Found existing project '{project_id}' with _id: {project_id}")

        # Generate test cases
        test_cases = testcase_generator.generate_testcase(document_path=path)
        logger.info( f"Generated {len(test_cases)} test cases for project: {project_id}")
        test_cases = [{"project_id": str(project_id), **tc} for tc in test_cases]
        # Insert test cases into collection
        testcases_collection = testcase_collection

        testcases_result = mongo_client.insert_many(testcases_collection, test_cases)
        logger.info(
            f"Inserted testcases into collection '{testcases_collection}' result: {testcases_result}"
        )
        updated_no_test_cases = existing_projects[-1]["no_test_cases"] + len(test_cases)
        updated_no_documents = existing_projects[-1]["no_documents"] + 1
        mongo_client.update_one(
            collection_name=Constants.fetch_constant("mongo_collections")["projects_collection"],
            query={"_id": ObjectId(project_id)},
            data={
                "no_test_cases": updated_no_test_cases,
                "no_documents": updated_no_documents,
                "updated_at": datetime.now()
            }
        )
        logger.info("Updated Project details in db")
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






