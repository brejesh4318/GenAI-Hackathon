import os
import uuid
from fastapi import APIRouter, HTTPException, status, Request, Form, UploadFile, File
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt
# from passlib.context import CryptContext
from datetime import datetime, timedelta, UTC
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
graph_pipeline = GraphPipe(llm=llm)
testcase_generator =  TestCaseGenerator(graph_pipe = graph_pipeline)

mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("FIRESTORE_DB_URI"),
    db_name=Constants.fetch_constant("mongo_db")["db_name"],
    max_pool=Constants.fetch_constant("mongo_db")["max_pool_size"],
    server_selection_timeout=Constants.fetch_constant("mongo_db")["server_selection_timeout"]
)
logger.info("MongoDB client initialized")

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
    return JSONResponse(content={
        "TotalTestCasesGenerated": 2847,
        "complianceCoveredTestCases": 1235,
        "complianceCoverage": 79,
        "timeSaved": 156,
        "recentProject": [
            {
                "projectName": "Cardiac Monitoring System",
                "projectId": "123456",
                "TestCasesGenerated": 234,
                "description": "FDA compliance test cases for cardiac monitor",
                "UpdatedTime": "2 hours",
                "status": "active"
            },
            {
                "projectName": "Patient Data Platform",
                "projectId": "123456",
                "TestCasesGenerated": 234,
                "description": "IEC 62304 software lifecycle compliance",
                "UpdatedTime": "1 day",
                "status": "review"
            },
            {
                "projectName": "Medical Device Integration",
                "projectId": "123456",
                "TestCasesGenerated": 234,
                "description": "ISO 13485 quality system validation",
                "UpdatedTime": "3 days",
                "status": "completed"
            }
        ]
    })

@router.post("/testcaseGenerator")
async def process_testcases(request:Request, project_name: str=Form(...), file: UploadFile= File(...)) :
    filename = None
    try:
        logger.info(f"Request from: {getattr(request.client, 'host', 'unknown')}, project: {project_name}, incoming file: {file.filename}")

        content_bytes = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"
        logger.info(f"Read uploaded file bytes: {len(content_bytes)} bytes, saving as: {filename}")

        path = Helper.save_pdf("tmp", content=content_bytes, filename=filename)
        logger.info(f"Saved uploaded file to temp path: {path}")

        # generate test cases
        test_cases = testcase_generator.generate_testcase(document_path=path)
        logger.info(f"Generated {len(test_cases) if hasattr(test_cases, '__len__') else 'unknown count'} test cases for project: {project_name}")

        # fetch existing project by name, else create new and use its _id
        project_collection = Constants.fetch_constant("mongo_collections")["projects_collection"]
        existing_projects = mongo_client.read(project_collection, {"project_name": project_name}, max_count=1)
        if existing_projects:
            project_id = str(existing_projects[0]["_id"])  # reuse existing project's _id
            logger.info(f"Found existing project '{project_name}' with _id: {project_id}")
        else:
            project_doc = {
                "project_name": project_name,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            project_id = mongo_client.insert_one(project_collection, project_doc)
            logger.info(f"Inserted project doc into collection '{project_collection}' result: {project_id}")

        # Insert test cases into test_cases collection
        testcases_collection = Constants.fetch_constant("mongo_collections")["testcases"]
        testcases_doc = {
            "project_id": project_id,
            "testcases": test_cases,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        testcases_result = mongo_client.insert_one(testcases_collection, testcases_doc)
        logger.info(f"Inserted testcases into collection '{testcases_collection}' result: {testcases_result}")

        return {"test_cases": test_cases}

    except Exception as exe:
        logger.exception("Failed to process testcase generation request")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={
                                "status": "Failed",
                                "message": str(exe),
                            })
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





