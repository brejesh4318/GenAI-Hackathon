import os
from urllib.request import Request
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
graph = GraphPipe(llm=llm)
testcase_generator =  TestCaseGenerator(graph = graph)



router = APIRouter(
    tags= ["Inference"],
    responses={status.HTTP_404_NOT_FOUND: {"description":"notfound"}}
)
logger.info("vectordb router is initialized")


@router.get("/")
async def root():
    return {"status":"healthy"}


@router.post("/testcase_generator")
async def process_testcases(request:Request, prd_id: str=Form(...), project_name: str=Form(...), file: UploadFile= File(...)) :
    
    try:
        content_bytes = await file.read()
        filename= f"{uuid.uuid4()}_{file.filename}"
        path = Helper.save_pdf("tmp", content=content_bytes, filename= filename)
        document_text = Helper.read_file(path)
        os.remove(path=path)
        test_cases = testcase_generator.generate_testcase(document_text=document_text)
        return {"test_cases": test_cases}

    except Exception as exe:
        os.remove(os.path.join("tmp", filename))
        raise HTTPException(status_code= status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "status": "Failed",
                        "message": exe.__str__(),
                    })


