from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .legal_assistant import handle_query, create_chat_session
from fastapi.middleware.cors import CORSMiddleware
from src.legal_assistant.centralized_logger import CentralizedLogger
import time

logger = CentralizedLogger().get_logger()
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Reference(BaseModel):
    title: str
    url: str

class QueryRequest(BaseModel):
    chat_id: str = None
    query: str

class QueryResponse(BaseModel):
    chat_id: str
    response: str
    optimized_search: str
    references: List[Reference]

@app.post("/query", response_model=QueryResponse)
async def query_legal_assistant(request: QueryRequest):
    start_time = time.time()
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    chat_id = request.chat_id or create_chat_session()
    response = handle_query(chat_id, request.query)
    
    time_taken = time.time() - start_time
    logger.info(f"time taken for response : {time_taken}")
    
    optimized_search = response['optimized_search'] if response['optimized_search'] is not None else ''
    references = response['references'] if response['references'] is not None else []
    
    return QueryResponse(chat_id=chat_id, response=response['answer'], optimized_search=optimized_search, references=references)



@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy", "version": "6.0"}