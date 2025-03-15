from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .legal_assistant import handle_query, create_chat_session
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    chat_id: str = None
    query: str

class QueryResponse(BaseModel):
    chat_id: str
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_legal_assistant(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    chat_id = request.chat_id or create_chat_session()
    response = handle_query(chat_id, request.query)
    
    return QueryResponse(chat_id=chat_id, response=response)



@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy"}