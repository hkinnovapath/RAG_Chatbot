# from fastapi import FastAPI
# from response_gen import generate_response

# app = FastAPI()

# @app.post("/chat")
# async def chat(query: str):
#     response = generate_response(query)
#     return {"response": response}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.response_gen import generate_response

app = FastAPI()

# CORS setup - uncomment and make sure it's configured
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can replace "*" with specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/chat")
async def chat(query: str):
    response = generate_response(query)  # Replace with your actual response generation logic
    return {"response": response}
