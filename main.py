from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Update CORS middleware with specific origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://main.djtbx90lomjk6.amplifyapp.com"],  # Specific origin
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],  # Specify methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ... rest of your FastAPI code remains the same ...

# Initialize Groq client with error handling
try:
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    client = None

class LyricsRequest(BaseModel):
    prompt: str
    style: Optional[str] = "modern"
    length: Optional[str] = "medium"

@app.post("/generate-lyrics")
async def generate_lyrics(request: LyricsRequest):
    try:
        # Check if client is initialized
        if client is None:
            raise HTTPException(
                status_code=500,
                detail="Groq client not initialized. Please check server configuration."
            )

        logger.info(f"Received request with prompt: {request.prompt}")
        
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        messages = [
            {
                "role": "system",
                "content": """You are a creative songwriter. Generate song lyrics based on the given prompt."""
            },
            {
                "role": "user",
                "content": f"Write song lyrics about: {request.prompt}. Style: {request.style}. Length: {request.length}"
            }
        ]

        logger.info("Calling Groq API...")
        
        completion = client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            stream=False
        )

        lyrics = completion.choices[0].message.content
        logger.info("Successfully generated lyrics")

        return {
            "lyrics": lyrics.strip(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error in generate_lyrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
async def root():
    return {"message": "Welcome to the Lyrics Generator API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq_client": "initialized" if client is not None else "not initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
