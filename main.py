from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
import uvicorn
import os
from dotenv import load_dotenv
from mangum import Mangum
import logging
#https://main.djtbx90lomjk6.amplifyapp.com
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lyrics Generator API",
    description="An API that generates song lyrics using AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if GROQ_API_KEY exists
if not os.getenv("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY not found in environment variables")
    raise Exception("GROQ_API_KEY not configured")

# Initialize Groq client with error handling
try:
    client = Groq(api_key="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")
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

class LyricsRequest(BaseModel):
    prompt: str
    style: Optional[str] = "modern"
    length: Optional[str] = "medium"

@app.get("/")
async def root():
    return {"message": "Welcome to the Lyrics Generator API"}

@app.post("/generate-lyrics")
async def generate_lyrics(request: LyricsRequest):
    try:
        logger.info(f"Received request with prompt: {request.prompt}")
        
        # Validate request
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": """You are a creative songwriter. Generate song lyrics based on the given prompt.
                Follow these guidelines:
                - Keep the tone consistent
                - Include metaphors and imagery
                - Create a clear structure with verses and chorus
                - Match the requested style and length"""
            },
            {
                "role": "user",
                "content": f"Write song lyrics about: {request.prompt}. Style: {request.style}. Length: {request.length}"
            }
        ]

        logger.info("Calling Groq API...")
        
        # Generate lyrics with error handling
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            logger.info("Successfully received response from Groq API")
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

        # Process response
        try:
            lyrics = completion.choices[0].message.content
            logger.info("Successfully processed lyrics")
        except Exception as e:
            logger.error(f"Error processing Groq response: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing response")

        return {
            "lyrics": lyrics.strip(),
            "status": "success"
        }

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# @app.post("/process-audio")
# async def process_audio():
#     # Your audio processing logic here
#     return {
#         "genre": "detected genre",
#         "lyrics": "Generated matching lyrics"
#     }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
