from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from typing import Optional
import os
import logging
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment from .env file")
except ImportError:
    logger.info("dotenv not installed, skipping .env loading")

app = FastAPI()

# CORS middleware with expanded configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main.djtbx90lomjk6.amplifyapp.com",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
logger.info(f"GROQ_API_KEY present: {GROQ_API_KEY is not None}")

# Initialize Groq client with version compatibility
client = None
groq_error_message = None
try:
    # Try to import the Groq library
    from groq import Groq
    logger.info("Successfully imported Groq library")
    
    if GROQ_API_KEY:
        try:
            # Initialize client with just the API key - no extra parameters
            # This should work with all versions of the Groq library
            client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized successfully")
            
            # Test the client
            test_response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="mixtral-8x7b-32768",
                max_tokens=5
            )
            logger.info("Groq client test successful")
        except Exception as e:
            logger.error(f"Error initializing or testing Groq client: {str(e)}")
            groq_error_message = str(e)
            client = None
    else:
        groq_error_message = "API key not found"
        logger.error("GROQ_API_KEY environment variable is not set")
except ImportError as e:
    groq_error_message = f"Failed to import Groq library: {str(e)}"
    logger.error(groq_error_message)
except Exception as e:
    groq_error_message = f"Unexpected error: {str(e)}"
    error_trace = traceback.format_exc()
    logger.error(f"Groq client setup failed: {str(e)}\n{error_trace}")

class LyricsRequest(BaseModel):
    prompt: str
    style: Optional[str] = "modern"
    length: Optional[str] = "medium"

@app.post("/generate-lyrics")
async def generate_lyrics(request: LyricsRequest):
    try:
        # Log the request
        logger.info(f"Received lyrics generation request with prompt: '{request.prompt}'")
        
        # Check if prompt is provided
        if not request.prompt:
            logger.warning("Empty prompt received")
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Check if client is initialized
        if client is None:
            logger.error("Groq client not initialized when handling request")
            return {
                "lyrics": f"This is a test response for prompt: {request.prompt}\nStyle: {request.style}\nLength: {request.length}",
                "status": "test_mode",
                "error": f"Groq client not initialized: {groq_error_message}"
            }

        # Prepare the messages for Groq
        messages = [
            {
                "role": "system",
                "content": "You are a creative songwriter. Generate song lyrics based on the given prompt."
            },
            {
                "role": "user",
                "content": f"Write song lyrics about: {request.prompt}. Style: {request.style}. Length: {request.length}"
            }
        ]

        logger.info("Calling Groq API...")
        
        # Make the API call with try-except
        try:
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
            
            # Return the formatted response
            return {
                "lyrics": lyrics.strip(),
                "status": "success"
            }
            
        except Exception as groq_error:
            logger.error(f"Error calling Groq API: {str(groq_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating lyrics: {str(groq_error)}"
            )

    except HTTPException as http_ex:
        # Re-raise HTTP exceptions as they are already formatted correctly
        raise
    except Exception as e:
        # Catch all other exceptions with detailed logging
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in generate_lyrics: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to the Lyrics Generator API",
        "endpoints": {
            "/": "This information",
            "/health": "API health status",
            "/generate-lyrics": "POST endpoint for lyrics generation"
        }
    }

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "groq_client": "initialized" if client is not None else "not initialized",
        "groq_error": groq_error_message
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)