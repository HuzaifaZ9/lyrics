from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import logging
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware with expanded configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main.djtbx90lomjk6.amplifyapp.com",
        "http://localhost:3000",  # For local development
        "*"  # As fallback during testing
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods for simplicity
    allow_headers=["*"],  # Allow all headers for simplicity
)

# Get Groq API key from environment with better error handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Debug environment variables
logger.info(f"GROQ_API_KEY present: {GROQ_API_KEY is not None}")
logger.info(f"Environment variables: {list(os.environ.keys())}")

# Initialize Groq client with detailed error handling
client = None
try:
    # Import Groq here to avoid import errors if the library isn't installed
    from groq import Groq
    
    # Try multiple environment variable names that might be used in deployment
    if not GROQ_API_KEY:
        potential_keys = ["GROQ_API_KEY", "GROQ_KEY", "GROQ_TOKEN", "GROQAPIKEY"]
        for key in potential_keys:
            potential_value = os.environ.get(key)
            if potential_value:
                GROQ_API_KEY = potential_value
                logger.info(f"Found API key in environment variable: {key}")
                break
    
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        # Test the client with a simple call to ensure it's working
        test_messages = [{"role": "user", "content": "Hello"}]
        try:
            test_response = client.chat.completions.create(
                messages=test_messages,
                model="mixtral-8x7b-32768",
                max_tokens=10
            )
            logger.info("Groq client successfully tested")
        except Exception as test_error:
            logger.error(f"Groq client test failed: {str(test_error)}")
            client = None
    else:
        logger.error("GROQ_API_KEY environment variable is not set")
except ImportError:
    logger.error("Failed to import Groq. Please install with 'pip install groq'")
except Exception as e:
    error_trace = traceback.format_exc()
    logger.error(f"Failed to initialize Groq client: {str(e)}\n{error_trace}")

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

        # Enhanced client check with more detailed error message
        if client is None:
            logger.error("Groq client not initialized when handling request")
            return {
                "lyrics": f"This is a test response for prompt: {request.prompt}\nStyle: {request.style}\nLength: {request.length}",
                "status": "test_mode",
                "error": "Groq client not initialized. Please check your GROQ_API_KEY environment variable."
            }

        # Prepare the messages for Groq
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
        "groq_api_key_present": GROQ_API_KEY is not None,
        "environment_variables": list(os.environ.keys())
    }

# Add a debug endpoint to check environment variables
@app.get("/debug")
async def debug_endpoint():
    logger.info("Debug endpoint accessed")
    # Only show that variables exist, not their values for security
    return {
        "environment_variables": list(os.environ.keys()),
        "groq_api_key_present": GROQ_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)