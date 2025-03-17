from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# Get Groq API key with better logging
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
logger.info(f"GROQ_API_KEY present: {GROQ_API_KEY is not None}")
logger.info(f"GROQ_API_KEY length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0}")

# Only show first/last few characters of API key for debugging (without exposing the full key)
if GROQ_API_KEY and len(GROQ_API_KEY) > 10:
    masked_key = f"{GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:]}"
    logger.info(f"GROQ_API_KEY format check: {masked_key}")

# Initialize Groq client with very detailed error handling
client = None
groq_error_message = None
try:
    # Try to import the Groq library
    logger.info("Attempting to import Groq library...")
    try:
        from groq import Groq
        logger.info("Successfully imported Groq library")
    except ImportError as imp_err:
        logger.error(f"Failed to import Groq library: {str(imp_err)}")
        groq_error_message = f"ImportError: {str(imp_err)}"
        raise
    
    # Check if API key is available
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set")
        groq_error_message = "API key not found in environment variables"
        raise ValueError("API key not found in environment variables")
    
    # Try to initialize the client
    logger.info("Initializing Groq client...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as init_err:
        logger.error(f"Failed to initialize Groq client: {str(init_err)}")
        groq_error_message = f"Client initialization error: {str(init_err)}"
        raise
    
    # Test the client with a minimal request
    logger.info("Testing Groq client with a minimal request...")
    try:
        test_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="mixtral-8x7b-32768",
            max_tokens=5
        )
        logger.info("Groq client test successful")
    except Exception as test_err:
        logger.error(f"Groq client test failed: {str(test_err)}")
        groq_error_message = f"API test error: {str(test_err)}"
        client = None
        raise
        
except Exception as e:
    error_trace = traceback.format_exc()
    logger.error(f"Groq client setup failed: {str(e)}\n{error_trace}")
    if not groq_error_message:
        groq_error_message = f"Unexpected error: {str(e)}"

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

        # Enhanced check for client initialization
        if client is None:
            logger.error("Groq client not initialized when handling request")
            # Return a more detailed error message
            return {
                "lyrics": f"This is a test response for prompt: {request.prompt}\nStyle: {request.style}\nLength: {request.length}",
                "status": "test_mode",
                "error": f"Groq client not initialized: {groq_error_message}",
                "debug_info": {
                    "api_key_present": GROQ_API_KEY is not None,
                    "api_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
                    "api_key_format": f"{GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:]}" if GROQ_API_KEY and len(GROQ_API_KEY) > 8 else "N/A"
                }
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
        "groq_error": groq_error_message,
        "api_key_present": GROQ_API_KEY is not None,
        "api_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
        "api_key_format": f"{GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:]}" if GROQ_API_KEY and len(GROQ_API_KEY) > 8 else "N/A"
    }

# Add direct API key testing endpoint
@app.get("/test-api-key")
async def test_api_key():
    logger.info("Test API key endpoint accessed")
    
    # Get the API key from environment again (in case it changed)
    current_key = os.environ.get("GROQ_API_KEY")
    
    # Create a temporary client for testing
    temp_client = None
    test_result = "failed"
    error_message = "No API key found"
    
    try:
        if current_key:
            from groq import Groq
            temp_client = Groq(api_key=current_key)
            
            # Test with minimal request
            test_response = temp_client.chat.completions.create(
                messages=[{"role": "user", "content": "Test"}],
                model="mixtral-8x7b-32768",
                max_tokens=5
            )
            
            test_result = "success"
            error_message = None
    except Exception as e:
        error_message = str(e)
    
    return {
        "test_result": test_result,
        "error_message": error_message,
        "api_key_present": current_key is not None,
        "api_key_length": len(current_key) if current_key else 0,
        "api_key_format": f"{current_key[:4]}...{current_key[-4:]}" if current_key and len(current_key) > 8 else "N/A"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)