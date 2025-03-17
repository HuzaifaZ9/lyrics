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
        "http://localhost:3000",  # For local development
        "*"  # As fallback during testing
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods for simplicity
    allow_headers=["*"],  # Allow all headers for simplicity
)

# Log available environment variables for debugging
logger.info(f"Available environment variables: {list(os.environ.keys())}")

# Get Groq API key with better logging
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
logger.info(f"Initial GROQ_API_KEY present: {GROQ_API_KEY is not None}")

# If the key contains template syntax, extract the actual value
# This handles variables in the format ${shared.VARIABLE_NAME}
if GROQ_API_KEY and "${" in GROQ_API_KEY:
    try:
        # Extract the actual variable name from ${shared.VARIABLE_NAME}
        var_path = GROQ_API_KEY.split("${")[1].split("}")[0]
        logger.info(f"Extracted variable path: {var_path}")
        
        # Handle "shared." prefix if present
        if "." in var_path:
            var_name = var_path.split(".")[1]
        else:
            var_name = var_path
            
        # Try to get the actual value from environment
        actual_key = os.environ.get(var_name)
        if actual_key:
            GROQ_API_KEY = actual_key
            logger.info(f"Successfully resolved variable to actual API key")
        else:
            logger.error(f"Could not find environment variable: {var_name}")
    except Exception as e:
        logger.error(f"Error parsing environment variable template: {str(e)}")

# Initialize Groq client
client = None
try:
    # Import Groq here to avoid import errors if the library isn't installed
    from groq import Groq
    
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        # Test the client with a minimal request to verify it works
        try:
            test_response = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="mixtral-8x7b-32768",
                max_tokens=5
            )
            logger.info("Groq client initialized and tested successfully")
        except Exception as test_error:
            logger.error(f"Groq API key validation failed: {str(test_error)}")
            client = None
    else:
        logger.error("GROQ_API_KEY environment variable is not set or is empty")
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

        # Enhanced check for client initialization
        if client is None:
            logger.error("Groq client not initialized when handling request")
            # Return a more detailed error message
            return {
                "lyrics": f"This is a test response for prompt: {request.prompt}\nStyle: {request.style}\nLength: {request.length}",
                "status": "test_mode",
                "error": "Groq client not initialized. Check API key configuration.",
                "debug_info": {
                    "api_key_present": GROQ_API_KEY is not None,
                    "api_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
                    "available_env_vars": list(os.environ.keys())
                }
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
        "groq_api_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
        "environment_variables": list(os.environ.keys())
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)