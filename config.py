import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "healthcare_db")
COLLECTION_PRESCRIPTIONS = os.getenv("COLLECTION_PRESCRIPTIONS", "prescriptions")
COLLECTION_ALLERGIES = os.getenv("COLLECTION_ALLERGIES", "allergies")

# Rules Configuration
DUPLICATE_TOLERANCE_DAYS = int(os.getenv("DUPLICATE_TOLERANCE_DAYS", "30"))
PRESCRIPTION_HISTORY_DAYS = int(os.getenv("PRESCRIPTION_HISTORY_DAYS", "90"))

# Cerebras API Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "your_api_key_here")
CEREBRAS_API_URL = os.getenv("CEREBRAS_API_URL", "https://api.cerebras.ai/v1/chat/completions")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

# Headers for Cerebras API
CEREBRAS_API_HEADERS = {
    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
    "Content-Type": "application/json"
}

# Mock settings
MOCK_CEREBRAS = os.getenv("MOCK_CEREBRAS", "True").lower() in ("true", "1", "yes")

# Logging & Output Configuration
LOG_FILE = "ai_agent.log"
LOG_LEVEL = logging.INFO
OUTPUT_DIR = "output"
OUTPUT_FILE = "prescription_analysis.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Agent")
logger.info("Configuration and Logging successfully initialized")
