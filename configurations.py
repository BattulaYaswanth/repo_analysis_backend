from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import certifi
import sys # Added for cleaner exit

load_dotenv()

# --- Configuration and Validation ---
uri = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME") 

if not uri:
    # Use sys.exit or raise an error if critical configuration is missing
    print("FATAL: MONGO_URI environment variable is not set.", file=sys.stderr)
    sys.exit(1)
if not DB_NAME:
    print("FATAL: DB_NAME environment variable is not set.", file=sys.stderr)
    sys.exit(1)
# ------------------------------------


def _is_tls_required(uri: str | None) -> bool:
    """Checks if a TLS/SSL connection is required based on URI or environment flags."""
    if not uri:
        return False
    
    env_flag = os.getenv("MONGO_TLS", "").lower()
    if env_flag in ("1", "true", "yes"):
        return True
    
    # Standard Atlas connection string
    if uri.startswith("mongodb+srv://"):
        return True
    
    # Check for query flags
    lower = uri.lower()
    if "tls=true" in lower or "ssl=true" in lower:
        return True
        
    return False

# Build MongoClient kwargs conditionally
client_kwargs = {"server_api": ServerApi("1")}
if _is_tls_required(uri):
    # Conditionally add TLS configuration
    client_kwargs.update({"tls": True, "tlsCAFile": certifi.where()})

# Create the client object
client = MongoClient(uri, **client_kwargs)

# --- Connection Test ---
try:
    # The ping command is a lightweight way to verify connectivity
    client.admin.command('ping')
    print("MongoDB connection successful! Client and database objects initialized.")

except Exception as e:
    # If connection fails, log the error and exit cleanly
    print(f"FATAL: MongoDB connection failed. Error: {e}", file=sys.stderr)
    sys.exit(1)
# -----------------------

# Initialize database and collections
db = client[DB_NAME]
collection = db["users"]
review_collection = db["review"]

# You can now use 'collection' and 'review_collection' in your application logic.