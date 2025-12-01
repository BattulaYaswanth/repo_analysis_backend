from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os,certifi

load_dotenv()

uri = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")  # (Renamed variable slightly for clarity)

# Determine whether to enable TLS options.
# In production (e.g. MongoDB Atlas) TLS is required and the CA bundle
# from `certifi` should be supplied. In local development a plain
# non-TLS Mongo instance is commonly used — forcing `tlsCAFile` there
# causes an SSL handshake failure (EOF). We support three heuristics:
#  - explicit `MONGO_TLS` env var (true/1/yes)
#  - `mongodb+srv://` URIs (SRV implies TLS)
#  - `tls=true` or `ssl=true` in the URI query

def _is_tls_required(uri: str | None) -> bool:
    if not uri:
        return False
    env_flag = os.getenv("MONGO_TLS", "").lower()
    if env_flag in ("1", "true", "yes"):
        return True
    if uri.startswith("mongodb+srv://"):
        return True
    # quick check for query flags
    lower = uri.lower()
    if "tls=true" in lower or "ssl=true" in lower:
        return True
    return False

# Build MongoClient kwargs conditionally
client_kwargs = {"server_api": ServerApi("1")}
if _is_tls_required(uri):
    client_kwargs.update({"tls": True, "tlsCAFile": certifi.where()})

# Create client with appropriate options for the environment
client = MongoClient(uri, **client_kwargs)

# 3. Fix: Use dictionary syntax client[DB_NAME] to use the actual string from your .env
db = client[DB_NAME]

collection = db["users"]
review_collection = db["review"]