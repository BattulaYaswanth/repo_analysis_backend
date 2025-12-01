from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os,certifi

load_dotenv()

uri = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")  # (Renamed variable slightly for clarity)

# 2. Add tlsCAFile=certifi.where() to fix the SSL Handshake Error
client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())

# 3. Fix: Use dictionary syntax client[DB_NAME] to use the actual string from your .env
db = client[DB_NAME]

collection = db["users"]
review_collection = db["review"]