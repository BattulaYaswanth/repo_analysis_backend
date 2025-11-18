from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
load_dotenv()
uri = os.getenv("MONGO_URI")
DB = os.getenv("DB_NAME")
client = MongoClient(uri,server_api=ServerApi("1"))
db=client.DB
collection = db["users"]
review_collection = db["review"]