from pymongo import MongoClient
from app.config import MONGODB_URI, MONGODB_DATABASE

_client = None

def get_mongodb():
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client.get_database(MONGODB_DATABASE)
