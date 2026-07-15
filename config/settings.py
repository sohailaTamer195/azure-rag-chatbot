import os 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOUNDRY_API_KEY")
ENDPOINT = os.getenv("FOUNDRY_AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
 
