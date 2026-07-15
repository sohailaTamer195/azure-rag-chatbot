from openai import AzureOpenAI
from config.settings import API_KEY, ENDPOINT

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-02-01"
)
