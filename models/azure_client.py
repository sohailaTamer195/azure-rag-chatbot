from openai import AzureOpenAI
from config.settings import API_KEY, ENDPOINT

if not API_KEY or not ENDPOINT:
    raise RuntimeError(
        "Missing Azure OpenAI credentials. Configure FOUNDRY_API_KEY and "
        "FOUNDRY_AZURE_OPENAI_ENDPOINT in Streamlit Cloud app secrets."
    )


client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-02-01"
)
