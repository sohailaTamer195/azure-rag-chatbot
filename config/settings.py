import os
from dotenv import load_dotenv

load_dotenv()

try:
	import streamlit as st
except ImportError:  # Allows non-Streamlit tooling to import this module.
	st = None


def _get_setting(name: str) -> str | None:
	value = os.getenv(name)
	if value:
		return value

	if st is not None:
		try:
			secrets = st.secrets
			value = secrets.get(name)
			if value:
				return value

			# Also support secrets pasted under a [default] TOML section.
			default_secrets = secrets.get("default", {})
			return default_secrets.get(name)
		except (FileNotFoundError, KeyError, TypeError):
			# Streamlit secrets are unavailable outside a configured app.
			pass

	return None


API_KEY = _get_setting("FOUNDRY_API_KEY")
ENDPOINT = _get_setting("FOUNDRY_AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = _get_setting("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = _get_setting("AZURE_OPENAI_EMBED_DEPLOYMENT")

