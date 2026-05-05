"""Loads Azure OpenAI credentials from .env and exposes a single client."""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")


def get_client() -> AzureOpenAI:
    if not (ENDPOINT and API_KEY):
        raise RuntimeError(
            "Missing Azure OpenAI credentials. "
            "Copy .env.example to .env and fill in AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )
    return AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )
