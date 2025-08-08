from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from .models import ModelConfig, ModelProvider
import os


def create_llm(config: ModelConfig) -> tuple[BaseChatModel, bool]:
    if config.provider == ModelProvider.AZURE_OPENAI:
        azure_endpoint = config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = config.model_name
        api_version = config.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return (
            AzureChatOpenAI(
                azure_deployment=azure_deployment,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=(
                    os.getenv("AZURE_OPENAI_API_KEY") if not config.api_key else config.api_key
                ),
                **config.extra_params,
            ),
            True,
        )
    elif config.provider == ModelProvider.GEMINI:
        return (
            ChatGoogleGenerativeAI(
                model=config.model_name or "gemini-2.5-pro",
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                google_api_key=(
                    os.getenv("GOOGLE_API_KEY") if not config.api_key else config.api_key
                ),
                **config.extra_params,
            ),
            True,
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
