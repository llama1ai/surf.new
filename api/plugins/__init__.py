from enum import Enum
from typing import Callable, List, Mapping, Any, AsyncIterator, TypedDict, Union, Optional
from ..models import ModelConfig, ModelProvider
from .base import base_agent
from .browser_use import browser_use_agent
from ..utils.types import AgentSettings


class WebAgentType(Enum):
    BASE = "base"
    EXAMPLE = "example"
    BROWSER_USE = "browser_use_agent"


class SettingType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    TEXT = "text"
    TEXTAREA = "textarea"


class SettingConfig(TypedDict):
    type: SettingType
    default: Union[int, float, str]
    min: Optional[Union[int, float]]
    max: Optional[Union[int, float]]
    step: Optional[Union[int, float]]
    maxLength: Optional[int]
    description: Optional[str]


AGENT_CONFIGS = {
    WebAgentType.BROWSER_USE.value: {
        "name": "Browser Agent",
        "description": "Agent with web browsing capabilities",
        "supported_models": [
            {
                "provider": ModelProvider.AZURE_OPENAI.value,
                "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o1", "gpt-5-chat"],
            },
            {
                "provider": ModelProvider.GEMINI.value,
                "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
            },
        ],
        "model_settings": {
            "max_tokens": {
                "type": SettingType.INTEGER.value,
                "default": 1000,
                "min": 1,
                "max": 4096,
                "description": "Maximum number of tokens to generate",
            },
            "temperature": {
                "type": SettingType.FLOAT.value,
                "default": 0.7,
                "min": 0,
                "max": 1,
                "step": 0.05,
                "description": "Controls randomness in the output",
            },
        },
        "agent_settings": {
            "steps": {
                "type": SettingType.INTEGER.value,
                "default": 100,
                "min": 10,
                "max": 125,
                "description": "Max number of steps to take",
            },
        },
    },
}


def get_web_agent(
    name: WebAgentType,
) -> Callable[[ModelConfig, AgentSettings, List[Mapping[str, Any]], str], AsyncIterator[str]]:
    if name == WebAgentType.BASE:
        return base_agent
    elif name == WebAgentType.BROWSER_USE:
        return browser_use_agent
    else:
        raise ValueError(f"Unsupported web agent: {name}")
