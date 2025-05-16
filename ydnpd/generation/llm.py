from typing import Any, Callable

from langchain_core.messages import (
    convert_to_openai_messages,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    FunctionMessage
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat
from langchain_together import ChatTogether
from dotenv import dotenv_values


SECRETS = dotenv_values(".env")


def _convert_openai_to_langchain_messages(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert OpenAI format messages to LangChain messages."""
    message_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
        "function": FunctionMessage
    }

    converted_messages = []
    for message in messages:
        message_type = message_map.get(message["role"])
        if message_type:
            kwargs = {"content": message["content"]}

            if message["role"] == "tool":
                kwargs["tool_call_id"] = message.get("tool_call_id")
            elif message["role"] == "function":
                kwargs["name"] = message.get("name")

            converted_messages.append(message_type(**kwargs))

    return converted_messages


def create_llm(
    model_path: str,
    **model_params
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """
    Create a chat function for the specified model.

    Args:
        model_path: Format "provider/model" (e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-latest")
        **model_params: Additional model parameters like temperature
    """
    provider, model_name = model_path.split("/", maxsplit=1)

    # Configure the base model
    if provider == "openai":
        chat = ChatOpenAI(
            model=model_name,
            api_key=SECRETS["OPENAI_API_KEY"],
            **model_params
        )
    elif provider == "anthropic":
        chat = ChatAnthropic(
            model=model_name,
            api_key=SECRETS["ANTHROPIC_API_KEY"],
            **model_params
        )
    elif provider == "bedrock":
        chat = BedrockChat(
            model_id=model_name,
            client_kwargs={
                "aws_access_key_id": SECRETS["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": SECRETS["AWS_SECRET_ACCESS_KEY"],
                "region_name": SECRETS["AWS_REGION", "us-east-1"]
            },
            model_kwargs=model_params
        )
    elif provider == "together":
        chat = ChatTogether(
            model=model_name,
            api_key=SECRETS.get("TOGETHER_API_KEY"),
            **model_params
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    def chat_function(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Get completion for the given messages."""
        langchain_messages = _convert_openai_to_langchain_messages(messages)
        response = chat.invoke(langchain_messages)
        message = convert_to_openai_messages(response)
        assert message["role"] == "assistant"
        return message["content"]

    return chat_function
