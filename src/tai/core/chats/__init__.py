"""Conversation contracts shared by strategies and model adapters."""

from .conversation import ChatCancelled, ChatTurn, LLMChat, ModelRole

__all__ = ["ChatCancelled", "ChatTurn", "LLMChat", "ModelRole"]
