from typing import Optional
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

store = {}

def format_message(instructions: str, user_question: str, chat_history: Optional[str]):
  message = f"{instructions} Answer to the following question: {user_question}"
  if chat_history:
    message+="Given this context of the conversation: {chat_history}"
  return message

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]