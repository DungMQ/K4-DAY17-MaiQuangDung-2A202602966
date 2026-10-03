from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session / within-thread memory only.
    - No persistent User.md profile storage.
    - Cannot recall facts across fresh sessions/threads.
    - Carries raw, uncompacted history into every turn, causing prompt token load
      to grow rapidly ($O(N^2)$ context processing).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process incoming user turn and account for prompt and generation tokens."""
        session = self._get_session(thread_id)

        # Baseline prompt context: entire uncompressed history + new user message
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += prompt_tokens

        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live LLM execution path
                response = self.langchain_agent.invoke(
                    {"messages": session.messages + [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                reply_text = str(response.get("messages", [-1])[-1].content)
                out_tokens = estimate_tokens(reply_text)
                session.token_usage += out_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": reply_text})
                return {"role": "assistant", "content": reply_text}
            except Exception:
                pass  # Fallback to offline mode on live failure

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def memory_file_size(self, user_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline response logic for Baseline Agent.

        Crucially:
        - Stores messages within the active thread_id only.
        - Has no knowledge of facts communicated in other thread_ids.
        - Answers cross-session recall questions with a generic negative response.
        """
        session = self._get_session(thread_id)
        lower = message.lower()

        is_recall_query = any(w in lower for w in ["tên gì", "nhắc lại", "ở đâu", "nghề gì", "uống gì", "món ăn", "nuôi con gì", "style"])

        if is_recall_query:
            # Baseline in a new thread has empty history and no User.md
            # Look only in current session messages
            current_history_text = " ".join(m["content"] for m in session.messages).lower()
            if not current_history_text:
                reply_text = "Chào bạn, đây là phiên trò chuyện mới. Mình chưa có thông tin nào về bạn từ trước."
            else:
                reply_text = "Mình chỉ ghi nhận thông tin đã trao đổi trong phiên làm việc này."
        else:
            reply_text = "Đã ghi nhận thông tin của bạn trong phiên trò chuyện này."

        out_tokens = estimate_tokens(reply_text)
        session.token_usage += out_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {"role": "assistant", "content": reply_text}

    def _maybe_build_langchain_agent(self):
        """Optionally build a LangChain/LangGraph agent with In-Memory checkpointing."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
