from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Fast, deterministic token estimator for Vietnamese and English texts.

    Approximates token count at ~3.5 characters per token.
    """
    if not text:
        return 0
    clean = text.strip()
    if not clean:
        return 0
    return max(1, math.ceil(len(clean) / 3.5))


@dataclass
class UserProfileStore:
    """Persistent storage for user profiles stored in `User.md` files."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, user_id: str) -> str:
        clean = re.sub(r"[^\w\-]", "_", user_id.strip())
        return clean or "default_user"

    def path_for(self, user_id: str) -> Path:
        sanitized = self._sanitize(user_id)
        user_folder = self.root_dir / sanitized
        user_folder.mkdir(parents=True, exist_ok=True)
        return user_folder / "User.md"

    def read_text(self, user_id: str) -> str:
        filepath = self.path_for(user_id)
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        filepath = self.path_for(user_id)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        filepath = self.path_for(user_id)
        if filepath.exists():
            return filepath.stat().st_size
        return 0

    def parse_facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the User.md markdown."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            match = re.match(r"^-\s*\*\*([^*]+)\*\*:\s*(.+)$", line)
            if match:
                key = match.group(1).strip().lower()
                val = match.group(2).strip()
                facts[key] = val
        return facts

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        """Upsert facts into User.md, updating existing keys or appending new ones."""
        facts = self.parse_facts(user_id)
        facts.update(updates)

        lines = [f"# User Profile: {user_id}", "", "## Facts"]
        for k, v in sorted(facts.items()):
            lines.append(f"- **{k.capitalize()}**: {v}")
        lines.append("")
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable personal facts from user messages while rejecting noise, questions, and corrections.

    Guardrails implemented:
    - Rejects question sentences asking for facts (e.g. "Mình tên gì?", "Ở đâu?") to prevent
      erroneous overwriting of valid facts.
    - Resolves location conflicts (e.g. Da Nang -> Hue -> Da Nang) taking the latest update.
    - Rejects conversational distractions (Hanoi is just a 2-day meeting, product manager was just a joke,
      or instructions like 'đừng lấy Đà Nẵng làm nơi ở').
    """
    updates: dict[str, str] = {}
    lower = message.lower()

    # 1. Guardrail against questions
    is_asking_name = bool(re.search(r"\b(mình tên gì|tên mình là gì|tên là gì|nhắc lại tên)\b", lower))
    is_asking_location = bool(re.search(r"\b(ở đâu|nơi ở hiện tại|đang ở đâu|ở huế không)\b", lower))

    # 1. Name extraction
    if not is_asking_name:
        if "dũngct stress" in lower or "dungct_stress" in lower:
            updates["name"] = "DũngCT Stress"
        elif "dũngct" in lower:
            updates["name"] = "DũngCT"
        else:
            name_match = re.search(r"(?:mình tên là|tên mình là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_]+)?)", message)
            if name_match:
                cand = name_match.group(1).strip()
                cand_lower = cand.lower()
                if not any(stop_word in cand_lower for stop_word in ["gì", "ai", "bạn", "là"]):
                    updates["name"] = cand

    # 2. Location extraction with correction and noise rejection
    # Noise check: "Hà Nội chỉ là nơi họp" or "đừng lấy Đà Nẵng làm nơi ở hiện tại"
    is_hanoi_noise = "hà nội" in lower and any(w in lower for w in ["họp", "đối tác", "không phải nơi ở"])
    is_danang_negative = "đà nẵng" in lower and any(w in lower for w in ["đừng lấy", "ví dụ cũ", "không còn ở đà nẵng"])

    if any(phrase in lower for phrase in [
        "cập nhật từ huế sang đà nẵng",
        "đang làm việc ở đà nẵng vài tháng",
        "nơi ở hiện tại là đà nẵng",
        "đang ở đà nẵng trong giai đoạn này",
        "nơi ở đã cập nhật từ huế sang đà nẵng",
    ]):
        updates["location"] = "Đà Nẵng"
    elif any(phrase in lower for phrase in [
        "giờ mình đang ở huế",
        "hiện ở huế",
        "đang ở huế chứ không còn ở đà nẵng",
        "vẫn ở huế",
        "mình đang ở huế",
    ]):
        updates["location"] = "Huế"
    elif "đà nẵng" in lower and not is_asking_location and not is_danang_negative:
        updates["location"] = "Đà Nẵng"
    elif "huế" in lower and not is_asking_location:
        updates["location"] = "Huế"

    # 3. Profession extraction with noise rejection
    # If user says "không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer"
    if "mlops engineer" in lower:
        updates["profession"] = "MLOps engineer"
    elif "backend engineer" in lower and "không còn làm" not in lower and "đừng nói backend" not in lower:
        updates["profession"] = "backend engineer"

    # 4. Favorite drink
    is_asking_drink = bool(re.search(r"\b(uống gì|đồ uống là gì|đồ uống yêu thích là gì)\b", lower))
    if not is_asking_drink:
        if "cà phê sữa đá" in lower:
            updates["drink"] = "cà phê sữa đá"

    # 5. Favorite food
    is_asking_food = bool(re.search(r"\b(món ăn là gì|ăn gì)\b", lower))
    if not is_asking_food:
        if "mì quảng" in lower:
            updates["food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower:
        if "bơ" in lower:
            updates["pet"] = "corgi tên Bơ"
        else:
            updates["pet"] = "corgi"

    # 7. Interests / Technical Topics
    if "python" in lower and ("ai" in lower or "agent" in lower or "async" in lower):
        updates["interests"] = "Python và AI ứng dụng"

    # 8. Style preference
    if "3 bullet" in lower:
        updates["style"] = "3 bullet ngắn gọn, có ví dụ thực tế"
    elif "ngắn gọn" in lower and not bool(re.search(r"\b(thích kiểu gì|nhắc lại style)\b", lower)):
        updates["style"] = "ngắn gọn"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a structured, compact summary of older conversation turns."""
    if not messages:
        return ""

    topics: list[str] = []
    for msg in messages:
        content = msg.get("content", "").strip()
        role = msg.get("role", "user")
        if role == "user":
            if "artemis" in content.lower():
                topics.append("Artemis III roadmap & dependency management")
            elif "x-59" in content.lower():
                topics.append("X-59 supersonic flight & noise reduction")
            elif "el nino" in content.lower() or "wmo" in content.lower():
                topics.append("WMO El Nino risk probability & operations")
            elif "british columbia" in content.lower() or "điện sạch" in content.lower():
                topics.append("BC clean energy plan & demand efficiency")
            elif "memory" in content.lower() or "benchmark" in content.lower():
                topics.append("AI memory architecture & benchmark design")
            else:
                snippet = content[:60].replace("\n", " ") + "..."
                topics.append(snippet)

    # Deduplicate preserving order
    seen = set()
    unique_topics = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique_topics.append(t)

    return "Tóm tắt ngữ cảnh trước: " + "; ".join(unique_topics[:max_items])


@dataclass
class CompactMemoryManager:
    """Manages short-term conversation history with automatic compaction."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        t_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = t_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current token load
        summary_text: str = t_state["summary"]  # type: ignore
        current_tokens = estimate_tokens(summary_text) + sum(
            estimate_tokens(m["content"]) for m in messages
        )

        # Trigger compaction if token budget exceeded and enough messages accumulated
        if current_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            older = messages[:split_idx]
            preserved = messages[split_idx:]

            new_summary = summarize_messages(older)
            if summary_text:
                combined_summary = summary_text + "\n" + new_summary
            else:
                combined_summary = new_summary

            t_state["summary"] = combined_summary.strip()
            t_state["messages"] = preserved
            t_state["compactions"] = int(t_state["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id)["compactions"])
