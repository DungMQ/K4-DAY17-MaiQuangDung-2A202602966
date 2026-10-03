from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig for testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dummy_provider = ProviderConfig(
        provider="custom",
        model_name="mock-model",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # low threshold so compaction triggers easily in tests
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, edited, and queried for size."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_01"

    # Write initial profile
    initial_text = "# User Profile: test_user_01\n\n## Facts\n- **Name**: DungCT\n- **Location**: Da Nang\n"
    file_path = store.write_text(user_id, initial_text)
    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # Read profile
    read_back = store.read_text(user_id)
    assert "DungCT" in read_back
    assert "Da Nang" in read_back

    # Edit profile
    success = store.edit_text(user_id, "Da Nang", "Hue")
    assert success is True
    updated = store.read_text(user_id)
    assert "Hue" in updated
    assert "Da Nang" not in updated

    # Non-existent replacement returns False
    failed = store.edit_text(user_id, "Hanoi", "Saigon")
    assert failed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long conversation threads trigger compaction and preserve recent messages."""
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread_id = "test_thread"

    # Append 5 messages exceeding the token threshold
    turns = [
        "Đây là tin nhắn số một với nội dung rất dài về kiến trúc bộ nhớ cho AI agent.",
        "Đây là tin nhắn số hai nói về chi phí prompt tokens và cách tối ưu hóa.",
        "Đây là tin nhắn số ba thảo luận về Artemis III và các mốc kiểm chứng kỹ thuật.",
        "Đây là tin nhắn số bốn về máy bay siêu thanh X-59 và giảm tiếng nổ sonic boom.",
        "Đây là tin nhắn số năm về kế hoạch năng lượng sạch tại British Columbia.",
    ]

    for turn in turns:
        manager.append(thread_id, "user", turn)

    # Compaction should have fired at least once
    assert manager.compaction_count(thread_id) > 0

    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 3
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced Agent recalls facts across fresh sessions while Baseline does not."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "test_user"

    # Turn 1 in thread-A: user shares personal preferences
    user_statement = "Chào bạn, mình tên là DũngCT và đồ uống yêu thích của mình là cà phê sữa đá."
    baseline.reply(user_id, "thread-A", user_statement)
    advanced.reply(user_id, "thread-A", user_statement)

    # Turn 2 in a completely fresh session (thread-B): query facts
    recall_query = "Mình tên gì và đồ uống yêu thích là gì?"
    base_resp = baseline.reply(user_id, "thread-B", recall_query)["content"]
    adv_resp = advanced.reply(user_id, "thread-B", recall_query)["content"]

    # Baseline has no User.md and an empty thread-B session -> cannot recall
    assert "DũngCT" not in base_resp
    assert "cà phê sữa đá" not in base_resp

    # Advanced reads User.md -> succeeds with full recall
    assert "DũngCT" in adv_resp
    assert "cà phê sữa đá" in adv_resp


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long conversation thread."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    thread_id = "stress_thread"
    user_id = "test_user"

    long_turns = [
        "Nội dung dài số 1: Giới thiệu hệ thống AI agent và các thành phần cốt lõi trong hệ thống phân tán.",
        "Nội dung dài số 2: Phân tích chi tiết về Artemis III của NASA với các mốc kiểm chứng kỹ thuật năm 2027.",
        "Nội dung dài số 3: Nghiên cứu máy bay X-59 siêu thanh đạt Mach 1.1 tại độ cao 29500 feet.",
        "Nội dung dài số 4: Cảnh báo WMO về xác suất El Nino tăng trên 80 phần trăm trong mùa hè.",
        "Nội dung dài số 5: Kế hoạch năng lượng sạch British Columbia và sáng kiến Power Smart 2.0.",
        "Nội dung dài số 6: Tổng kết các quy luật vận hành và quản lý rủi ro từ các mẩu tin trước.",
        "Nội dung dài số 7: Khảo sát trade-off giữa recall dài hạn và chi phí token trong hệ thống memory.",
        "Nội dung dài số 8: Bàn luận về việc lưu trữ profile người dùng bền vững trong User.md markdown.",
    ]

    for turn in long_turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    # Advanced agent must have triggered compaction
    assert advanced.compaction_count(thread_id) > 0

    # Advanced prompt tokens processed must be significantly lower than Baseline's uncompressed prompt tokens
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    assert adv_prompt_tokens < base_prompt_tokens
