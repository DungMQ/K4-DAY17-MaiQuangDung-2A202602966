from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON benchmark conversations from disk."""
    content = Path(path).read_text(encoding="utf-8")
    return json.loads(content)


def recall_points(answer: str, expected: list[str]) -> float:
    """Calculate recall score (0.0 to 1.0) based on substring presence in answer."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    matches = sum(1 for item in expected if item.lower() in answer_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Calculate a lightweight response quality score (0.0 to 1.0).

    Factors:
    - Base recall accuracy (50%)
    - Structured formatting with bullets (30%)
    - Appropriate answer length and brevity (20%)
    """
    if not answer or not answer.strip():
        return 0.0

    r_score = recall_points(answer, expected)
    score = r_score * 0.5

    # Structure check: bulleted format
    if any(marker in answer for marker in ["- ", "* ", "1.", "2."]):
        score += 0.3

    # Length check: concise yet informative
    word_count = len(answer.split())
    if 5 <= word_count <= 80:
        score += 0.2
    elif word_count > 80:
        score += 0.1

    return min(1.0, round(score, 2))


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Run an agent through conversations and fresh-thread recall evaluations."""
    all_threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Get primary user id
    primary_user = conversations[0].get("user_id", "default_user")
    initial_mem_size = getattr(agent, "memory_file_size", lambda u: 0)(primary_user)

    for conv in conversations:
        user_id = conv.get("user_id", primary_user)
        conv_id = conv.get("id", "conv")
        all_threads.append(conv_id)

        # 1. Process conversation turns
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=conv_id, message=turn)

        # 2. Evaluate cross-session recall in a fresh thread
        recall_questions = conv.get("recall_questions", [])
        if recall_questions:
            recall_thread_id = f"{conv_id}_recall"
            all_threads.append(recall_thread_id)
            for q_item in recall_questions:
                q_text = q_item["question"]
                expected = q_item["expected_contains"]
                resp = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=q_text)
                answer = resp.get("content", "")

                r_pts = recall_points(answer, expected)
                q_pts = heuristic_quality(answer, expected)
                recall_scores.append(r_pts)
                quality_scores.append(q_pts)

    # Aggregate token statistics across all generated threads
    total_agent_tokens = sum(agent.token_usage(t) for t in all_threads)
    total_prompt_tokens = sum(agent.prompt_token_usage(t) for t in all_threads)
    total_compactions = sum(agent.compaction_count(t) for t in all_threads)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    final_mem_size = getattr(agent, "memory_file_size", lambda u: 0)(primary_user)
    memory_growth = max(0, final_mem_size - initial_mem_size)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table using tabulate."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both Standard Benchmark and Long-Context Stress Benchmark suites."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2 - DAY 17: MEMORY SYSTEMS FOR AI AGENT BENCHMARK")
    print("=" * 80)

    # 1. Standard Benchmark
    if standard_path.exists():
        print("\n### 1. STANDARD BENCHMARK (data/conversations.json)")
        print("Evaluating multi-turn, cross-session recall across 10 conversations...\n")
        standard_convs = load_conversations(standard_path)

        baseline_agent = BaselineAgent(config, force_offline=True)
        adv_agent = AdvancedAgent(config, force_offline=True)

        row_baseline = run_agent_benchmark("Baseline Agent", baseline_agent, standard_convs, config)
        row_advanced = run_agent_benchmark("Advanced Agent", adv_agent, standard_convs, config)

        print(format_rows([row_baseline, row_advanced]))
    else:
        print(f"Warning: Standard dataset not found at {standard_path}")

    # 2. Long-Context Stress Benchmark
    if stress_path.exists():
        print("\n### 2. LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
        print("Evaluating compaction efficiency on 16 long turns with noisy context...\n")
        stress_convs = load_conversations(stress_path)

        baseline_stress = BaselineAgent(config, force_offline=True)
        adv_stress = AdvancedAgent(config, force_offline=True)

        row_stress_base = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, config)
        row_stress_adv = run_agent_benchmark("Advanced Agent", adv_stress, stress_convs, config)

        print(format_rows([row_stress_base, row_stress_adv]))
    else:
        print(f"Warning: Stress dataset not found at {stress_path}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
