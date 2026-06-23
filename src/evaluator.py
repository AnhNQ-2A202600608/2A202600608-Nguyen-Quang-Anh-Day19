"""
evaluator.py
============
So sánh Flat RAG vs GraphRAG trên 20 benchmark questions về EV industry.

Câu hỏi được phân loại:
  - Simple    (5): single-hop, câu hỏi đơn giản
  - Multi-hop (5): cần kết nối 2+ entities
  - Comparative (5): so sánh nhiều công ty/metric
  - Trend (5): phân tích xu hướng, timeline

Output: CSV + biểu đồ so sánh
"""

import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ─────────────────────────────────────────────────────────────
# 20 Benchmark Questions
# ─────────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    # ── Simple (1-5) ──
    {
        "id": 1,
        "category": "Simple",
        "question": "What was Tesla's market share in the US EV market in Q1 2024?",
        "expected_key_facts": ["51.3%", "Q1 2024", "Tesla"],
    },
    {
        "id": 2,
        "category": "Simple",
        "question": "How many new electric vehicles were sold in the US in Q1 2024?",
        "expected_key_facts": ["268,909", "Q1 2024"],
    },
    {
        "id": 3,
        "category": "Simple",
        "question": "What percentage of the global electric car stock does China account for?",
        "expected_key_facts": ["40%", "China"],
    },
    {
        "id": 4,
        "category": "Simple",
        "question": "What was the average transaction price for a new EV in Q1 2024?",
        "expected_key_facts": ["$55,167", "average transaction price"],
    },
    {
        "id": 5,
        "category": "Simple",
        "question": "What is CATL and where does it have factories in Europe?",
        "expected_key_facts": ["CATL", "battery", "Germany", "Hungary"],
    },

    # ── Multi-hop (6-10) ──
    {
        "id": 6,
        "category": "Multi-hop",
        "question": "What is the connection between BYD and Warren Buffett's Berkshire Hathaway?",
        "expected_key_facts": ["Berkshire Hathaway", "10%", "stake", "BYD"],
    },
    {
        "id": 7,
        "category": "Multi-hop",
        "question": "How did Tesla's price cuts in China affect its competitive position against NIO?",
        "expected_key_facts": ["Tesla", "price cut", "NIO", "China", "competition"],
    },
    {
        "id": 8,
        "category": "Multi-hop",
        "question": "What role did the Inflation Reduction Act play in EV leasing trends in Q1 2024?",
        "expected_key_facts": ["IRA", "Inflation Reduction Act", "leasing", "$7,500", "27%"],
    },
    {
        "id": 9,
        "category": "Multi-hop",
        "question": "How are Chinese EV manufacturers connected to the Thai automotive market?",
        "expected_key_facts": ["Thailand", "Chinese", "factory", "BYD", "investment"],
    },
    {
        "id": 10,
        "category": "Multi-hop",
        "question": "What is the relationship between CATL, Ford, and the IRA investment in Michigan?",
        "expected_key_facts": ["CATL", "Ford", "Michigan", "IRA"],
    },

    # ── Comparative (11-15) ──
    {
        "id": 11,
        "category": "Comparative",
        "question": "Compare Tesla's YoY EV sales growth in Q1 2024 vs. Cadillac's growth in the same period.",
        "expected_key_facts": ["Tesla", "-13.3%", "Cadillac", "499.2%", "Q1 2024"],
    },
    {
        "id": 12,
        "category": "Comparative",
        "question": "Which EV manufacturers achieved over 50% year-over-year growth in Q1 2024?",
        "expected_key_facts": ["BMW", "Cadillac", "Ford", "Hyundai", "Kia", "Rivian"],
    },
    {
        "id": 13,
        "category": "Comparative",
        "question": "How does EV battery pack pricing in China compare to the United States?",
        "expected_key_facts": ["24%", "lower", "China", "battery pack prices"],
    },
    {
        "id": 14,
        "category": "Comparative",
        "question": "Compare the EV market penetration forecast for the US, Europe, and China by 2030.",
        "expected_key_facts": ["37%", "40%", "48%", "US", "Europe", "China", "2030"],
    },
    {
        "id": 15,
        "category": "Comparative",
        "question": "How does the charging infrastructure availability differ between high and low EV adoption areas?",
        "expected_key_facts": ["935", "public chargers", "per million", "top ten"],
    },

    # ── Trend / Timeline (16-20) ──
    {
        "id": 16,
        "category": "Trend",
        "question": "Describe the trend in US EV sales growth from Q1 2022 through Q1 2024.",
        "expected_key_facts": ["81.2%", "46.4%", "2.6%", "slowing", "growth"],
    },
    {
        "id": 17,
        "category": "Trend",
        "question": "How has China's share of global EV exports changed, and what drove this growth?",
        "expected_key_facts": ["35%", "2022", "exports", "industrial policy", "subsidies"],
    },
    {
        "id": 18,
        "category": "Trend",
        "question": "What has been the trend in EV average transaction prices from 2023 to Q1 2024?",
        "expected_key_facts": ["9.0%", "decrease", "$55,167", "price reduction"],
    },
    {
        "id": 19,
        "category": "Trend",
        "question": "How has ZEV regulation impacted EV adoption in US states over time?",
        "expected_key_facts": ["ZEV", "5%", "zero-emission", "regulations", "13 more"],
    },
    {
        "id": 20,
        "category": "Trend",
        "question": "Trace BYD's rise from a Chinese domestic company to the world's largest EV producer.",
        "expected_key_facts": ["BYD", "largest", "surpassed Tesla", "exports", "global"],
    },
]


# ─────────────────────────────────────────────────────────────
# Scoring (heuristic)
# ─────────────────────────────────────────────────────────────
def _score_answer(answer: str, expected_key_facts: List[str]) -> Dict[str, Any]:
    """
    Heuristic scoring: kiểm tra bao nhiêu key facts xuất hiện trong answer.
    Trả về score 0-10 và danh sách facts tìm thấy/không tìm thấy.
    """
    answer_lower = answer.lower()
    found = []
    missing = []
    for fact in expected_key_facts:
        if fact.lower() in answer_lower:
            found.append(fact)
        else:
            missing.append(fact)

    score = round(10 * len(found) / max(len(expected_key_facts), 1), 1)
    return {
        "score": score,
        "facts_found": found,
        "facts_missing": missing,
        "completeness": f"{len(found)}/{len(expected_key_facts)}",
    }


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────
def run_evaluation(
    flat_rag_results: List[Dict],
    graphrag_results: List[Dict],
    questions: List[Dict] = BENCHMARK_QUESTIONS,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    So sánh kết quả Flat RAG vs GraphRAG.

    Parameters
    ----------
    flat_rag_results  : output của FlatRAG.batch_query()
    graphrag_results  : output của batch_query_graphrag()
    questions         : BENCHMARK_QUESTIONS list
    save_csv          : lưu kết quả ra CSV

    Returns
    -------
    pandas.DataFrame với bảng so sánh đầy đủ
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []

    for q_meta, flat_res, graph_res in zip(questions, flat_rag_results, graphrag_results):
        flat_score  = _score_answer(flat_res["answer"],  q_meta["expected_key_facts"])
        graph_score = _score_answer(graph_res["answer"], q_meta["expected_key_facts"])

        winner = "GraphRAG" if graph_score["score"] > flat_score["score"] else (
                 "FlatRAG" if flat_score["score"] > graph_score["score"] else "Tie")

        rows.append({
            "ID": q_meta["id"],
            "Category": q_meta["category"],
            "Question": q_meta["question"][:80] + "..." if len(q_meta["question"]) > 80 else q_meta["question"],
            "FlatRAG_Score": flat_score["score"],
            "GraphRAG_Score": graph_score["score"],
            "FlatRAG_Completeness": flat_score["completeness"],
            "GraphRAG_Completeness": graph_score["completeness"],
            "FlatRAG_Time_s": flat_res["time_sec"],
            "GraphRAG_Time_s": graph_res["time_sec"],
            "FlatRAG_Context_chars": flat_res["context_length"],
            "GraphRAG_Context_chars": graph_res["context_length"],
            "Winner": winner,
            "FlatRAG_Answer": flat_res["answer"][:200],
            "GraphRAG_Answer": graph_res["answer"][:200],
            "Missing_in_FlatRAG": ", ".join(flat_score["facts_missing"]),
            "Missing_in_GraphRAG": ", ".join(graph_score["facts_missing"]),
        })

    df = pd.DataFrame(rows)

    if save_csv:
        csv_path = OUTPUT_DIR / "evaluation_results.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[Evaluator] Results saved to: {csv_path}")

    return df


def print_evaluation_summary(df: pd.DataFrame) -> None:
    """In tóm tắt so sánh ra console."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY: FLAT RAG vs GRAPHRAG")
    print("=" * 70)

    print(f"\n🏆 Overall Scores:")
    print(f"  FlatRAG  avg score: {df['FlatRAG_Score'].mean():.2f}/10")
    print(f"  GraphRAG avg score: {df['GraphRAG_Score'].mean():.2f}/10")

    winner_counts = df["Winner"].value_counts()
    print(f"\n🥇 Winner counts:")
    for k, v in winner_counts.items():
        print(f"  {k}: {v} questions")

    print(f"\n⏱️  Average latency:")
    print(f"  FlatRAG:  {df['FlatRAG_Time_s'].mean():.2f}s")
    print(f"  GraphRAG: {df['GraphRAG_Time_s'].mean():.2f}s")

    print(f"\n📂 By Category:")
    cat_summary = df.groupby("Category")[["FlatRAG_Score", "GraphRAG_Score"]].mean()
    print(cat_summary.to_string())

    # Cases where GraphRAG wins significantly
    graph_wins = df[df["GraphRAG_Score"] > df["FlatRAG_Score"] + 2]
    if len(graph_wins) > 0:
        print(f"\n✅ Cases where GraphRAG clearly outperforms FlatRAG (+2 points):")
        for _, row in graph_wins.iterrows():
            print(f"  Q{row['ID']} [{row['Category']}]: {row['Question'][:60]}...")
            print(f"    FlatRAG={row['FlatRAG_Score']}, GraphRAG={row['GraphRAG_Score']}")

    # Cases where FlatRAG hallucinated (score 0) but GraphRAG correct (score ≥ 5)
    hallucination_cases = df[(df["FlatRAG_Score"] <= 2) & (df["GraphRAG_Score"] >= 5)]
    if len(hallucination_cases) > 0:
        print(f"\n⚠️  Probable hallucination in FlatRAG (FlatRAG≤2, GraphRAG≥5):")
        for _, row in hallucination_cases.iterrows():
            print(f"  Q{row['ID']}: {row['Question'][:60]}...")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def plot_evaluation(df: pd.DataFrame) -> None:
    """Tạo biểu đồ so sánh chi tiết."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0D1117")
    fig.suptitle("GraphRAG vs Flat RAG — Evaluation Report",
                 color="white", fontsize=16, fontweight="bold", y=1.01)

    colors_flat  = "#58A6FF"
    colors_graph = "#F97316"

    for ax in axes.flat:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    # ── Plot 1: Score per question ──
    ax1 = axes[0, 0]
    x = range(len(df))
    ax1.bar([i - 0.2 for i in x], df["FlatRAG_Score"],  width=0.4, label="FlatRAG",  color=colors_flat,  alpha=0.85)
    ax1.bar([i + 0.2 for i in x], df["GraphRAG_Score"], width=0.4, label="GraphRAG", color=colors_graph, alpha=0.85)
    ax1.set_xlabel("Question ID")
    ax1.set_ylabel("Score (0–10)")
    ax1.set_title("Score per Question")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    ax1.legend(facecolor="#0D1117", edgecolor="#30363D", labelcolor="white")
    ax1.set_ylim(0, 11)

    # ── Plot 2: Average score by category ──
    ax2 = axes[0, 1]
    cat_data = df.groupby("Category")[["FlatRAG_Score", "GraphRAG_Score"]].mean()
    categories = cat_data.index.tolist()
    x2 = range(len(categories))
    ax2.bar([i - 0.2 for i in x2], cat_data["FlatRAG_Score"],  width=0.4, color=colors_flat,  alpha=0.85, label="FlatRAG")
    ax2.bar([i + 0.2 for i in x2], cat_data["GraphRAG_Score"], width=0.4, color=colors_graph, alpha=0.85, label="GraphRAG")
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel("Avg Score")
    ax2.set_title("Average Score by Category")
    ax2.legend(facecolor="#0D1117", edgecolor="#30363D", labelcolor="white")
    ax2.set_ylim(0, 11)

    # ── Plot 3: Winner distribution (pie) ──
    ax3 = axes[1, 0]
    winner_counts = df["Winner"].value_counts()
    pie_colors = {"GraphRAG": colors_graph, "FlatRAG": colors_flat, "Tie": "#6E7681"}
    wedge_colors = [pie_colors.get(k, "#999") for k in winner_counts.index]
    wedges, texts, autotexts = ax3.pie(
        winner_counts.values,
        labels=winner_counts.index,
        autopct="%1.0f%%",
        colors=wedge_colors,
        textprops={"color": "white"},
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax3.set_title("Winner Distribution")

    # ── Plot 4: Latency comparison ──
    ax4 = axes[1, 1]
    ax4.scatter(range(len(df)), df["FlatRAG_Time_s"],  color=colors_flat,  label="FlatRAG",  s=80, alpha=0.9)
    ax4.scatter(range(len(df)), df["GraphRAG_Time_s"], color=colors_graph, label="GraphRAG", s=80, alpha=0.9, marker="^")
    ax4.plot(range(len(df)), df["FlatRAG_Time_s"],  color=colors_flat,  alpha=0.5, linewidth=1)
    ax4.plot(range(len(df)), df["GraphRAG_Time_s"], color=colors_graph, alpha=0.5, linewidth=1)
    ax4.axhline(df["FlatRAG_Time_s"].mean(),  color=colors_flat,  linestyle="--", alpha=0.6, linewidth=0.8)
    ax4.axhline(df["GraphRAG_Time_s"].mean(), color=colors_graph, linestyle="--", alpha=0.6, linewidth=0.8)
    ax4.set_xlabel("Question ID")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Query Latency")
    ax4.legend(facecolor="#0D1117", edgecolor="#30363D", labelcolor="white")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "evaluation_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"[Evaluator] Chart saved to: {save_path}")
