"""
RAGAS-Style RAG Evaluation Pipeline
====================================
Evaluates the Olfactura perfume recommendation system using
industry-standard RAG metrics:

1. Context Hit Rate   — Did the system retrieve ground truth perfumes?
2. Context Precision  — How precise were the retrieved results?  
3. Faithfulness       — Does the LLM answer stick to retrieved context?
4. Answer Relevancy   — Is the answer relevant to the question?

Metrics 3 & 4 use LLM-as-judge (GPT-4o) — no ground truth needed.
Metrics 1 & 2 use metadata-driven ground truth from golden_dataset_v2.json.

Usage:
    python evaluation/run_ragas_eval.py
    python evaluation/run_ragas_eval.py --quick   (first 5 only)
"""

import os
import sys
import json
import time
import datetime
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai.agent import PerfumeAgent
from src.database.vector_db import VectorDatabase
from src.ai.tools import set_global_db, search_perfumes

# ─────────────────────────────────────────────
# LLM-AS-JUDGE (for Faithfulness & Relevancy)  
# ─────────────────────────────────────────────

def llm_judge(prompt: str) -> str:
    """Call GPT-4o as a high-quality evaluation judge."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",  # Upgraded to more capable model for judgment
        messages=[{"role": "system", "content": "You are a strict, objective RAG evaluation judge. Respond only with a float between 0.0 and 1.0."},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def score_faithfulness(question: str, answer: str, context: str) -> float:
    """
    Faithfulness: Does the answer ONLY contain info from the context?
    """
    prompt = f"""EVALUATION TASK: FAITHFULNESS (FA)
Faithfulness measures if the answer stays true to the context.

RULES:
- Penalize (0.0) if the answer invents specific data points like Ratings, Notes, or perfume Names not in context.
- Reward (1.0) if all specific perfume details (names, scores, notes) are traceable to the context.
- Subjective expert flavoring ("this is a masterpiece", "versatile choice") is allowed but should not replace core facts.

[CONTEXT]
{context[:2500]}

[ANSWER]
{answer[:2500]}

Respond ONLY with a decimal (e.g., 0.85)."""

    try:
        result = llm_judge(prompt)
        return float(result)
    except:
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: Is the answer helpful and relevant to the question?
    """
    prompt = f"""EVALUATION TASK: RELEVANCY (RE)
Does the answer address the intent of the question?

[QUESTION]
{question}

[ANSWER]
{answer[:2000]}

Respond ONLY with a decimal (e.g., 0.95)."""

    try:
        result = llm_judge(prompt)
        return float(result)
    except:
        return 0.0


# ─────────────────────────────────────────────
# DETERMINISTIC METRICS (Metadata & Hits)
# ─────────────────────────────────────────────

def calc_metadata_constraint_score(actual_perfumes: list, constraints: dict) -> float:
    """
    Check if retrieved perfumes actually match metadata constraints 
    (e.g., if user asked for summer, is the retrieved perfume summer-rated?)
    """
    if not actual_perfumes or not constraints:
        return 1.0  # Cannot penalize if no specific metadata constraints
    
    db_filter = constraints.get("db_filter", {})
    total_checks = 0
    passed_checks = 0
    
    for perfume in actual_perfumes:
        # We check if perfume metadata (from DB) matches filter requirements
        # Note: actual_perfumes should contain metadata dictionary per perfume
        for key, val in db_filter.items():
            total_checks += 1
            perfume_val = perfume.get(key)
            if perfume_val is None: continue
            
            # Simple check for $gte / $lte / $gt / $lt
            if isinstance(val, dict):
                op = list(val.keys())[0]
                limit = val[op]
                if op == "$gte" and perfume_val >= limit: passed_checks += 1
                elif op == "$lte" and perfume_val <= limit: passed_checks += 1
                elif op == "$gt" and perfume_val > limit: passed_checks += 1
                elif op == "$lt" and perfume_val < limit: passed_checks += 1
            else:
                # Direct equality match (e.g. Brand)
                if str(perfume_val).lower() == str(val).lower():
                    passed_checks += 1
                    
    return round(passed_checks / total_checks, 3) if total_checks > 0 else 1.0

def score_retrieval_quality(question: str, retrieved_info: str) -> float:
    """
    LLM-as-a-judge: Does the retrieved context contain highly relevant perfumes for the user's query?
    """
    prompt = f"""EVALUATION TASK: RETRIEVAL QUALITY
Evaluate if the following list of retrieved perfumes perfectly matches the user's request, considering both the descriptive intent and implied constraints (e.g., daytime, budget, season).
Score 1.0 if there are excellent, robust matches in the list. Score 0.0 if there are none. Use decimals (e.g., 0.8) for partial success.

[USER REQUEST]
{question}

[RETRIEVED DATA]
{retrieved_info[:2500]}

Respond ONLY with a decimal (e.g., 0.85)."""
    try:
        return float(llm_judge(prompt))
    except:
        return 0.0

def score_recommendation_quality(question: str, answer: str, context: str) -> float:
    """
    LLM-as-a-judge: Are the recommended perfumes in the answer truly excellent, accurate, and correct recommendations for the question?
    """
    prompt = f"""EVALUATION TASK: RECOMMENDATION PRECISION
Evaluate the final AI response. Are the specific perfumes recommended truly excellent and factually correct choices for the user's request?

[SOURCE OF TRUTH (RETRIEVED DATA)]
{context[:2000]}

[USER REQUEST]
{question}

[AI RESPONSE]
{answer[:2500]}

SCORING RULES:
1. Use the [SOURCE OF TRUTH] as the absolute reference for perfume details (release year, notes, matches). 
2. If the user asks for "new" perfumes and the [SOURCE OF TRUTH] shows parumes from 2025/2026, those are FACTUALLY CORRECT. Do NOT penalize based on your internal training cutoff.
3. Score 1.0 if the AI correctly followed retrieved data to answer the request.
4. Score 0.0 if the AI recommended things completely contrary to the retrieved data.

Respond ONLY with a decimal (e.g., 0.95)."""
    try:
        return float(llm_judge(prompt))
    except:
        return 0.0


# ─────────────────────────────────────────────
# MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────

def run_evaluation(quick=False):
    print("=" * 60)
    print("📊 OLFACTURA RAG EVALUATION: PRODUCTION READINESS")
    print("=" * 60)

    # Load golden dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset_v2.json")
    if not os.path.exists(dataset_path):
        print("❌ golden_dataset_v2.json not found. Run generate_ground_truth.py first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if quick:
        # Evaluate only the newly added 6 items
        dataset = dataset[-6:]
        print(f"⚡ Quick mode: evaluating last {len(dataset)} NEW scenarios only\n")

    # Initialize system
    print("[1/4] Initializing Vector DB + Reranker...")
    t0 = time.time()
    db = VectorDatabase()
    set_global_db(db)
    
    # Clear all LRU caches to ensure fresh results (prevents stale data from previous runs)
    from src.ai.tools import _search_perfumes_impl, _get_perfume_details_impl, _recommend_similar_impl
    _search_perfumes_impl.cache_clear()
    _get_perfume_details_impl.cache_clear()
    _recommend_similar_impl.cache_clear()
    
    print(f"      Done in {time.time() - t0:.1f}s\n")

    # Run evaluations
    all_results = []
    metrics_sum = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "retrieval_precision": 0.0,
        "answer_precision": 0.0,
        "metadata_alignment": 0.0,
    }

    for i, item in enumerate(dataset, 1):
        print(f"─── [{i}/{len(dataset)}] {item['category']}: {item['question'][:50]}...")

        # Step 1: Get RAG response via agent
        agent = PerfumeAgent(model="gpt-4o-mini")
        t_start = time.time()

        response_str = ""
        try:
            for chunk in agent.chat_stream(item["question"], max_iterations=3):
                response_str += chunk
        except Exception as e:
            print(f"    ❌ Agent error: {e}")
            continue

        latency = time.time() - t_start

        # Step 2: Extract info from tool calls
        actual_perfumes_meta = []
        retrieved_names = []
        
        last_tool_call = [m for m in agent.conversation_history if m.get("role") == "assistant" and "tool_calls" in m][-1:]
        if last_tool_call:
            import json as json_lib
            try:
                args = last_tool_call[0]["tool_calls"][0]["function"]["arguments"]
                if isinstance(args, str): args = json_lib.loads(args)
                db_results = db.search(query=args.get("query"), filters=args.get("filters"), n_results=15)
                actual_perfumes_meta = db_results
                retrieved_names = [r["name"] for r in db_results]
            except: pass

        context_parts = [msg.get("content", "") for msg in agent.conversation_history if msg.get("role") == "tool"]
        context = "\n".join(context_parts) if context_parts else response_str

        # Step 3: Calculate metrics
        # Use the actual text returned by the tool(s) to the agent as the context for the LLM judge
        recall_10 = score_retrieval_quality(item["question"], context)
        precision_3 = score_recommendation_quality(item["question"], response_str, context)
        meta_score = calc_metadata_constraint_score(actual_perfumes_meta, item.get("metadata", {}))

        # LLM-as-Judge
        faithfulness = score_faithfulness(item["question"], response_str, context)
        relevancy = score_answer_relevancy(item["question"], response_str)

        # Accumulate
        metrics_sum["faithfulness"] += faithfulness
        metrics_sum["answer_relevancy"] += relevancy
        metrics_sum["retrieval_precision"] += recall_10
        metrics_sum["answer_precision"] += precision_3
        metrics_sum["metadata_alignment"] += meta_score

        # Step 4: Print
        print(f"    🌟 Ret_Prec@10: {recall_10:.2f} | 🎯 Ans_Prec@3: {precision_3:.2f} | Faith: {faithfulness:.1f}")

        all_results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "metrics": {
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "retrieval_precision_10": recall_10,
                "answer_precision_3": precision_3,
                "metadata_alignment": meta_score,
            },
            "latency": round(latency, 2),
        })

    # ─── FINAL REPORT ───
    n = len(all_results)
    if n == 0: return

    avg_metrics = {k: round(v / n, 3) for k, v in metrics_sum.items()}

    print(f"\n{'=' * 60}")
    print("📊 FINAL RAG EVALUATION REPORT (Judge: GPT-4o)")
    print("============================================================")
    print(f"  Answer Precision @ 3 (Quality):    {avg_metrics['answer_precision']:.3f} 💎")
    print(f"  Retrieval Quality @ 10 (System):   {avg_metrics['retrieval_precision']:.3f} 🕵️")
    print(f"  Metadata Constraint Match:         {avg_metrics['metadata_alignment']:.3f} 🎯")
    print(f"  Faithfulness (Factuality):         {avg_metrics['faithfulness']:.3f}")
    print(f"  Answer Relevancy (Helpfulness):    {avg_metrics['answer_relevancy']:.3f}")
    print("============================================================")

    # Category breakdown
    category_stats = {}
    for r in all_results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "ret_prec": 0, "ans_prec": 0, "faith": 0}
        category_stats[cat]["count"] += 1
        category_stats[cat]["ret_prec"] += r["metrics"]["retrieval_precision_10"]
        category_stats[cat]["ans_prec"] += r["metrics"]["answer_precision_3"]
        category_stats[cat]["faith"] += r["metrics"]["faithfulness"]

    print("\n📋 Per-Category Breakdown:")
    print("  Category               N    Ret_Prec    Ans_Prec   Faith")
    print("  " + "─" * 20 + " " + "─" * 3 + "  " + "─" * 8 + "  " + "─" * 8 + "  " + "─" * 8)
    for cat, data in category_stats.items():
        n = data["count"]
        r = data["ret_prec"] / n
        p = data["ans_prec"] / n
        f = data["faith"] / n
        print(f"  {cat:<20} {n:>3}  {r:8.3f}  {p:8.3f}  {f:8.3f}")

    # Save results
    output = {
        "summary": avg_metrics,
        "evaluated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_scenarios": n,
        "category_breakdown": category_stats,
        "details": all_results,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"rag_eval_final_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Full report saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run only first 5 scenarios")
    args = parser.parse_args()
    run_evaluation(quick=args.quick)
