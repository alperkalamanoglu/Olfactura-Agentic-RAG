"""
Metadata-Driven Ground Truth Generator
=======================================
Generates golden evaluation dataset by querying structured metadata
from the ChromaDB vector database. No LLM involved — pure deterministic
filtering ensures unbiased ground truth.

Usage:
    python evaluation/generate_ground_truth.py
"""

import os
import sys
import json

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.vector_db import VectorDatabase

# ─────────────────────────────────────────────────
# TEST SCENARIOS: Each has a question + DB filters
# to deterministically find ground truth perfumes.
# ─────────────────────────────────────────────────

EVAL_SCENARIOS = [
    # ── CATEGORY: Season + Gender ──
    {
        "id": "season_01",
        "category": "Season + Gender",
        "question": "Recommend a fresh summer perfume for men",
        "db_filter": {"season_summer": {"$gte": 1.0}, "gender_score": {"$gt": 0.6}},
        "sort_by": "weighted_rating",
        "semantic_query": "fresh summer citrus aquatic men",
        "n_ground_truth": 5,
    },
    {
        "id": "season_02",
        "category": "Season + Gender",
        "question": "Best winter fragrance for women with strong longevity",
        "db_filter": {"season_winter": {"$gte": 1.0}, "gender_score": {"$lt": 0.4}},
        "sort_by": "weighted_rating",
        "semantic_query": "warm winter long lasting women",
        "n_ground_truth": 5,
    },
    {
        "id": "season_03",
        "category": "Season + Gender",
        "question": "A spring perfume suitable for both men and women",
        "db_filter": {"season_spring": {"$gte": 1.0}, "gender_score": {"$gte": 0.35, "$lte": 0.65}},
        "sort_by": "weighted_rating",
        "semantic_query": "spring unisex fresh floral",
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Accord-Specific ──
    {
        "id": "accord_01",
        "category": "Accord-Specific",
        "question": "A dark smoky oud perfume",
        "db_filter": {},
        "sort_by": "weighted_rating",
        "semantic_query": "dark smoky oud intense mysterious",
        "accord_must_contain": ["oud"],
        "n_ground_truth": 5,
    },
    {
        "id": "accord_02",
        "category": "Accord-Specific",
        "question": "Sweet vanilla gourmand dessert perfume",
        "db_filter": {},
        "sort_by": "weighted_rating",
        "semantic_query": "sweet vanilla gourmand dessert caramel",
        "accord_must_contain": ["vanilla"],
        "n_ground_truth": 5,
    },
    {
        "id": "accord_03",
        "category": "Accord-Specific",
        "question": "Fresh citrus cologne for hot weather",
        "db_filter": {},
        "sort_by": "weighted_rating",
        "semantic_query": "fresh citrus aquatic cologne summer",
        "accord_must_contain": ["citrus"],
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Price-Tier ──
    {
        "id": "price_01",
        "category": "Price-Tier",
        "question": "Best cheap perfume under budget for men",
        "db_filter": {"price_tier_score": {"$lt": 3.0}, "gender_score": {"$gt": 0.6}},
        "sort_by": "weighted_rating",
        "semantic_query": "cheap affordable masculine",
        "n_ground_truth": 5,
    },
    {
        "id": "price_02",
        "category": "Price-Tier",
        "question": "Most luxurious premium niche perfume",
        "db_filter": {},
        "sort_by": "weighted_rating",
        "semantic_query": "luxury niche premium exclusive high quality",
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Brand ──
    {
        "id": "brand_01",
        "category": "Brand",
        "question": "Best perfumes by Tom Ford",
        "db_filter": {"brand": "Tom Ford"},
        "sort_by": "weighted_rating",
        "semantic_query": None,
        "n_ground_truth": 5,
    },
    {
        "id": "brand_02",
        "category": "Brand",
        "question": "Best perfumes by Dior",
        "db_filter": {"brand": "Dior"},
        "sort_by": "weighted_rating",
        "semantic_query": None,
        "n_ground_truth": 5,
    },
    {
        "id": "brand_03",
        "category": "Brand",
        "question": "Most popular Chanel fragrances",
        "db_filter": {"brand": "Chanel"},
        "sort_by": "popularity_score",
        "semantic_query": None,
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Time of Day ──
    {
        "id": "tod_01",
        "category": "Time of Day",
        "question": "A perfume perfect for daytime office wear",
        "db_filter": {"tod_day": {"$gte": 1.0}},
        "sort_by": "weighted_rating",
        "semantic_query": "clean fresh professional office daytime",
        "n_ground_truth": 5,
    },
    {
        "id": "tod_02",
        "category": "Time of Day",
        "question": "Seductive night out fragrance",
        "db_filter": {"tod_night": {"$gte": 1.0}},
        "sort_by": "weighted_rating",
        "semantic_query": "seductive night club party dark",
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Semantic-Only (No Metadata Filter) ──
    {
        "id": "semantic_01",
        "category": "Semantic-Only",
        "question": "Dark mysterious intense fragrance",
        "db_filter": {},
        "sort_by": None,
        "semantic_query": "dark mysterious intense smoky deep",
        "n_ground_truth": 5,
    },
    {
        "id": "semantic_02",
        "category": "Semantic-Only",
        "question": "Clean soapy out of the shower scent",
        "db_filter": {},
        "sort_by": None,
        "semantic_query": "clean soapy shower fresh laundry",
        "n_ground_truth": 5,
    },
    {
        "id": "semantic_03",
        "category": "Semantic-Only",
        "question": "Romantic date night warm sensual perfume",
        "db_filter": {},
        "sort_by": None,
        "semantic_query": "romantic date night warm sensual seductive",
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Year (New Releases) ──
    {
        "id": "year_01",
        "category": "New Releases",
        "question": "Best new perfumes",
        "db_filter": {"year": {"$gte": 2025}},
        "sort_by": "weighted_rating",
        "semantic_query": None,
        "n_ground_truth": 5,
    },

    # ── CATEGORY: Combined Filters ──
    {
        "id": "combo_01",
        "category": "Combined",
        "question": "Cheap fresh summer cologne for men released after 2020",
        "db_filter": {
            "season_summer": {"$gte": 1.0},
            "gender_score": {"$gt": 0.6},
            "price_tier_score": {"$lt": 4.0},
            "year": {"$gte": 2020}
        },
        "sort_by": "weighted_rating",
        "semantic_query": "fresh summer citrus aquatic cheap",
        "n_ground_truth": 5,
    },
    {
        "id": "combo_02",
        "category": "Combined",
        "question": "Premium winter vanilla for women with great longevity",
        "db_filter": {
            "season_winter": {"$gte": 1.0},
            "gender_score": {"$lt": 0.4},
        },
        "sort_by": "weighted_rating",
        "semantic_query": "vanilla warm winter long lasting women premium quality",
        "n_ground_truth": 5,
    },
]


def generate_ground_truth(db: VectorDatabase):
    """
    For each scenario, query the DB with TWO strategies and merge results
    to create a wide, fair ground truth pool:
    
    Strategy A: Metadata filter + sort by rating → "objectively best" perfumes
    Strategy B: Semantic query + reranker → "most relevant" perfumes
    
    Merging both ensures the agent isn't punished for finding a semantically
    perfect perfume that happens to have a slightly lower rating.
    """
    golden_dataset = []

    for scenario in EVAL_SCENARIOS:
        print(f"\n[{scenario['id']}] {scenario['category']}: {scenario['question']}")

        all_names = set()      # dedup by name
        all_results = []       # ordered list

        # ── Strategy A: Rating-sorted with metadata filters ──
        results_a = db.search(
            query=scenario.get("semantic_query"),
            filters=scenario["db_filter"] if scenario["db_filter"] else None,
            sort_by=scenario.get("sort_by"),
            n_results=20,
        )

        # ── Strategy B: Pure semantic (no sort_by, let reranker decide) ──
        if scenario.get("semantic_query"):
            results_b = db.search(
                query=scenario["semantic_query"],
                filters=scenario["db_filter"] if scenario["db_filter"] else None,
                sort_by=None,   # Pure semantic relevance
                n_results=20,
            )
        else:
            results_b = []

        # Merge both strategies (A first, then B's unique additions)
        for r in results_a + results_b:
            name = r.get("name", "")
            if name and name not in all_names:
                all_names.add(name)
                all_results.append(r)

        # Post-filter by accord if specified
        accord_filter = scenario.get("accord_must_contain", [])
        if accord_filter:
            all_results = [
                r for r in all_results
                if any(acc.lower() in r.get("accords_str", "").lower() for acc in accord_filter)
            ]

        # Take top 10 as acceptable ground truth pool
        n = 10
        top_results = all_results[:n]

        if not top_results:
            print(f"  ⚠️ No results found! Skipping.")
            continue

        ground_truth_names = []
        for r in top_results:
            name = f"{r['name']} by {r['brand']}"
            ground_truth_names.append(name)

        # Print first 5 for console readability
        for r in top_results[:5]:
            rating = r.get('weighted_rating', r.get('rating', 0))
            print(f"  → {r['name']} by {r['brand']} (Rating: {rating:.2f})")
        if len(top_results) > 5:
            print(f"  ... and {len(top_results) - 5} more in pool")

        golden_dataset.append({
            "id": scenario["id"],
            "category": scenario["category"],
            "question": scenario["question"],
            "ground_truth": ground_truth_names,
            "ground_truth_short": [r["name"] for r in top_results],
            "metadata": {
                "db_filter": scenario["db_filter"],
                "sort_by": scenario.get("sort_by"),
                "semantic_query": scenario.get("semantic_query"),
                "accord_filter": accord_filter,
                "pool_size": len(top_results),
                "strategy": "dual (rating + semantic)",
            }
        })

    return golden_dataset


def main():
    print("=" * 60)
    print("🔬 METADATA-DRIVEN GROUND TRUTH GENERATOR")
    print("=" * 60)

    db = VectorDatabase()

    golden = generate_ground_truth(db)

    output_path = os.path.join(os.path.dirname(__file__), "golden_dataset_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Generated {len(golden)} evaluation scenarios")
    print(f"📁 Saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
