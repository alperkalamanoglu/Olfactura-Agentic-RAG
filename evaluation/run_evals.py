import os
import sys
import json
import time

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai.agent import PerfumeAgent
from src.database.vector_db import VectorDatabase
from src.ai.tools import set_global_db

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

def load_dataset():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("=" * 60)
    print("🧪 OLFACTURA RAG EVALUATION PIPELINE 🧪")
    print("=" * 60)
    
    print("\n[V] Initializing Vector Database and Reranker...")
    start_time = time.time()
    db = VectorDatabase()
    set_global_db(db)
    print(f"    Done in {time.time() - start_time:.2f} seconds.")
    
    dataset = load_dataset()
    total_queries = len(dataset)
    successful_queries = 0
    total_hits = 0
    total_expected = 0
    
    print(f"\n[V] Starting evaluation of {total_queries} benchmark queries...\n")
    
    for i, item in enumerate(dataset, 1):
        query = item['query']
        acceptable = [name.lower() for name in item['acceptable_matches']]
        min_expected = item.get('min_expected_matches', 1)
        category = item.get('category', 'General')
        
        print(f"--- Test {i}/{total_queries} [{category}] ---")
        print(f"Q: '{query}'")
        
        # Initialize fresh agent for each query to prevent context pollution
        agent = PerfumeAgent(model="gpt-4o-mini")
        
        # Run standard chat generation
        start_q_time = time.time()
        
        # Instead of getting string return, we simulate what app.py does
        gen = agent.chat_stream(query, max_iterations=3)
        response_str = ""
        try:
            for chunk in gen:
                response_str += chunk
        except Exception as e:
            print(f"    [X] Agent Failed: {str(e)}")
            continue
            
        latency = time.time() - start_q_time
        response_lower = response_str.lower()
        
        # Calculate Hits
        hits = 0
        found_matches = []
        for expected_name in acceptable:
            if expected_name in response_lower:
                hits += 1
                found_matches.append(expected_name)
        
        success = hits >= min_expected
        if success:
            successful_queries += 1
            print(f"    [+] PASS (Hits: {hits}/{min_expected}) - Time: {latency:.2f}s")
            print(f"        Found: {', '.join(found_matches)}")
        else:
            print(f"    [-] FAIL (Hits: {hits}/{min_expected}) - Time: {latency:.2f}s")
            print(f"        Expected at least {min_expected} from: {', '.join(item['acceptable_matches'][:3])}...")
        
        total_hits += hits
        total_expected += min_expected
        print()
    
    # Calculate Final Metrics
    print("=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    
    success_rate = (successful_queries / total_queries) * 100
    # Recall based on minimum expected
    recall_rate = min(100.0, (total_hits / max(1, total_expected)) * 100)
    
    print(f"Total Queries: {total_queries}")
    print(f"Successful Queries: {successful_queries} ({success_rate:.1f}%)")
    print(f"Overall Recall (vs Min Expected): {recall_rate:.1f}%")
    print(f"Total Hits: {total_hits}")
    
    print("\nSummary Validation:")
    if success_rate > 80:
        print("✅ EXCELLENT: System is highly accurate and production-ready.")
    elif success_rate > 60:
        print("⚠️ FAIR: System is acceptable but performance could be tuned.")
    else:
        print("❌ POOR: System requires significant prompt or vector DB tuning.")

if __name__ == "__main__":
    run_evaluation()
