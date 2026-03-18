import sys
import os
import json
import time

# Ensure project root is in path
sys.path.append(os.getcwd())

try:
    from src.ai.agent import PerfumeAgent
except ImportError:
    # Try adding parent dir if running from subfolder
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from src.ai.agent import PerfumeAgent

try:
    from tqdm import tqdm
except ImportError:
    # Mock tqdm if not installed
    def tqdm(iterable): return iterable

def run_eval():
    dataset_path = "evaluation/golden_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print("Initializing Agent...")
    agent = PerfumeAgent()
    results = []
    
    print(f"🚀 Starting Evaluation on {len(dataset)} questions...")
    print("-" * 60)
    
    total_hits = 0
    total_latency = 0
    
    for i, item in enumerate(tqdm(dataset)):
        question = item["question"]
        ground_truths = item["ground_truth"]
        category = item.get("category", "General")
        
        start_time = time.time()
        try:
            agent.reset_conversation()
            # Force string conversion just in case
            response = str(agent.chat(question))
        except Exception as e:
            print(f"Error processing '{question}': {e}")
            response = ""
            latency = 0
        else:
            latency = time.time() - start_time
        
        total_latency += latency
        
        # Hit Rate Logic
        response_lower = response.lower()
        hit = False
        matched_gt = []
        
        for gt in ground_truths:
            # Check if ground truth name appears in explanation
            # (Case insensitive)
            if gt.lower() in response_lower:
                hit = True
                matched_gt.append(gt)
        
        if hit:
            total_hits += 1
            status = "✅"
        else:
            status = "❌"
            
        print(f"{status} [{category}] Q: {question[:40]}... -> Found: {matched_gt if hit else 'None'} ({latency:.2f}s)")
        
        results.append({
            "question": question,
            "ground_truth": ground_truths,
            "response": response,
            "hit": hit,
            "matched": matched_gt,
            "latency": latency
        })
        
    accuracy = (total_hits / len(dataset)) * 100
    avg_latency = total_latency / len(dataset)
    
    print("\n" + "="*60)
    print(f"📊 FINAL RESULTS")
    print("="*60)
    print(f"✅ Accuracy (Hit Rate): {accuracy:.2f}% ({total_hits}/{len(dataset)})")
    print(f"⏱️ Avg Latency:        {avg_latency:.2f}s")
    print("="*60)
    
    with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"accuracy": accuracy, "avg_latency": avg_latency},
            "details": results
        }, f, indent=2)
    print("📝 Detailed report saved to evaluation/eval_results.json")

if __name__ == "__main__":
    run_eval()
