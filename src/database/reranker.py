import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, local_model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initializes the Hybrid Reranker.
        Strategy: 
          1. Try using GPU-hosted BGE Reranker if BGE_RERANKER_URL is set.
          2. Fallback to Local CPU Cross-Encoder (FlashRank) if API fails or URL missing.
        """
        self.bge_url = os.getenv("BGE_RERANKER_URL", "").rstrip("/")
        self.local_model = None
        self.local_model_name = local_model_name
        
        if self.bge_url:
            logger.info(f"🌐 BGE Reranker URL found: {self.bge_url}. Will attempt GPU processing first.")
            self.mode = "GPU"
        else:
            logger.info("🔒 No BGE URL found. Reranker forced to LOCAL (CPU) mode.")
            self.mode = "LOCAL"
            self._init_local_model()

    def _init_local_model(self):
        """Initializes the local CrossEncoder model via FlashRank (CPU)."""
        if self.local_model is None:
            logger.info(f"📥 Loading Local Reranker model (FlashRank: {self.local_model_name})...")
            try:
                from flashrank import Ranker
                self.local_model = Ranker(model_name=self.local_model_name)
                self.mode = "LOCAL"
                logger.info("✅ Local Reranker (FlashRank - CPU) loaded.")
            except Exception as e:
                logger.error(f"❌ Error loading Local Reranker (FlashRank): {e}")

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Reranks documents using the active strategy (GPU TEI API or Local CPU).
        Returns a list of relevance scores (logits).
        """
        if not documents:
            return []

        # 1. GPU Strategy (TEI BGE Reranker API)
        if self.mode == "GPU" and self.bge_url:
            import requests
            try:
                # Text Embeddings Inference (TEI) expects {"query": str, "texts": List[str]}
                payload = {
                    "query": query,
                    "texts": documents
                }
                
                response = requests.post(
                    f"{self.bge_url}/rerank" if not self.bge_url.endswith("/rerank") else self.bge_url,
                    json=payload,
                    timeout=3.0
                )
                
                if response.status_code == 200:
                    results = response.json()
                    scores = [0.0] * len(documents)
                    
                    for res in results:
                        # Return scores as raw logits (BAAI/bge-reranker-large default)
                        # Our vector_db.py already handles sigmoid normalization for these.
                        scores[res["index"]] = float(res["score"])
                        
                    return scores
                else:
                    logger.warning(f"⚠️ BGE API returned HTTP {response.status_code}. Falling back to LOCAL CPU.")
            except Exception as e:
                logger.warning(f"⚠️ BGE API Error (Timeout/Connection): {e}. Falling back to LOCAL CPU.")
            
            # If we reach here, API failed. Switch mode permanently for this lifecycle to prevent repeated timeouts.
            self.mode = "LOCAL"
            self._init_local_model()

        # 2. Local Strategy (Fallback or Default CPU)
        if self.local_model:
            import math
            query_norm = query.lower().strip()
            # FlashRank requires a list of dicts with 'id' and 'text'
            passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
            
            try:
                from flashrank import RerankRequest
                req = RerankRequest(query=query_norm, passages=passages)
                results = self.local_model.rerank(req)
                
                # To prevent breaking VectorDB's existing logistic math handling, 
                # we must map FlashRank's output probability [0, 1] back into an approximate logit.
                scores = [0.0] * len(documents)
                for res in results:
                    idx = res["id"]
                    p = max(1e-7, min(1.0 - 1e-7, float(res["score"])))
                    logit = math.log(p / (1.0 - p))
                    scores[idx] = logit
                    
                return scores
            except Exception as e:
                logger.error(f"❌ Local Rerank Error: {e}")
                return [0.0] * len(documents)
        
        return [0.0] * len(documents)
