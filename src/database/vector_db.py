import os
import json
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from rapidfuzz import process, fuzz
import pandas as pd
from .reranker import Reranker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetches embeddings from OpenAI for a list of texts.
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

class VectorDatabase:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "perfumes"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Explicitly use cosine distance. Default is L2.
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        self.embedding_manager = EmbeddingManager()
        self._reranker = None # Lazy initialization

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def add_perfumes(self, perfumes: List[Dict[str, Any]], batch_size: int = 100):
        """
        Adds multiple perfumes to the vector database in batches.
        """
        for i in tqdm(range(0, len(perfumes), batch_size), desc="Ingesting perfumes"):
            batch = perfumes[i:i + batch_size]
            
            # Filter out perfumes with empty semantic_text (OpenAI rejects empty strings)
            valid_batch = [p for p in batch if p.get('semantic_text') and p['semantic_text'].strip()]
            
            if not valid_batch:
                continue
                
            # Generate unique IDs from brand + name
            ids = []
            seen_in_batch = {}
            for p in valid_batch:
                base_id = f"{p.get('brand', 'Unknown')}_{p.get('clean_name', 'Unknown')}".replace(' ', '_').replace('&', 'and')
                if base_id in seen_in_batch:
                    seen_in_batch[base_id] += 1
                    final_id = f"{base_id}_{seen_in_batch[base_id]}"
                else:
                    seen_in_batch[base_id] = 0
                    final_id = base_id
                ids.append(final_id)
            
            documents = [p['semantic_text'] for p in valid_batch]
            
            # Prepare metadata (ensure all values are strings or numbers for ChromaDB)
            metadatas = []
            for p in valid_batch:
                # Attempt to convert year to int for numeric filtering
                year_val = p.get('year', 0)
                try:
                    if isinstance(year_val, str) and year_val.isdigit():
                        year_int = int(year_val)
                    else:
                        year_int = int(year_val) if year_val else 0
                except:
                    year_int = 0

                # --- Boolean Flags for Seasons and Time of Day (Chromadb doesn't support list metadata) ---
                seasons_dict = p.get('seasons', {})
                tod_dict = p.get('time_of_day', {})

                meta = {
                    "brand": str(p.get('brand', 'Unknown')),
                    "name": str(p.get('clean_name', 'Unknown')),
                    "family": str(p.get('family', 'Unknown')),
                    "gender_score": float(p.get('gender_score', 0.5)),
                    "price_tier_score": float(p.get('price_tier_score', 5.0)),
                    "longevity_score": float(p.get('longevity_score', 0.0)),
                    "sillage_score": float(p.get('sillage_score', 0.0)),
                    "year": year_int,
                    "rating": float(p.get('rating', 0.0)),
                    "votes": int(p.get('votes', 0)),
                    "weighted_rating": float(p.get('weighted_rating', 0.0)),
                    "popularity_score": float(p.get('popularity_score', 0.0)),
                    # Flags: 1.0 = Suitable, 0.0 = Not suitable
                    "season_winter": 1.0 if seasons_dict.get('winter', 0) >= 40 else 0.0,
                    "season_spring": 1.0 if seasons_dict.get('spring', 0) >= 40 else 0.0,
                    "season_summer": 1.0 if seasons_dict.get('summer', 0) >= 40 else 0.0,
                    "season_fall": 1.0 if seasons_dict.get('fall', 0) >= 40 else 0.0,
                    "tod_day": 1.0 if tod_dict.get('day', 0) >= 40 else 0.0,
                    "tod_night": 1.0 if tod_dict.get('night', 0) >= 40 else 0.0,
                }
                metadatas.append(meta)

            try:
                embeddings = self.embedding_manager.get_embeddings(documents)
                
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            except Exception as e:
                logger.error(f"Error ingesting batch starting at index {i}: {e}")
                # Continue with next batch instead of crashing the whole process
                continue

    def search(self, 
               query: Optional[str] = None, 
               n_results: int = 5, 
               filters: Optional[Dict] = None,
               sort_by: Optional[str] = None,
               ascending: bool = False,
               excluded_notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Public API for text-based semantic search.
        Delegates to _execute_search.
        """
        query_embedding = None
        if query:
            query_embedding = self.embedding_manager.get_embeddings([query])[0]
            
        return self._execute_search(
            query_embedding=query_embedding,
            query_text=query,
            filters=filters,
            sort_by=sort_by,
            n_results=n_results,
            excluded_notes=excluded_notes
        )


    def get_by_name(self, name: str, fuzzy: bool = True) -> Dict[str, Any]:
        """
        Retrieves a perfume with hierarchical matching:
        1. Exact match (Case-insensitive)
        2. Fuzzy match (Partial ratio)
        3. Tie-breaker: Shortest name + Most popular
        """
        name_query = name.lower().strip()
        
        # Fetch all metadata AND embeddings
        all_data = self.collection.get(include=["metadatas", "documents", "embeddings"])
        ids = all_data['ids']
        metadatas = all_data['metadatas']
        documents = all_data['documents']
        embeddings = all_data['embeddings']

        # Pre-process records for matching
        records = []
        for i, meta in enumerate(metadatas):
            brand = str(meta.get('brand', '')).lower()
            perfume_name = str(meta.get('name', '')).lower()
            full_name = f"{brand} {perfume_name}"
            
            # --- STEP 1: Perfect Match Check ---
            if name_query == perfume_name or name_query == full_name:
                record = meta.copy()
                record['id'] = ids[i]
                record['semantic_text'] = documents[i]
                record['embedding'] = embeddings[i] # Include embedding
                return {"record": record, "suggestions": []}
            
            records.append({
                "index": i,
                "full_name": full_name,
                "short_name": perfume_name,
                "votes": meta.get('votes', 0),
                "metadata": meta,
                "document": documents[i],
                "embedding": embeddings[i], # Include embedding
                "id": ids[i]
            })

        # --- STEP 2: Fuzzy Matching ---
        if fuzzy:
            # Use token_sort_ratio to handle word reordering (e.g. "Dylan Blue Pour Homme" vs "Pour Homme Dylan Blue")
            all_full_names = [r['full_name'] for r in records]
            best_matches = process.extract(name_query, all_full_names, scorer=fuzz.token_sort_ratio, limit=10)
            # Filter for high quality matches (>80, handles missing brands/reordering better)
            candidates = [m for m in best_matches if m[1] >= 80]
            if candidates:
                # Find the actual record objects for these candidates
                match_results = []
                for name_text, score, _ in candidates:
                    # Find all records with this exact full_name text
                    for r in records:
                        if r['full_name'] == name_text:
                            # Add scoring data for tie-breaking
                            match_results.append({
                                "record": r,
                                "score": score,
                                "name_len": len(r['short_name']),
                                "popularity": r['votes']
                            })
                
                # --- STEP 3: Tie-Breaking ---
                # 1. Best Score (Descending)
                # 2. Shortest Name (Ascending) - preferring shorter, direct names
                # 3. Popularity (Descending) - choosing the more common one
                match_results.sort(key=lambda x: (-x['score'], x['name_len'], -x['popularity']))
                
                best = match_results[0]['record']
                best_meta = best['metadata'].copy()
                best_meta['id'] = best['id']
                best_meta['semantic_text'] = best['document']
                best_meta['embedding'] = best['embedding'] # Fixed: Include embedding for fuzzy matches
                return {"record": best_meta, "suggestions": []}
            
            # If no strong match, provide suggestions
            all_short_names = [r['short_name'] for r in records]
            near_matches = process.extract(name_query, all_short_names, limit=3)
            return {"record": None, "suggestions": [m[0] for m in near_matches]}

        return {"record": None, "suggestions": []}

    def recommend_similar(self, perfume_id: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Recommends perfumes similar to a given perfume ID (URL).
        """
        try:
            # Get the embedding of the reference perfume
            ref = self.collection.get(ids=[perfume_id], include=["embeddings"])
            
            # Defensive checks for embeddings
            if ref is None:
                logger.warning(f"No data returned for perfume_id: {perfume_id}")
                return []
            
            embeddings = ref.get('embeddings')
            if embeddings is None or len(embeddings) == 0:
                logger.warning(f"No embeddings found for perfume_id: {perfume_id}")
                return []
                
            # Embeddings can be list or numpy array, and usually nested [[...]]
            ref_embedding = embeddings[0]
            
            # If still nested (some versions of Chroma do this), unwrap
            try:
                if len(ref_embedding) > 0 and hasattr(ref_embedding[0], '__len__') and not isinstance(ref_embedding[0], (str, bytes)):
                    ref_embedding = ref_embedding[0]
            except:
                pass
            
            # Delegate to search_by_embedding to safely handle `$or`/`$and` logic and get hybrid scores
            results = self.search_by_embedding(
                query_embedding=ref_embedding,
                filters=filters,
                n_results=n_results + 1  # Get 1 extra to account for filtering itself
            )
            
            # Filter out the original perfume ID
            hits = [hit for hit in results if hit.get('id') != perfume_id]
            
            return hits[:n_results]
        except Exception as e:
            logger.error(f"Error in recommend_similar: {e}")
            raise

    def search_by_embedding(self, 
                            query_embedding: List[float], 
                            query_text: Optional[str] = None,
                            filters: Optional[Dict] = None, 
                            sort_by: Optional[str] = None, 
                            n_results: int = 5,
                            excluded_notes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Public API for vector-based search (e.g. centroid search).
        """
        return self._execute_search(
            query_embedding=query_embedding,
            query_text=query_text,
            filters=filters,
            sort_by=sort_by,
            n_results=n_results,
            excluded_notes=excluded_notes
        )

    def _execute_search(self,
                        query_embedding: Optional[List[float]],
                        query_text: Optional[str],
                        filters: Optional[Dict] = None, 
                        sort_by: Optional[str] = None, 
                        n_results: int = 5,
                        excluded_notes: Optional[List[str]] = None):
        """
        Internal worker methodology for executing search logic.
        """
        candidates_pool_size = 50 # Pulling 50 protects against excluded notes filtering
        
        if sort_by:
             candidates_pool_size = 80 # Broader net if sorting by specific metric (e.g. price)

        # ChromaDB Query Syntax for multiple where clauses.
        # If there are multiple keys, or if '$or' is present alongside other keys,
        # ChromaDB requires them to be wrapped in a single root '$and'.
        def _sanitize_chroma_filters(d):
            if not isinstance(d, dict) or not d:
                return None if d == {} else d
                
            def _flatten(node):
                flat = []
                for k, v in node.items():
                    if k == "excluded_notes": continue
                    if k in ("$and", "$or"):
                        if isinstance(v, list):
                            processed_list = []
                            for item in v:
                                if isinstance(item, dict):
                                    item_flat = _flatten(item)
                                    if len(item_flat) == 1:
                                        processed_list.append(item_flat[0])
                                    elif len(item_flat) > 1:
                                        processed_list.append({"$and": item_flat})
                            flat.append({k: processed_list})
                    elif isinstance(v, dict):
                        keys = list(v.keys())
                        if len(keys) > 1 and all(ik.startswith("$") for ik in keys):
                            for ik, iv in v.items():
                                flat.append({k: {ik: iv}})
                        else:
                            flat.append({k: v})
                    else:
                        flat.append({k: v})
                return flat

            flattened = _flatten(d)
            if len(flattened) == 0:
                return None
            elif len(flattened) == 1:
                return flattened[0]
            else:
                return {"$and": flattened}

        search_filters = _sanitize_chroma_filters(filters)
        if search_filters == {}:
            search_filters = None

        if query_embedding is not None and len(query_embedding) > 0:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidates_pool_size,
                where=search_filters,
                include=["metadatas", "documents", "distances"]
            )
        else:
            # CRITICAL: ChromaDB .get() returns in INSERTION ORDER, not by quality.
            # We must fetch a large pool and let our scoring logic select the best ones.
            filter_only_pool = 1000
            results = self.collection.get(
                where=search_filters,
                limit=filter_only_pool,
                include=["metadatas", "documents"]
            )
            
        hits = []
        ids = results['ids'][0] if isinstance(results['ids'][0], list) else results['ids']
        metadatas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
        documents = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
        distances = results.get('distances', [[]])[0] if isinstance(results.get('distances', [[]])[0], list) else results.get('distances', [])

        for i in range(len(ids)):
            hit = metadatas[i].copy()
            hit['id'] = ids[i]
            hit['semantic_text'] = documents[i]
            if i < len(distances):
                # ChromaDB Default Metric is Squared L2.
                # OpenAI Embeddings are normalized.
                # Formula: CosineSim = 1 - (SquaredL2 / 2)
                hit['initial_cosine_score'] = 1 - (distances[i] / 2) 
            else:
                 hit['initial_cosine_score'] = 0
            hits.append(hit)
        
        # Filter by excluded notes
        if excluded_notes:
            import re
            excluded_lower = [note.lower() for note in excluded_notes]
            filtered_hits = []
            for hit in hits:
                semantic_lower = hit['semantic_text'].lower()
                has_excluded = False
                for note in excluded_lower:
                    if re.search(r'\b' + re.escape(note) + r'\b', semantic_lower):
                        has_excluded = True
                        break
                if not has_excluded:
                    filtered_hits.append(hit)
            hits = filtered_hits
        # --- PRE-RANKING ---
        if query_embedding is not None and len(query_embedding) > 0 and len(hits) > 0:
            for hit in hits:
                hit['pre_rank_score'] = hit['initial_cosine_score']
                
            hits.sort(key=lambda x: x.get('pre_rank_score', 0), reverse=True)
            # Send top candidates to reranker (capped at 35 for speed).
            MAX_RERANK_POOL = 35
            hits_to_rerank = hits[:MAX_RERANK_POOL]
            
            # --- RERANKING STAGE ---
            if query_text:
                import time
                docs_for_rerank = [hit['semantic_text'] for hit in hits_to_rerank]
                
                t_rerank_start = time.time()
                rerank_scores = self.reranker.rerank(query_text, docs_for_rerank)
                t_rerank_end = time.time()
                
                try:
                    from src.ai.logger import log_event, agent_logger
                    log_event(agent_logger, "PERFORMANCE", "Reranker Executed", {"seconds": round(t_rerank_end - t_rerank_start, 3), "docs": len(docs_for_rerank)})
                except ImportError:
                    pass
                
                for i, hit in enumerate(hits_to_rerank):
                    hit['relevance_score'] = rerank_scores[i]
            else:
                # Vector-only search: Map Cosine to Logit Proxy
                import math
                for hit in hits_to_rerank:
                    sim = max(0.01, min(0.99, hit['initial_cosine_score']))
                    logit = math.log(sim / (1 - sim))
                    hit['relevance_score'] = logit

            # --- FINAL HYBRID SCORING ---
            if sort_by == 'votes':
                w_relevance = 0.5; w_pop = 0.5
                for hit in hits_to_rerank:
                    raw_logit = hit.get('relevance_score', 0)
                    norm_relevance = 1 / (1 + pow(2.718, -(raw_logit + 4.0) / 2.0))
                    norm_pop = hit.get('popularity_score', 0)
                    hit['hybrid_score'] = (norm_relevance * w_relevance) + (norm_pop * w_pop)
                hits_to_rerank.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
                hits = hits_to_rerank

            elif sort_by:
                for hit in hits_to_rerank:
                    raw_logit = hit.get('relevance_score', 0)
                    norm_relevance = 1 / (1 + pow(2.718, -(raw_logit + 4.0) / 2.0))
                    val = hit.get(sort_by, 0)
                    
                    # Heuristic normalization based on field name
                    if sort_by == 'weighted_rating':
                        # PARADOX FIX: 'Value for Money' bias inflates clone ratings. 
                        # Blend normalized rating (70%) with popularity_score (30%) to ensure
                        # highly-rated but unknown clones don't dominate global masterpieces.
                        norm_rating = min(1.0, val / 5.0)
                        norm_pop = hit.get('popularity_score', 0)
                        norm_val = (norm_rating * 0.70) + (norm_pop * 0.30)
                    elif sort_by == 'votes':
                        # Use the pre-computed normalized popularity log-score instead of raw votes
                        norm_val = hit.get('popularity_score', 0)
                    elif 'rating' in sort_by:
                        norm_val = min(1.0, val / 5.0)
                    elif 'score' in sort_by: # popularity_score, etc (0-1)
                        norm_val = min(1.0, val)
                    else:
                        norm_val = min(1.0, val / 10.0)
                        
                    hit['hybrid_score'] = (norm_relevance * 0.4) + (norm_val * 0.6)
                hits_to_rerank.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
                hits = hits_to_rerank
            else:
                # Default Sort (Semantic focus with Minor Quality/Trend tie-breaker)
                for hit in hits_to_rerank:
                    # Map raw Cross-Encoder logits (typically -8 to +4) to realistic percentage
                    raw_logit = hit.get('relevance_score', 0)
                    norm_relevance = 1 / (1 + pow(2.718, -(raw_logit + 4.0) / 2.0))
                    
                    norm_rating = hit.get('weighted_rating', 0) / 5.0
                    norm_pop = hit.get('popularity_score', 0)
                    
                    # 75% Semantic Relevance vs 25% (Rating + Popularity Trend)
                    # This balance ensures the AI recommends perfumes that are both
                    # semantically accurate AND widely loved / purchasable in the market.
                    community_quality = (norm_rating * 0.6) + (norm_pop * 0.4)
                    hit['hybrid_score'] = (norm_relevance * 0.75) + (community_quality * 0.25)
                hits_to_rerank.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
                hits = hits_to_rerank

        elif sort_by and len(hits) > 0:
            if sort_by == 'weighted_rating':
                # Apply same Rating+Popularity blending as reranker path
                for hit in hits:
                    norm_rating = min(1.0, hit.get('weighted_rating', 0) / 5.0)
                    norm_pop = hit.get('popularity_score', 0)
                    hit['_sort_score'] = (norm_rating * 0.70) + (norm_pop * 0.30)
                hits.sort(key=lambda x: x.get('_sort_score', 0), reverse=True)
            else:
                hits.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        return hits[:n_results]

if __name__ == "__main__":
    pass
