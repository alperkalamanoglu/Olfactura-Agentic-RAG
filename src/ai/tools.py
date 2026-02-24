import streamlit as st
import json
from typing import List, Dict, Any, Optional
from src.database.vector_db import VectorDatabase
from src.ai.formatters import format_price_tier, format_gender, format_longevity, format_sillage
from src.ai.logger import log_tool_call

# Global DB instance (will be initialized on first use)
_db = None

def get_db():
    global _db
    if _db is None:
        _db = VectorDatabase()
    return _db

def set_global_db(db_instance):
    """Dependency injection for Streamlit caching"""
    global _db
    _db = db_instance

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_notes_smart(record: dict) -> str:
    """Helper to extract notes correctly even if only 'general' notes exist, missing 'top/heart/base' structure."""
    def _format_flat(n_str: str) -> str:
        if n_str and n_str not in ['N/A', 'None']:
            # If the notes are separated by pipes, convert them into a markdown list
            if " | " in n_str:
                parts = [p.strip() for p in n_str.split(" | ")]
                # Create a newline-separated list with indentation
                return "\n" + "\n".join([f"    - {p}" for p in parts])
            
            # If the notes lack standard hierarchy, just return the raw string so it prints inline
            if "Top:" not in n_str and "Heart:" not in n_str and "Base:" not in n_str and "General:" not in n_str:
                return n_str
                
            return f"\n    - {n_str}"
        return 'N/A'

    notes = str(record.get('notes_str', '')).strip()
    if notes and notes not in ['N/A', 'None']:
        return _format_flat(notes)
        
    semantic = str(record.get('semantic_text', '')).strip()
    if 'Notes:' in semantic:
        after_notes = semantic.split('Notes:', 1)[-1].strip()
        if 'Keywords:' in after_notes:
            return _format_flat(after_notes.split('Keywords:')[0].strip(' .'))
        return _format_flat(after_notes.strip(' .'))
    return 'N/A'

# ---------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------



def _recommend_similar_impl(names_tuple: tuple, filters_json: Optional[str], n_results: int, additional_query: Optional[str] = None) -> str:
    """Cached similarity lookup"""
    names = list(names_tuple)
    filters = json.loads(filters_json) if filters_json else None
    
    try:
        db = get_db()
        collected_embeddings = []
        found_records = []
        found_names = []
        
        for name in names:
            res = db.get_by_name(name)
            record = res["record"]
            if record and 'embedding' in record:
                collected_embeddings.append(record['embedding'])
                found_names.append(record['name'])
                found_records.append(record)
        
        if not collected_embeddings:
            return f"Could not find any of the reference perfumes: {', '.join(names)}."
            
        def _create_query(rec):
            family = rec.get('family', '')
            accords = rec.get('accords_str', '')
            notes = rec.get('notes_str', '')
            return f"{family} perfume. Accords: {accords}. Notes: {notes}".strip()

        if len(collected_embeddings) == 1:
            centroid = collected_embeddings[0]
            query_text = _create_query(found_records[0])
            strategy_desc = f"Similar to {found_names[0]}"
        else:
            # Multi-perfume centroid search
            num_vecs = len(collected_embeddings)
            centroid = [sum(col) / num_vecs for col in zip(*collected_embeddings)]
            query_text = "Blend of: " + " | ".join([_create_query(r) for r in found_records])
            strategy_desc = f"Blend of {', '.join(found_names)}"
            
        if additional_query:
            strategy_desc += f", modified by '{additional_query}'"
            try:
                # Calculate weighted embedding: 85% original concept, 15% new requirements
                # High anchor percentage prevents the centroid from drifting into cheap generic aquatics
                add_embedding = db.embedding_manager.get_embeddings([additional_query])[0]
                centroid = [(r * 0.85) + (a * 0.15) for r, a in zip(centroid, add_embedding)]
            except Exception:
                pass # fallback to original centroid if embedding fails

        # 1. Primary Retrieval (Broad Pool based on Cosine Similarity to Spatial Centroid)
        results_raw = db.search_by_embedding(
            query_embedding=centroid, 
            query_text=None, # Bypass database-level reranker to retain granular control
            filters=filters, 
            n_results=100 
        )
        
        # 2. Dynamic Semantic Thresholding
        # If the user asks for "Aventus but more powdery", the inherent contradiction might 
        # prevent any single perfume from achieving >70% similarity to the centroid. 
        # Therefore, if a modifier query exists, we lower the initial cosine threshold 
        # and delegate the final judgment to the Cross-Encoder Reranker.
        valid_candidates = []
        min_cosine_threshold = 0.50 if additional_query else 0.70 
        
        for p in results_raw:
            if p['name'] in found_names: continue
            if p.get('initial_cosine_score', 0) >= min_cosine_threshold:
                valid_candidates.append(p)
                
        # 3. Cross-Encoder Reranking Phase 
        # (Re-sort the elite candidates contextually based on the new concept, e.g. "more masculine")
        if additional_query and len(valid_candidates) > 0:
            docs_for_rerank = [p['semantic_text'] for p in valid_candidates]
            import time
            t_start = time.time()
            
            # Formulate a QA-style natural query for the MS-MARCO TinyBERT Reranker
            # We must teach the reranker what the anchor actually smells like (e.g. Aventus = fruity, sweet, leather, smoky)
            anchor_profiles = " | ".join([f"{r['name']} (Profile: {r.get('accords_str', '')})" for r in found_records])
            natural_rerank_query = f"I am looking for a perfume with a base DNA of {anchor_profiles}, but modified to heavily feature {additional_query} characteristics."
            
            rerank_scores = db.reranker.rerank(natural_rerank_query, docs_for_rerank)
            
            try:
                from src.ai.logger import log_event, agent_logger
                log_event(agent_logger, "PERFORMANCE", "Agent Rerank Executed", {"seconds": round(time.time() - t_start, 3), "docs": len(docs_for_rerank)})
            except: pass
            
            for i, p in enumerate(valid_candidates):
                raw_logit = rerank_scores[i]
                p['relevance_score'] = raw_logit
                # Logit to 0-1 probability
                norm_relevance = 1 / (1 + pow(2.718, -(raw_logit + 4.0) / 2.0))
                
                # Blend the original DNA adherence (cosine) with the new modification strength (Relevance)
                # This ensures the Match Percentage is a true reflection of "Hybrid Similarity"
                p['hybrid_score'] = (p.get('initial_cosine_score', 0) * 0.45) + (norm_relevance * 0.55)
                
            # Yüksekten düşüğe rerank skoruyla diz
            valid_candidates.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        else:
            # Ek query yoksa saf benzerlikle (cosine) diz
            valid_candidates.sort(key=lambda x: x.get('initial_cosine_score', 0), reverse=True)
            
        results = valid_candidates[:n_results]
        
        if not results:
            return f"I applied your constraints to {strategy_desc}, but could not find any perfume in our database that closely matches the original scent DNA without straying too far into unrelated scent profiles."
            
        # Format output (Markdown)
        output = f"### Top recommendations ({strategy_desc}):\n\n"
        for i, p in enumerate(results, 1):
            # Calculate match percentage properly
            if additional_query and 'hybrid_score' in p:
                match_pct = int(p.get('hybrid_score', 0) * 100)
            else:
                match_pct = int(p.get('initial_cosine_score', 0) * 100)
                
            output += f"### {i}. {p['name']} by {p['brand']} (Match: ~{match_pct}%)\n"
            output += f"- **Rating:** {p.get('rating', 'N/A')}/5 ({p.get('votes', 0):,} Global Community Reviews)\n"
            output += f"- **Year:** {p.get('year') if p.get('year') not in [0, '0', None] else '-'}\n"
            output += f"- **Gender:** {format_gender(p.get('gender_score'))}\n- **Price:** {format_price_tier(p.get('price_tier_score'))}\n"
            
            # Use structured metadata for cleaner output
            accords = p.get('accords_str', 'N/A')
            notes = extract_notes_smart(p)
            output += f"- **Accords:** {accords}\n"
            output += f"- **Notes:** {notes}\n"
            output += "---\n\n"
            
        return output
    except Exception as e:
        return f"Error in recommend_similar: {str(e)}"

# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def search_perfumes(query: Optional[str] = None, 
                    filters: Optional[Dict] = None, 
                    sort_by: Optional[str] = None, 
                    n_results: int = 3,
                    excluded_notes: Optional[List[str]] = None) -> str:
    """Searches the vector database for perfumes matching the query and filters."""
    # Catch cases where LLM mistakenly puts 'excluded_notes' inside 'filters'
    if filters and "excluded_notes" in filters:
        if not excluded_notes:
            excluded_notes = filters.pop("excluded_notes")
        else:
            del filters["excluded_notes"]

    clean_query = query.strip().lower() if query else None
    
    try:
        results = get_db().search(
            query=clean_query,
            filters=filters,
            sort_by=sort_by,
            n_results=n_results,
            excluded_notes=excluded_notes
        )
        
        if not results:
            return "No perfumes found matching your criteria."
            
        # --- QUALITY FILTER (Anti-Hallucination) ---
        if clean_query:
            top_score = results[0].get('hybrid_score', results[0].get('initial_cosine_score', 0))
            if top_score < 0.30:
                return "MATCH_QUALITY_TOO_LOW: I found some distant relatives, but nothing that truly matches your specific query. Rather than suggesting irrelevant scents, I recommend trying more common scent profiles or different keywords."
            
        # Format results (Markdown)
        formatted_output = ""
        for i, p in enumerate(results, 1):
            if clean_query and p.get('hybrid_score', 0) > 0:
                match_pct = int(p.get('hybrid_score', 0) * 100)
                match_label = "AI Match"
            else:
                quality = p.get('_sort_score', p.get('weighted_rating', 0) / 5.0)
                match_pct = int(min(1.0, quality) * 100)
                match_label = "Quality"
            perfume_text = f"### {i}. {p['name']} by {p['brand']} ({match_label}: {match_pct}%)\n"
            perfume_text += f"- **Rating:** {p.get('rating', 'N/A')}/5 ({p.get('votes', 0):,} Global Community Reviews)\n"
            perfume_text += f"- **Year:** {p.get('year') if p.get('year') not in [0, '0', None] else '-'}\n"
            perfume_text += f"- **Gender:** {format_gender(p.get('gender_score'))}\n- **Price:** {format_price_tier(p.get('price_tier_score'))}\n"
            
            accords = p.get('accords_str', 'N/A')
            notes = extract_notes_smart(p)
            perfume_text += f"- **Accords:** {accords}\n"
            perfume_text += f"- **Notes:** {notes}\n"
            perfume_text += "---\n\n"
            
            formatted_output += perfume_text
            
        log_tool_call("search_perfumes", {"query": clean_query, "excluded": excluded_notes})
        return formatted_output
        
    except Exception as e:
        return f"Error searching perfumes: {str(e)}"

def get_perfume_details(perfume_name: str) -> str:
    """Retrieves full details for a single perfume from the database."""
    clean_name = perfume_name.strip().lower()
    try:
        res = get_db().get_by_name(clean_name)
        record = res["record"]
        suggestions = res["suggestions"]

        if not record:
            if suggestions:
                return f"I couldn't find an exact match for '{perfume_name}'. Did you mean one of these: {', '.join(suggestions)}?"
            return f"Could not find a perfume named '{perfume_name}'."
        
        output = f"### {record['name']} by {record['brand']}\n"
        output += f"**Rating:** {record.get('rating', 'N/A')}/5 ({record.get('votes', 0):,} Global Community Reviews)\n"
        output += f"**Year:** {record.get('year') if record.get('year') not in [0, '0', None] else '-'}\n"
        output += f"**Gender:** {format_gender(record.get('gender_score', 0.5))}\n**Price:** {format_price_tier(record.get('price_tier_score', 5))}\n"
        output += f"**Longevity:** {format_longevity(record.get('longevity_score', 5))}\n"
        output += f"**Sillage:** {format_sillage(record.get('sillage_score', 5))}\n"
        output += f"**Family:** {record.get('family', 'N/A')}\n"
        output += f"**Accords:** {record.get('accords_str', 'N/A')}\n"
        output += f"**Notes:** {extract_notes_smart(record)}\n"
        
        log_tool_call("get_perfume_details", {"name": clean_name})
        return output
    except Exception as e:
        return f"Error getting perfume details: {str(e)}"

def recommend_similar(reference_perfume_names: Any, 
                      additional_query: Optional[str] = None,
                      filters: Optional[Dict] = None, 
                      n_results: int = 3) -> str:
    """Finds similar perfumes using vector centroid search and optional reranking."""
    if isinstance(reference_perfume_names, str):
        if reference_perfume_names.startswith("["):
            try:
                names = json.loads(reference_perfume_names)
            except:
                names = [reference_perfume_names]
            else:
                names = [reference_perfume_names]
    else:
        names = reference_perfume_names
        
    clean_names = tuple(sorted([str(n).strip().lower() for n in names]))
    filters_json = json.dumps(filters, sort_keys=True) if filters else None
    
    result = _recommend_similar_impl(clean_names, filters_json, n_results, additional_query)
    
    log_tool_call("recommend_similar", {
        "names": clean_names, 
        "additional_query": additional_query,
        "filters": filters
    })
    
    return result


def _compare_perfumes_impl(perfume_names: tuple) -> str:
    """The actual database heavy-lifting for comparison (Cached)"""
    db = get_db()
    results = []
    
    for name in perfume_names:
        clean_name = name.strip().lower()
        res = db.get_by_name(clean_name)
        record = res.get("record")
        
        if record:
            results.append(record)
        else:
            # If not found, try suggestions
            suggestions = res.get("suggestions", [])
            if suggestions:
                # Try first suggestion
                res2 = db.get_by_name(suggestions[0])
                if res2.get("record"):
                    results.append(res2["record"])
            
    if not results:
        return "Could not find details for those perfumes."
    
    # Format output with human-readable labels (Markdown)
    output = ""
    for i, p in enumerate(results, 1):
        output += f"### {i}. {p['name']} by {p['brand']}\n"
        output += f"**Rating:** {p.get('rating', 'N/A')}/5 ({p.get('votes', 0):,} Global Community Reviews)\n"
        output += f"**Year:** {p.get('year') if p.get('year') not in [0, '0', None] else '-'}\n"
        output += f"**Gender:** {format_gender(p.get('gender_score', 0.5))}\n**Price:** {format_price_tier(p.get('price_tier_score', 5))}\n"
        output += f"**Longevity:** {format_longevity(p.get('longevity_score', 5))}\n"
        output += f"**Sillage:** {format_sillage(p.get('sillage_score', 5))}\n"
        output += f"**Family:** {p.get('family', 'N/A')}\n"
        output += f"**Accords:** {p.get('accords_str', 'N/A')}\n"
        output += f"**Notes:** {extract_notes_smart(p)}\n"
        
        # Pull semantic text to extract Community Pros (first 3 Keywords)
        semantic = p.get("semantic_text", "")
        
        if semantic and "Keywords:" in semantic:
            keywords_text = semantic.split("Keywords:")[-1].strip()
            pros = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            if pros:
                top_pros = ", ".join(pros[:3])
                output += f"**Top Community Pros:** {top_pros}\n\n"
            else:
                output += "\n"
        else:
            output += "\n"
            
        output += "---\\n" # Divider
        
    return output

def compare_perfumes(perfume_names: List[str]) -> str:
    """Comparison tool - retrieves full details for multiple perfumes."""
    clean_names = tuple([str(n).strip() for n in perfume_names])
    
    result = _compare_perfumes_impl(clean_names)
    
    log_tool_call("compare_perfumes", {"names": clean_names})
    
    return result
