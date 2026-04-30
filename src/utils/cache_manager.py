import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, 
                 dynamic_cache_path: str = "dynamic_cache.json", 
                 static_cache_path: str = "quick_cache.json", 
                 max_size: int = 100):
        """
        Manages Static (Quick Suggestions) and Dynamic (LRU) tool-level caching.
        """
        self.dynamic_cache_path = dynamic_cache_path
        self.static_cache_path = static_cache_path
        self.max_size = max_size
        
        # In-memory storage
        self.static_cache = self._load_json(self.static_cache_path)
        self.dynamic_cache = self._load_json(self.dynamic_cache_path)
        
        logger.info(f"💾 Cache Initialized. Static: {len(self.static_cache)} items, Dynamic: {len(self.dynamic_cache)} items.")

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Safely loads a JSON file from disk."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache from {path}: {e}")
        return {}

    def _generate_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generates a stable hash key from tool name and sorted arguments."""
        # Normalize falsy values (None, {}, []) to None for consistent keys
        # This prevents cache misses from filters=None vs filters={} mismatch
        normalized = {}
        for k, v in args.items():
            if v is None or v == {} or v == []:
                normalized[k] = None
            else:
                normalized[k] = v
        args_str = json.dumps(normalized, sort_keys=True)
        raw_key = f"{tool_name}_{args_str}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        """Looks up a result in both static and dynamic caches."""
        key = self._generate_key(tool_name, args)
        
        # 1. Check static cache first
        if key in self.static_cache:
            logger.info(f"✨ [CACHE_HIT] [STATIC] {tool_name}")
            return self.static_cache[key]
            
        # 2. Check dynamic cache
        if key in self.dynamic_cache:
            # Update hit data (optional metadata)
            self.dynamic_cache[key]["last_accessed"] = datetime.now().isoformat()
            self.dynamic_cache[key]["hits"] = self.dynamic_cache[key].get("hits", 0) + 1
            logger.info(f"🚀 [CACHE_HIT] [DYNAMIC] {tool_name}")
            return self.dynamic_cache[key]["result"]
            
        return None

    def set(self, tool_name: str, args: Dict[str, Any], result: Any):
        """Sets a result in the dynamic cache with LRU logic."""
        key = self._generate_key(tool_name, args)
        
        # LRU: If at capacity and adding a new key, remove the oldest one
        if len(self.dynamic_cache) >= self.max_size and key not in self.dynamic_cache:
            try:
                oldest_key = min(self.dynamic_cache.keys(), 
                                 key=lambda k: self.dynamic_cache[k].get("last_accessed", "0"))
                del self.dynamic_cache[oldest_key]
            except Exception:
                self.dynamic_cache.pop(next(iter(self.dynamic_cache)))

        self.dynamic_cache[key] = {
            "result": result,
            "last_accessed": datetime.now().isoformat(),
            "hits": 1
        }

    def set_static(self, tool_name: str, args: Dict[str, Any], result: Any):
        """Explicitly sets a result in the static cache (for pre-warming)."""
        key = self._generate_key(tool_name, args)
        self.static_cache[key] = result
        
    def save_static_cache(self):
        """Saves memory-resident static cache back to disk (quick_cache.json)."""
        try:
            abs_path = os.path.abspath(self.static_cache_path)
            with open(self.static_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.static_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Static cache (quick) saved to: {abs_path}")
        except Exception as e:
            logger.error(f"Failed to save static cache: {e}")
        
    def save_dynamic_cache(self):
        """Saves memory-resident dynamic cache back to disk."""
        try:
            abs_path = os.path.abspath(self.dynamic_cache_path)
            with open(self.dynamic_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Dynamic cache saved to: {abs_path}")
        except Exception as e:
            logger.error(f"Failed to save dynamic cache: {e}")
