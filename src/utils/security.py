import re
import time
import streamlit as st
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Handles input validation, injection detection, and rate limiting for the Streamlit app.
    """
    
    def __init__(self, max_chars: int = 500, requests_per_min: int = 10):
        self.max_chars = max_chars
        self.requests_per_min = requests_per_min
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Injection patterns (enhanced with regex flexibility)
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous\s+)?instructions",
            r"forget\s+(all\s+)?(previous\s+)?(instructions|everything)",
            r"system\s+prompt",
            r"you\s+are\s+(now\s+)?(a\s+)?large\s+language\s+model",
            r"bypass(\s+filters)?",
            r"from\s+now\s+on",
            r"delete\s+all\s+files",
            r"exec\(",
            r"eval\(",
            r"<script>"
        ]

    def sanitize_input(self, text: str) -> str:
        """
        Cleans the input string.
        """
        if not text:
            return ""
        # Truncate and strip
        cleaned = text[:self.max_chars].strip()
        # Basic HTML tag removal
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        return cleaned

    def check_moderation(self, text: str) -> bool:
        """
        Calls OpenAI's free moderation API.
        Returns True if safe, False if violates policy (hate, violence, etc).
        """
        try:
            response = self.client.moderations.create(input=text)
            output = response.results[0]
            if output.flagged:
                logger.warning(f"Security Alert: Input flagged by OpenAI Moderation: {output.categories}")
                return False
            return True
        except Exception as e:
            logger.error(f"Moderation API Error: {e}")
            # Fallback to True to not block users if API is down, 
            # or False if you want strict security.
            return True

    def is_safe_input(self, text: str) -> bool:
        """
        Checks for potential prompt injection or malicious patterns locally.
        """
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Security Alert: Blocked suspicious pattern '{pattern}' in input.")
                return False
        return True

    def check_rate_limit(self) -> bool:
        """
        Checks if the current session has exceeded the rate limit.
        """
        now = time.time()
        if 'request_timestamps' not in st.session_state:
            st.session_state.request_timestamps = []
            
        st.session_state.request_timestamps = [
            t for t in st.session_state.request_timestamps if now - t < 60
        ]
        
        if len(st.session_state.request_timestamps) >= self.requests_per_min:
            return False
            
        st.session_state.request_timestamps.append(now)
        return True
