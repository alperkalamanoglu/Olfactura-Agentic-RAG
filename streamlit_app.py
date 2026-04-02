"""
The Scent Advisor - Premium Streamlit UI v3
Features: Streaming, Tool Call Panel, Perfume Cards, Enhanced Styling, Structured Logging
"""
import streamlit as st
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from src.ai.agent import PerfumeAgent
from src.utils.security import SecurityManager

# Page config
st.set_page_config(
    page_title="Olfactura",
    page_icon="🌸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.database.vector_db import VectorDatabase
from src.ai.tools import set_global_db




# ---------------------------------------------------------
# SUGGESTION → QUERY MAPPING
# ---------------------------------------------------------
# Each Quick Suggestion button maps to a fixed search query.
# For "Like Baccarat Rouge", the value is used as reference_perfume_names (recommend_similar tool).
SUGGESTION_MAP = {
    # Occasions
    "🍷 Romantic Date Night": "romantic date night warm sensual seductive amber musk oriental",
    "👔 Office & Professional": "office professional fresh clean crisp woody subtle inoffensive",
    "🌴 Summer Beach Vacation": "beach tropical coconut marine salty solar fresh",
    "💒 Wedding Day Elegance": "wedding ceremony elegant romantic timeless sophisticated white floral",
    
    # Vibes & Esthetics
    "🔮 Dark & Mysterious": "dark mysterious smoky intense oud",
    "☔ Cozy Rainy Day": "rainy day petrichor green earthy ozonic woody cozy atmospheric",
    "🦄 Unique & Niche": "niche unique artistic statement unconventional complex rare obscure",
    "🧁 Edible & Gourmand": "edible gourmand sweet cake pastry dessert cookie sugar honey",
    
    # Recommend Similar
    "🧬 Like Baccarat Rouge but cheaper": "Baccarat Rouge 540",
    "🧬 Like YSL Black Opium": "Black Opium",
    "🧬 Like By Kilian Angels' Share": "Angels' Share",
    "🧬 Like Creed Aventus but cheaper": "Aventus",
    
    # Compare Classics
    "🆚 Bleu de Chanel vs Dior Sauvage": ["Bleu de Chanel", "Sauvage"],
    "🆚 Tom Ford Oud Wood vs Versace Oud Noir": ["Oud Wood", "Oud Noir"],
    
    # Scent Profiles
    "🍃 Fresh, Citrus & Aquatic": "fresh clean light aquatic crisp citrus",
    "☕ Coffee, Caramel & Vanilla": "coffee vanilla caramel gourmand sweet",
    "🌲 Deep Woods & Spices": "woody spicy sandalwood cedar pepper vetiver warm earthy",
    "🌹 White Florals & Rose": "sheer airy dewy transparent floral rose jasmine fresh modern white floral",
    "🍒 Boozy Cherry & Almond": "boozy cherry almond liqueur amaretto syrupy sweet",
}

# ---------------------------------------------------------
# RESOURCE CACHING (CRITICAL FOR PERFORMANCE)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Engine & Reranker... 🧠")
def init_resources():
    """
    Loads ChromaDB and the Reranker model into memory ONCE.
    This cached instance is shared across user sessions.
    """
    try:
        db = VectorDatabase()
        # Trigger lazy loading of Reranker and perform a WARMUP INFERENCE
        # First inference always compiles the ONNX/PyTorch graph, which causes a spike. Doing this hides the spike behind the startup spinner.
        _ = db.reranker 
        try:
            db.reranker.rerank("warmup query", ["warmup document"])
        except Exception:
            pass
        return db
    except Exception as e:
        st.error(f"Failed to load AI resources: {e}")
        return None

# Load & Inject Dependency
db_instance = init_resources()
if db_instance:
    set_global_db(db_instance)

# Premium Light Mode CSS
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
/* Root variables - LIGHT MODE */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f8f8;
    --bg-tertiary: #f0f0f0;
    --accent-gold: #d4af37;
    --accent-rose: #c9a0a0;
    --text-primary: #0a0a0f;
    --text-secondary: #4a4a4a;
    color-scheme: light !important;
}
/* Hide 'Press Enter to submit' hint */
[data-testid="InputInstructions"] {
    display: none !important;
}
/* Main background */
.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, var(--bg-tertiary) 100%);
    color: var(--text-primary) !important;
}
/* Force all text to be visible */
.stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label {
    color: var(--text-primary) !important;
}
.stMarkdown {
    color: var(--text-primary) !important;
}
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Title styling */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-rose) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.3rem;
}
.subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: var(--text-secondary) !important;
    text-align: center;
    margin-bottom: 1.5rem;
}
/* Hide Streamlit Header Anchor Link Icons (Force all H tags anchor links to hide) */
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
    display: none !important;
    pointer-events: none !important;
}
.stMarkdown a.header-anchor {
    display: none !important;
    pointer-events: none !important;
}
/* User message container */
/* Force scrollbar always visible to prevent layout shift during streaming */
html {
    overflow-y: scroll !important;
}
.user-msg-container {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
    width: 100%;
}
/* User message */
.user-msg {
    background: linear-gradient(135deg, rgba(212, 175, 55, 0.25) 0%, rgba(212, 175, 55, 0.15) 100%);
    border: 1px solid rgba(212, 175, 55, 0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    color: #0a0a0f !important;
    max-width: 75%;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    display: inline-block;
    word-break: break-word;
}
/* Assistant Chat Message styling to make it a bubble */
[data-testid="stChatMessage"] {
    background: rgba(0, 0, 0, 0.03) !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 1.2rem !important;
    margin: 0.8rem 0 !important;
    /* Use width+box-sizing instead of max-width to prevent right-edge reflow during streaming */
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03) !important;
    color: #0a0a0f !important;
    gap: 0 !important;
}
/* Safely hide the avatar box (first child of the stChatMessage flex row) */
[data-testid="stChatMessage"] > div:first-child {
    display: none !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f5f5 0%, #e8e8e8 100%) !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #0a0a0f !important;
}
/* (Tooltip CSS moved to conditional logic below) */
/* Input styling - Transparent outer wrappers fix iOS dark corner bleed */
.stTextInput > div, 
.stTextInput > div > div,
.stTextInput > div[data-baseweb="input"] {
    background-color: transparent !important;
}
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    border: 1px solid rgba(0, 0, 0, 0.2) !important;
    border-radius: 25px !important;
    color: #000000 !important;
    padding: 0.8rem 1.2rem !important;
    caret-color: #000000 !important;
}
.stTextInput > div > div > input::placeholder {
    color: #666666 !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 15px rgba(212, 175, 55, 0.25) !important;
}
/* Button styling - Submit */
.stFormSubmitButton > button {
    -webkit-appearance: none !important;
    appearance: none !important;
    background-color: var(--accent-gold) !important;
    background: linear-gradient(135deg, var(--accent-gold) 0%, #b8962e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.35) !important;
}
/* Normal buttons (Quick Suggestions & Reset) */
.stButton > button {
    -webkit-appearance: none !important;
    appearance: none !important;
    background: #ffffff !important;
    color: #4a4a4a !important;
    border: 1px solid rgba(212, 175, 55, 0.4) !important;
    border-radius: 15px !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
}
.stButton > button:hover {
    background: rgba(212, 175, 55, 0.1) !important;
    color: #0a0a0f !important;
    border-color: #d4af37 !important;
    transform: translateY(-1px) !important;
}
/* Dropdown/Select Box Styling */
.stSelectbox > div > div > div,
.stRadio > div[role="radiogroup"],
div[data-baseweb="select"] > div {
    background-color: #fcfcfc !important;
    color: #0a0a0f !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 12px !important;
}
/* Slider Bar Container Background */
.stSlider > div {
    background: rgba(0, 0, 0, 0.03) !important;
    padding: 5px 10px !important;
    border-radius: 15px !important;
}
/* Header styling in Light Mode */
header[data-testid="stHeader"] {
    background: transparent !important;
}
header[data-testid="stHeader"] * {
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] span {
    color: #0a0a0f !important;
}
div[role="listbox"] {
    background-color: #ffffff !important;
}
div[role="option"] {
    color: #0a0a0f !important;
    background-color: #ffffff !important;
}
div[role="option"]:hover {
    background-color: rgba(212, 175, 55, 0.15) !important;
}
/* Radio buttons */
.stRadio label {
    color: #0a0a0f !important;
}
/* Slider labels */
.stSlider label {
    color: #0a0a0f !important;
}
/* FIX: st.pills readability when Safari OS is in Dark Mode but Streamlit is Light */
/* Pills outer containers - transparent to block dark mode bleed */
[data-testid="stPills"],
[data-testid="stPills"] > div,
[data-testid="stPills"] > div > div {
    background-color: transparent !important;
}
[data-testid="stPills"] > div > div > div[role="button"]:not([aria-pressed="true"]) * {
    color: #4a4a4a !important;
}
[data-testid="stPills"] > div > div > div[role="button"] {
    background-color: #fcfcfc !important; 
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
}
[data-testid="stPills"] > div > div > div[role="button"][aria-pressed="true"] {
    background-color: rgba(212, 175, 55, 0.15) !important;
    border-color: #d4af37 !important;
}
[data-testid="stPills"] > div > div > div[role="button"][aria-pressed="true"] * {
    color: #927517 !important;
    font-weight: 600 !important;
}
/* DISABLED STATE FIXES - CRITICAL */
.stButton > button:disabled, 
.stFormSubmitButton > button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    background: #ddd !important;
    color: #999 !important;
    box-shadow: none !important;
    transform: none !important;
}
.stTextInput > div > div > input:disabled {
    opacity: 0.6 !important;
    background-color: #f0f0f0 !important;
    color: #999 !important;
    cursor: not-allowed !important;
}
/* Sticky Input Container Style */
div[data-testid="stBottom"] > div {
    background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 20%);
    padding-bottom: 20px;
}

/* MOBILE RESPONSIVE FIX: Force 2-column layout on mobile */
/* Streamlit 1.x uses data-testid="stColumn", older used "column" - cover both */
@media (max-width: 768px) {
    /* Force horizontal block to stay as a row */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }

    /* By default all columns inside a block become 50% width on phone */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: calc(50% - 0.4rem) !important;
        flex: 1 1 calc(50% - 0.4rem) !important;
        min-width: calc(50% - 0.4rem) !important;
        max-width: calc(50% - 0.4rem) !important;
    }

    /* EXCEPTION: Chat Input Form - override: keep text input wide, button narrow */
    /* Target the columns SPECIFICALLY inside the form container */
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
        width: 80% !important;
        flex: 1 1 80% !important;
        min-width: 80% !important;
        max-width: 80% !important;
    }
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
    div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
        width: calc(20% - 0.4rem) !important;
        flex: 1 1 calc(20% - 0.4rem) !important;
        min-width: calc(20% - 0.4rem) !important;
        max-width: calc(20% - 0.4rem) !important;
    }

    /* Compact suggestion buttons to fit in half-width columns */
    .stButton > button {
        padding: 0.35rem 0.2rem !important;
        font-size: 0.72rem !important;
        white-space: normal !important;
        line-height: 1.2 !important;
        height: auto !important;
        min-height: 2.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    # GPU Health Check (runs ONCE per session, not per message)
    use_gpu = False
    vllm_url = os.getenv("VLLM_BASE_URL", "")
    if vllm_url:
        try:
            import requests
            # Strip /v1 suffix for health check endpoint
            health_url = vllm_url.rstrip("/").replace("/v1", "") + "/health"
            resp = requests.get(health_url, timeout=3.0)
            use_gpu = resp.status_code == 200
        except Exception:
            use_gpu = False
    
    st.session_state.use_gpu = use_gpu

    try:
        st.session_state.agent = PerfumeAgent(use_gpu=use_gpu)
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
    
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "✨ **Welcome! I am your personal AI fragrance consultant, Olfactura.**\n\nI can help you find your perfect signature scent for **everyday wear, the office, or a special date night**.\n\nNot sure where to start? You can try one of the following:\n- *\"Can you recommend some fresh and clean summer scents?\"*\n- *\"Perfumes similar to Baccarat Rouge 540 but much cheaper\"*\n- *\"Can you recommend some floral perfumes but with no rose?\"*\n\nYou can also click on one of the **Quick Suggestions** in the left sidebar to get started instantly!\n\n**What kind of fragrance are you looking for today?**"
        }
    ]

if "tool_history" not in st.session_state:
    st.session_state.tool_history = []

if "latest_search" not in st.session_state:
    st.session_state.latest_search = None



if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "last_gender_filter" not in st.session_state:
    st.session_state.last_gender_filter = []
    
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

import ast

# Helper functions
def lock_ui():
    """Callback to lock UI state and retrieve text from text_input before rerun."""
    text = st.session_state.get("user_input_string", "")
    if text and text.strip():
        st.session_state.pending_input = text.strip()
        st.session_state.is_processing = True

def submit_question():
    """Callback to handle question submission (Not used directly by form due to its nature, kept for Suggestions)"""
    pass

def set_suggestion(msg):
    """Callback for suggestion buttons. Appends [CACHED_QUERY] tag for deterministic tool queries."""
    # Look up the canonical query for this suggestion
    cached_query = SUGGESTION_MAP.get(msg)
    if cached_query:
        st.session_state.pending_input = f"{msg} [CACHED_QUERY: {cached_query}]"
    else:
        st.session_state.pending_input = msg
    st.session_state.is_processing = True

# Helper functions
def save_feedback(query, response, rating, comment="", context=None):
    """
    Save user feedback.
    Saves locally to JSONL, AND pushes to HuggingFace Dataset if HF_TOKEN is set.
    """
    import uuid
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "rating": rating,
        "comment": comment,
        "context": context or {}, # Extra details like gender filter, tools used, etc.
        "response": response # Full response (no truncation) for deep analysis
    }
    
    # 1. Local Save (Works on your PC, wiped on HF restart)
    try:
        feedback_dir = "user_feedbacks"
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_path = os.path.join(feedback_dir, "feedback_logs.jsonl")
        
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Local save error: {e}")

    # 2. HuggingFace Dataset Save (Permanent Cloud Storage)
    hf_token = os.environ.get("HF_TOKEN")
    hf_dataset_repo = os.environ.get("HF_DATASET_REPO") # e.g. "alper/olfactura-logs"
    
    if hf_token and hf_dataset_repo:
        try:
            from huggingface_hub import HfApi
            import tempfile
            
            api = HfApi()
            file_name = f"data/log_{log_entry['id']}.json"
            
            # Create a temporary file to upload
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                json.dump(log_entry, f)
                temp_path = f.name
                
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=file_name,
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                token=hf_token
            )
            os.unlink(temp_path) # Clean up
        except Exception as e:
            print(f"HuggingFace Dataset Upload Error: {e}")

def log_interaction(query, response, context=None):
    """
    DEPRECATED: kept for backwards compat but does nothing now.
    Logging is handled by agent_activity.log (uploaded via upload_backend_logs).
    """
    pass

def upload_backend_logs():
    """
    Uploads the full agent_activity.log to HuggingFace Dataset after each message.
    This replaces the old per-message JSON approach — single file, no separate commits.
    Runs in a background thread so it never blocks the UI.
    """
    import threading

    def _upload():
        hf_token = os.environ.get("HF_TOKEN")
        hf_dataset_repo = os.environ.get("HF_DATASET_REPO")
        if not (hf_token and hf_dataset_repo):
            return
        try:
            from huggingface_hub import HfApi
            from src.ai.logger import LOG_FILE  # canonical path set by logger.py
            api = HfApi()
            if os.path.exists(LOG_FILE):
                api.upload_file(
                    path_or_fileobj=LOG_FILE,
                    path_in_repo="logs/agent_activity.log",
                    repo_id=hf_dataset_repo,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message="log sync"
                )
        except Exception as e:
            print(f"Agent Log Upload Error: {e}")

    threading.Thread(target=_upload, daemon=True).start()

def extract_tool_calls(agent):
    """Extract tool calls from agent conversation history."""
    calls = []
    # We look at the last few messages
    for msg in agent.conversation_history[-5:]:
        if isinstance(msg, dict):
            # Regular dict message (This handles the new streaming tool outputs)
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        calls.append({
                            "name": tc["function"]["name"],
                            "args": args
                        })
                    except:
                        pass
        else:
            # Pydantic object (ChatCompletionMessage)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        calls.append({
                            "name": tc.function.name,
                            "args": args
                        })
                    except:
                        pass
    return calls

def extract_latest_search_results(agent):
    """
    Parses the last tool output to find search results.
    Returns a list of perfume dictionaries if found.
    """
    import re
    # Scan history backwards for tool outputs
    for msg in reversed(agent.conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content", "")
            
            # Look for [RAW_DATA]...[/RAW_DATA] block
            match = re.search(r'\[RAW_DATA\](.*?)\[/RAW_DATA\]', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except:
                    continue
            
            # Fallback: Try old literal_eval method
            try:
                data = ast.literal_eval(content)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                     if 'name' in data[0] and 'brand' in data[0]:
                         return data
            except:
                continue
    return None

def display_perfume_cards(results):
    """Renders perfume results as a grid of stylized cards."""
    st.markdown("### 🔍 Search Results")
    
    # Grid Layout (3 columns)
    cols = st.columns(3)
    
    for i, perfume in enumerate(results[:9]): # Limit to 9 cards
        with cols[i % 3]:
            # Rating to Star
            rating = perfume.get('rating', 0) or 0
            stars = "⭐" * int(rating)
            
            # Semantic Score badge with detailed breakdown
            score_badge = ""
            score_details = ""
            if 'hybrid_score' in perfume:
                 # Map 0-1 score to 0-100%
                 match_pct = int(perfume['hybrid_score'] * 100)
                 
                 # Extract individual components
                 relevance_raw = perfume.get('relevance_score', 0)
                 # Sigmoid normalization
                 relevance_norm = 1 / (1 + pow(2.718, -relevance_raw))
                 relevance_pct = int(relevance_norm * 100)
                 
                 rating_val = perfume.get('weighted_rating', 0)
                 rating_pct = int((rating_val / 5.0) * 100)
                 
                 # Determine weights based on context (this should match vector_db.py logic)
                 # For now, assume standard semantic search (70% relevance, 30% rating)
                 w_rel = 70
                 w_rat = 30
                 
                 # Build score display HTML inline
                 score_html = f"""
                 <div style='background: rgba(100, 100, 255, 0.15); border: 1px solid rgba(100, 100, 255, 0.4); border-radius: 8px; padding: 6px 10px; margin: 8px 0; font-size: 0.75rem;'>
                     <strong>🎯 AI Match: {match_pct}%</strong>
                 </div>
                 <div style='font-size: 0.7rem; color: #aaa; margin-top: 5px; line-height: 1.4;'>
                     <div style='color: #90EE90;'>📊 Semantic Relevance: {relevance_pct}%</div>
                     <div style='color: #d4af37;'>⭐ Quality Rating: {rating_pct}% ({rating_val:.2f}/5)</div>
                     <div style='color: #888; font-size: 0.65rem; margin-top: 3px;'>Formula: {w_rel}% × Semantic + {w_rat}% × Rating</div>
                 </div>
                 """
            elif 'score' in perfume:
                 # Distance to similarity %
                 sim = max(0, (1 - perfume['score']) * 100)
                 score_html = f"<span class='tool-badge'>Match: {int(sim)}%</span>"
            else:
                 score_html = ""
            
            # Build complete card HTML
            import textwrap
            
            # Use dedent to remove indentation preventing Markdown code-block interpretation
            card_html = textwrap.dedent(f"""
            <div style="
                background: rgba(255, 255, 255, 0.05); 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 15px;
                height: 100%;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; color: #666; font-size: 0.8rem; margin-bottom: 5px;">
                    <span>{perfume.get('brand')}</span>
                    <span style="background: rgba(212, 175, 55, 0.1); color: #d4af37; padding: 2px 6px; border-radius: 4px;">{perfume.get('year') if (perfume.get('year') and str(perfume.get('year')) != '0') else 'N/A'}</span>
                </div>
                <div style="font-weight: bold; font-size: 1.1rem; color: #f5f5f5; margin-bottom: 5px;">
                    {perfume.get('name')}
                </div>
                <div style="font-size: 0.9rem; color: #d4af37; margin-bottom: 10px;">
                    {rating:.1f} {stars}
                </div>
                {score_html}
                <div style="font-size: 0.8rem; color: #aaa; margin-top: 10px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;">
                    {perfume.get('description', 'No description available.')[:100]}...
                </div>
            </div>
            """)
            st.markdown(card_html, unsafe_allow_html=True)

# Main content with header
st.markdown('<h1 class="main-title">🌸 Olfactura</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Fragrance Consultant</p>', unsafe_allow_html=True)

st.markdown("---")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            import re as _re
            # CACHED_QUERY can contain arrays like ['Perfume A', 'Perfume B'], so .* safely strips to the end
            _display = _re.sub(r'\s*\[CACHED_QUERY:.*$', '', message["content"]).strip()
            st.markdown(f'''
            <div class="user-msg-container">
                <div class="user-msg">
                    {_display}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            # Check for tool calls
            tool_html = ""
            if message.get("tools"):
                tool_names = message["tools"]
                natural_tools = []
                for t in tool_names:
                    if "search_perfumes" in t: natural_tools.append("Perfume Search")
                    elif "hybrid" in t: natural_tools.append("AI Suggestion Engine")
                    else: natural_tools.append(t.replace("_", " ").title())
                
                tools_str = ", ".join(natural_tools)
                tool_html = f'<div style="font-size: 0.75rem; color: #888; margin-bottom: 8px;">🔧 {tools_str} tool was used</div>'
            
            with chat_container.chat_message("assistant", avatar="✨"):
                if tool_html:
                    st.markdown(tool_html, unsafe_allow_html=True)
                st.markdown(message['content'])


# Initialize Feedback Session States
if "feedback_completed" not in st.session_state:
    st.session_state.feedback_completed = set()

# Display Feedback Widget (under last message)
feedback_placeholder = st.empty()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and not st.session_state.is_processing:
    # Use the response text as a unique identifier for feedback
    current_response_id = hash(st.session_state.messages[-1]["content"])
    
    with feedback_placeholder.container():
        # Only show the feedback widget if we haven't already completed feedback for THIS response
        if current_response_id not in st.session_state.feedback_completed:
            # Inline feedback: icons + text on same line
            feedback = st.feedback("thumbs")
            st.markdown("<span style='font-size: 0.8rem; color: #888; margin-left: 4px;'>How was this recommendation?</span>", unsafe_allow_html=True)
            
            if feedback is not None:
                # Capture useful analytics context at the time of feedback
                analytics_context = {
                    "gender_filter_active": st.session_state.get("last_gender_filter", []),
                    "tools_invoked": st.session_state.messages[-1].get("tools", []) if st.session_state.messages else []
                }
                
                if feedback == 0: # Downvoted
                    with st.expander("Help us improve! What was wrong?", expanded=True):
                        comment = st.text_area("Details (optional):", placeholder="e.g. Too sweet, wrong season...")
                        if st.button("Send Feedback"):
                            save_feedback(st.session_state.last_query, st.session_state.last_response, "Dislike", comment, context=analytics_context)
                            st.session_state.feedback_completed.add(current_response_id)
                            st.session_state.feedback_thanks_msg = "Thank you! Your feedback has been recorded safely."
                            st.rerun()
                else: # Upvoted
                    save_feedback(st.session_state.last_query, st.session_state.last_response, "Like", context=analytics_context)
                    st.session_state.feedback_completed.add(current_response_id)
                    st.session_state.feedback_thanks_msg = "Glad you liked it! 🌸 Thank you for the positive feedback."
                    st.rerun()
        else:
            # Feedback was already completed for this widget. 
            # If we just completed it and have a thank you message in queue, show it with auto-fade HTML/JS
            if "feedback_thanks_msg" in st.session_state:
                thanks_msg = st.session_state.feedback_thanks_msg
                html_code = f"""
                <div id="feedback-success-banner" style="margin-top: 15px; padding: 0.6rem 1rem; border-radius: 8px; background-color: rgba(40, 167, 69, 0.1); border: 1px solid rgba(40, 167, 69, 0.3); color: #28a745; font-weight: 500; transition: opacity 0.5s ease; width: fit-content; display: inline-block;">
                    {thanks_msg}
                </div>
                <script>
                    setTimeout(function() {{
                        var el = document.parentWindow.document.getElementById('feedback-success-banner') || 
                                 window.parent.document.getElementById('feedback-success-banner') ||
                                 document.getElementById('feedback-success-banner');
                        if (el) {{
                            el.style.opacity = '0';
                            setTimeout(function() {{ el.style.display = 'none'; }}, 500);
                        }}
                    }}, 2500);
                </script>
                """
                st.markdown(html_code, unsafe_allow_html=True)
                
                # Pop the message so it doesn't show again on completely unrelated app reruns
                del st.session_state.feedback_thanks_msg


# Initialize send_button to avoid NameError if skipped
send_button = False

# ---------------------------------------------------------
# INPUT HANDLING & CONFLICT SOLUTION logic (Run before widgets)
# ---------------------------------------------------------
should_run = False
user_input = ""

# Check for pending input from callbacks (Like quick suggestions)
if st.session_state.is_processing and "pending_input" in st.session_state:
    user_input = st.session_state.pending_input
    should_run = True
    feedback_placeholder.empty()

current_gender = st.session_state.get("gender_filter", [])
if current_gender is None: current_gender = []
if not isinstance(current_gender, list): current_gender = [current_gender]

last_gender = st.session_state.get("last_gender_filter", [])

if current_gender != last_gender:
    # Filter changed! Update tracker
    st.session_state.last_gender_filter = current_gender
    # If the user was typing something when they clicked a filter, restore it 
    # since st.form's submit block hasn't been hit yet (prevents auto-submitting)
    if "user_input_string" in st.session_state and st.session_state.user_input_string:
        # Save temp text but do NOT run execution
        pass


# Sticky input container with Quick Suggestions (Always render, styling changes based on state)
# Error Toast/Message (Persistent across reruns)
if "ui_error" in st.session_state and st.session_state.ui_error:
    st.error(st.session_state.ui_error)
    del st.session_state.ui_error

# --- SIDEBAR: Quick Suggestions & Filters ---
with st.sidebar:
    # Gender Filter (Multi-select Pills)
    st.markdown("### 🎯 Filter by Gender")
    selected_gender = st.pills(
        "Gender Filter",
        options=["Masculine", "Feminine", "Unisex"],
        default=None, # Default is None (All)
        selection_mode="multi",
        key="gender_filter",
        label_visibility="collapsed",
        disabled=st.session_state.is_processing
    )
    st.markdown("<div style='margin-bottom: 20px; border-bottom: 1px solid #eaeaea; padding-bottom: 10px;'></div>", unsafe_allow_html=True) # Divider

    st.markdown("### 💡 Inspiration Board")
    st.markdown("<p style='font-size: 0.85rem; color: #666;'>Pick a vibe to start your fragrance journey:</p>", unsafe_allow_html=True)
    
    CATEGORIZED_SUGGESTIONS = {
        "👥 Occasions": [
            "🍷 Romantic Date Night",
            "👔 Office & Professional",
            "🌴 Summer Beach Vacation",
            "💒 Wedding Day Elegance"
        ],
        "🌿 Scent Profiles": [
            "🍃 Fresh, Citrus & Aquatic",
            "☕ Coffee, Caramel & Vanilla",
            "🌲 Deep Woods & Spices",
            "🌹 White Florals & Rose",
            "🍒 Boozy Cherry & Almond"
        ],
        "🎭 Vibes & Esthetics": [
            "🔮 Dark & Mysterious",
            "☔ Cozy Rainy Day",
            "🦄 Unique & Niche",
            "🧁 Edible & Gourmand"
        ],
        "🧬 Recommend Similar": [
            "🧬 Like Baccarat Rouge but cheaper",
            "🧬 Like YSL Black Opium",
            "🧬 Like By Kilian Angels' Share",
            "🧬 Like Creed Aventus but cheaper"
        ],
        "🆚 Compare Classics": [
            "🆚 Bleu de Chanel vs Dior Sauvage",
            "🆚 Tom Ford Oud Wood vs Versace Oud Noir"
        ]
    }

    # Render as a sleek vertical list with category headers
    idx = 0
    for category, items in CATEGORIZED_SUGGESTIONS.items():
        st.markdown(f"<p style='font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; margin-top: 12px; color: #444;'>{category}</p>", unsafe_allow_html=True)
        for sugg in items:
            st.button(
                sugg,
                key=f"sugg_{idx}",
                use_container_width=True,
                on_click=set_suggestion,
                args=(sugg,),
                disabled=st.session_state.is_processing
            )
            idx += 1
# ----------------------------------

# Input area (wrapped in form for Enter key support)
# Init Security Manager
if "security_manager" not in st.session_state:
    st.session_state.security_manager = SecurityManager(max_chars=500, requests_per_min=15)

# Input area — conditional rendering to avoid the ghost/double-render bug.
# Root cause: changing `disabled` on the SAME widget key causes Streamlit to
# paint a second copy before removing the old one. Rendering two DIFFERENT
# forms (one per state) fixes this completely.
# Input forms wrapped in st.empty() to prevent double-render ghost on first query
input_form_slot = st.empty()
with input_form_slot.container():
    if st.session_state.is_processing:
        # --- PROCESSING STATE: show a visually identical but fully disabled form ---
        with st.form(key="chat_input_form_disabled", clear_on_submit=False, border=False):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_input(
                    "Ask something about perfumes...",
                    key="user_input_string_disabled",
                    placeholder="🔮 Consulting the fragrance database...",
                    label_visibility="visible",
                    disabled=True
                )
            with col2:
                st.form_submit_button("➤", use_container_width=True, disabled=True)
    else:
        # --- IDLE STATE: show the real interactive form ---
        with st.form(key="chat_input_form", clear_on_submit=True, border=False):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_input(
                    "Ask something about perfumes...",
                    key="user_input_string",
                    placeholder="e.g. 'Recommend a fresh summer scent' or 'Compare Sauvage and Eros'",
                    label_visibility="visible",
                    disabled=False
                )
            with col2:
                submitted = st.form_submit_button(
                    "➤",
                    use_container_width=True,
                    on_click=lock_ui
                )

# Close sticky input container (Removed HTML wrapper)

if should_run and user_input:
    try:
        # 1. Security & Moderation
        if not st.session_state.security_manager.check_rate_limit():
            st.session_state.ui_error = "⚠️ You are sending messages too fast. Please wait a moment."
            st.session_state.is_processing = False
            st.rerun()
        else:
            clean_input = st.session_state.security_manager.sanitize_input(user_input)
            
            if not st.session_state.security_manager.is_safe_input(clean_input):
                st.session_state.ui_error = "⛔ Security Alert: Restricted patterns detected."
                st.session_state.is_processing = False
                st.rerun()
            elif not st.session_state.security_manager.check_moderation(clean_input):
                st.session_state.ui_error = "⛔ Security Alert: Policy violation detected."
                st.session_state.is_processing = False
                st.rerun()
            else:
                # CRITICAL: Sync UI filter state with Agent's internal state for Tool Injection
                _ALL_GENDERS = {"Masculine", "Feminine", "Unisex"}
                if "gender_filter" in st.session_state and st.session_state.gender_filter:
                    if isinstance(st.session_state.gender_filter, list):
                        selected = st.session_state.gender_filter
                        # Selecting all 3 = effectively no filter → maps to prewarm's None entry
                        if set(selected) == _ALL_GENDERS:
                            st.session_state.agent.gender_filter = []
                        else:
                            st.session_state.agent.gender_filter = selected
                    else:
                        # Fallback: single string
                        st.session_state.agent.gender_filter = [st.session_state.gender_filter] if st.session_state.gender_filter != "All" else []
                else:
                    st.session_state.agent.gender_filter = []

            # Strip [CACHED_QUERY: ...] tag for display (user shouldn't see internal cache hints)
            # CACHED_QUERY can contain arrays like ['A', 'B'], so .*$ safely strips to the end
            import re
            display_input = re.sub(r'\s*\[CACHED_QUERY:.*$', '', clean_input).strip()
            
            # Store the clean display version (without cache tag) in chat history
            st.session_state.messages.append({"role": "user", "content": display_input})
            
            with chat_container:
                st.markdown(f'''
                <div class="user-msg-container">
                    <div class="user-msg">
                        {display_input}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with chat_container.chat_message("assistant", avatar="✨"):
                response_placeholder = st.empty()
                
                # 2. Initialize generator (send FULL input with cache tag to Agent)
                gen = st.session_state.agent.chat_stream(clean_input)
                
                # 3. Wait for first chunk with Spinner (This covers Tool Execution time)
                first_chunk = ""
                with st.spinner("🔮 Consults the fragrance database..."):
                    try:
                        # Grab first chunk manually to unblock spinner
                        first_chunk = next(gen)
                    except StopIteration:
                        # Generator empty (rare)
                        pass
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        response = None
                        first_chunk = None

                # 4. Stream the rest
                if first_chunk is not None:
                    try:
                        def stream_wrapper():
                            yield first_chunk
                            buffer = ""
                            for chunk in gen:
                                buffer += chunk
                                # Batching chunks to prevent React DOM overload on long chats.
                                # Streamlit can clog if it receives 50+ token messages per second.
                                if len(buffer) >= 4:
                                    # Micro-sleep allows the frontend to gracefully paint the frame
                                    time.sleep(0.005) 
                                    yield buffer
                                    buffer = ""
                            if buffer:
                                yield buffer
                                
                        response = response_placeholder.write_stream(stream_wrapper())
                    except Exception as e:
                        st.error(f"Error streaming response: {e}")
                        response = None

            if response:
                # Extract tool calls for debug
                tool_calls = extract_tool_calls(st.session_state.agent)
                if tool_calls:
                    st.session_state.tool_history.extend(tool_calls)
                
                # Store Results with tool metadata
                tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "tools": tool_names
                })
                # Save last query and response for feedback
                st.session_state.last_response = response
                
                # Upload the full agent_activity.log to HF in background (single file, no separate commits)
                upload_backend_logs()
                
    finally:
        # ALWAYS RESET STATE (Success or Error)
        st.session_state.is_processing = False
        if "pending_input" in st.session_state:
            del st.session_state.pending_input
        
        # Force rerun to unlock UI
        st.rerun()

# Close sticky input container
st.markdown('</div>', unsafe_allow_html=True)

# Debug: Show Tool History at the bottom (Removed for Production)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
_backend_label = "Self-Hosted vLLM (GPU)" if st.session_state.get("use_gpu", False) else "OpenAI GPT-4o-mini"
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Powered by {_backend_label} & ChromaDB • 2300+ Fragrances
    <br><br>
    <span style="font-size: 0.7rem; opacity: 0.5;">
        Educational research project. Data collected from public sources. No commercial use intended.<br>
        All rights belong to their respective owners. | Created by Alper • 2026
    </span>
</div>
""", unsafe_allow_html=True)
