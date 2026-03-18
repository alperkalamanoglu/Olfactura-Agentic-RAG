import pytest
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.vector_db import VectorDatabase
from src.ai.tools import set_global_db

# Try to mock the API keys if missing, so tests pass locally without full .env
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy-key-for-testing")

@pytest.fixture(scope="session")
def real_db():
    """
    Returns the actual initialized VectorDatabase for integration tests.
    Uses the real ChromaDB collection.
    """
    # We initialize it once per test session to save time
    db = VectorDatabase()
    
    # Inject it into tools module so tests hitting `search_perfumes` use this instance
    set_global_db(db)
    
    return db

@pytest.fixture
def sample_perfume_data():
    """Returns a dummy perfume data dictionary for unit testing."""
    return {
        "id": "test_perfume_1",
        "name": "Test Aqua",
        "brand": "Test Brand",
        "gender": "Unisex",
        "gender_score": 0.5,
        "price_tier_score": 3.0,
        "rating": 4.5,
        "notes_str": "Citrus, Sea Water",
        "accords_str": "Marine, Fresh",
        "semantic_text": "A fresh marine scent for summer days.",
        "season_summer": 1.0,
        "season_winter": 0.0
    }
