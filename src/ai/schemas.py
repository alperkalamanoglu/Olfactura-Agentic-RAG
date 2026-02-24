"""
Pydantic schemas for tool input validation.
Ensures LLM-generated tool arguments are type-safe before execution.
Prevents malformed JSON from reaching tool functions (e.g., 'sort_by=null' corruption bug).
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional


class SearchPerfumesInput(BaseModel):
    """Input schema for search_perfumes tool"""
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields (catches malformed LLM output)
    
    query: Optional[str] = Field(
        default=None,
        description="Natural language description of desired scent (e.g., 'fresh citrus for summer'). "
                    "If only a brand is requested, omit this field."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters in MongoDB-style syntax. "
                    "Examples: {'brand': 'Dior'}, {'gender_score': {'$lt': 0.4}}, {'price_tier_score': {'$lt': 4.0}}"
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Sort logic. Use 'weighted_rating' for 'best/top rated'. "
                    "Use 'votes' for 'most popular'. Omit for default relevance sorting."
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of results to return. Default is 3."
    )
    excluded_notes: Optional[List[str]] = Field(
        default=None,
        description="List of note names to exclude (e.g., ['Rose', 'Jasmine'])."
    )


class GetPerfumeDetailsInput(BaseModel):
    """Input schema for get_perfume_details tool"""
    model_config = ConfigDict(extra="forbid")
    
    perfume_name: str = Field(
        description="Name of the perfume to retrieve details for (e.g., 'Sauvage', 'Black Opium')"
    )


class RecommendSimilarInput(BaseModel):
    """Input schema for recommend_similar tool"""
    model_config = ConfigDict(extra="forbid")
    
    reference_perfume_names: List[str] = Field(
        description="List of reference perfume names to blend or find similarities for."
    )
    additional_query: Optional[str] = Field(
        default=None,
        description="Optional textual modifiers (e.g., 'more woody', 'fresher version', 'less sweet')."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional constraints (e.g., {'price_tier_score': {'$lt': 5.0}} for cheaper alternatives)."
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of recommendations (default: 3)."
    )


class ComparePerfumesInput(BaseModel):
    """Input schema for compare_perfumes tool"""
    model_config = ConfigDict(extra="forbid")
    
    perfume_names: List[str] = Field(
        min_length=2,
        max_length=5,
        description="List of perfume names to compare (e.g., ['Sauvage', 'Eros', 'The One'])"
    )


# Map tool names to their Pydantic models
TOOL_SCHEMAS = {
    "search_perfumes": SearchPerfumesInput,
    "get_perfume_details": GetPerfumeDetailsInput,
    "recommend_similar": RecommendSimilarInput,
    "compare_perfumes": ComparePerfumesInput,
}
