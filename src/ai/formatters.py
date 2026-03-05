"""
Helper functions for formatting tool outputs in human-readable format.
"""

def format_price_tier(score: float) -> str:
    """Convert price tier score (0-10) to human-readable label"""
    if score < 2.5:
        return "Budget ($)"
    elif score < 4.0:
        return "Affordable ($$)"
    elif score < 6.0:
        return "Mid-Range ($$$)"
    elif score < 8.0:
        return "Premium ($$$$)"
    else:
        return "Luxury ($$$$$)"

def format_gender(score: float) -> str:
    """Convert gender score (0-1) to human-readable label"""
    if score < 0.4:  # Sync with filter: Feminine
        return "Feminine"
    elif score > 0.6:  # Sync with filter: Masculine
        return "Masculine"
    else:  # 0.4 <= score <= 0.6: Unisex
        return "Unisex"

def format_longevity(score: float) -> str:
    """Convert longevity score (0-10) to human-readable label"""
    if score < 2.5:
        return "Very Weak"
    elif score < 5.0:
        return "Weak"
    elif score < 7.5:
        return "Moderate"
    elif score < 9.0:
        return "Long Lasting"
    else:
        return "Eternal"

def format_sillage(score: float) -> str:
    """Convert sillage score (0-10) to human-readable label"""
    if score < 2.5:
        return "Intimate"
    elif score < 5.0:
        return "Moderate"
    elif score < 7.5:
        return "Strong"
    else:
        return "Enormous"
