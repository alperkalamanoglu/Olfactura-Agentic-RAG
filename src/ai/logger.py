"""
Logging utilities for tracking agent tool calls and parameters.
Structured JSON logging with no duplicates.
"""
import logging
from logging.handlers import RotatingFileHandler
import json
import os
from datetime import datetime

# Log directory setup
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "agent_activity_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "agent_activity.log")

def get_logger(name):
    """Get a configured logger with file handler and console handler"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs by clearing existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Rotating file handler (5MB limit, keep last 3 files)
    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=5*1024*1024, 
        backupCount=3, 
        encoding='utf-8'
    )
    
    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    
    # Clean, readable format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create specific loggers
agent_logger = get_logger("AGENT")
tool_logger = get_logger("TOOL")

def log_event(logger, event_type: str, message: str, data: dict = None):
    """
    Log a structured event.
    Format: [EVENT_TYPE] Message | {json_data}
    """
    log_msg = f"[{event_type}] {message}"
    if data:
        try:
            # Serialize data to compact JSON
            json_str = json.dumps(data, default=str, ensure_ascii=False)
            log_msg += f" | {json_str}"
        except Exception as e:
            log_msg += f" | (serialization error: {e})"
            
    logger.info(log_msg)

def log_tool_call(tool_name: str, args: dict, result_preview: str = None):
    """Helper specifically for tool execution logging"""
    data = {"args": args}
    if result_preview:
        data["result"] = result_preview[:200] + "..." if len(result_preview) > 200 else result_preview
        
    log_event(tool_logger, "EXEC", f"Calling {tool_name}", data)
