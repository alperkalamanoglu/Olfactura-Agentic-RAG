"""
Agent class for orchestrating tool calls with dual-backend support
(Self-hosted vLLM on GPU as primary, OpenAI GPT-4o-mini as fallback)
"""
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
from src.ai.tools import search_perfumes, get_perfume_details, recommend_similar, compare_perfumes
from src.ai.schemas import TOOL_SCHEMAS, SearchPerfumesInput, GetPerfumeDetailsInput, RecommendSimilarInput, ComparePerfumesInput
from src.ai.logger import log_event, agent_logger
from src.ai.prompts import SYSTEM_PROMPT_TEMPLATE
from datetime import datetime
import time

load_dotenv()

# Auto-generate OpenAI Tool Definitions from Pydantic schemas
def _pydantic_to_openai_tool(name: str, description: str, schema_cls) -> dict:
    """Convert a Pydantic model to OpenAI function calling format."""
    json_schema = schema_cls.model_json_schema()
    
    # OpenAI expects 'parameters' with 'properties' and 'required'
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])
    
    # Clean up Pydantic-specific keys that OpenAI doesn't understand
    clean_properties = {}
    for prop_name, prop_schema in properties.items():
        clean_prop = {k: v for k, v in prop_schema.items() if k != "default"}
        # Handle anyOf (Pydantic's way of saying Optional)
        if "anyOf" in clean_prop:
            types = [t for t in clean_prop["anyOf"] if t.get("type") != "null"]
            if types:
                clean_prop.update(types[0])
            del clean_prop["anyOf"]
        clean_properties[prop_name] = clean_prop
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": clean_properties,
                "required": required
            }
        }
    }

TOOL_DEFINITIONS = [
    _pydantic_to_openai_tool(
        "search_perfumes",
        "Search for perfumes using semantic query and/or metadata filters. Use this when user asks for recommendations based on descriptions, occasions, or specific attributes.",
        SearchPerfumesInput
    ),
    _pydantic_to_openai_tool(
        "get_perfume_details",
        "Get comprehensive information about a specific perfume by name. Use when user asks about a particular perfume's composition, notes, or characteristics.",
        GetPerfumeDetailsInput
    ),
    _pydantic_to_openai_tool(
        "recommend_similar",
        "Find perfumes similar to a reference perfume based on scent profile. Use when user says 'similar to X', 'like X but...', or asks for alternatives.",
        RecommendSimilarInput
    ),
    _pydantic_to_openai_tool(
        "compare_perfumes",
        "Compare multiple perfumes side-by-side showing their ratings, longevity, sillage, and gender scores. Use when user asks to compare or wants to see differences between options.",
        ComparePerfumesInput
    ),
]

# Map function names to actual Python functions
TOOL_MAP = {
    "search_perfumes": search_perfumes,
    "get_perfume_details": get_perfume_details,
    "recommend_similar": recommend_similar,
    "compare_perfumes": compare_perfumes
}

class PerfumeAgent:
    """AI Agent for perfume recommendations with GPU/Cloud dual-backend support"""
    
    # Context window limits mapped to hardware backend mode
    GPU_MAX_CHARS = 12000
    GPU_MAX_MESSAGES = 6
    CLOUD_MAX_CHARS = 60000
    CLOUD_MAX_MESSAGES = 20

    def __init__(self, gender_filter: list = None, use_gpu: bool = False):
        """
        Args:
            gender_filter: List of gender selections from UI
            use_gpu: Whether the GPU backend (vLLM) is available (determined at session start)
        """
        self.use_gpu = use_gpu
        self.gender_filter = gender_filter if gender_filter is not None else []
        self.conversation_history: List[Dict] = []
        
        if use_gpu:
            # PRIMARY: Vast.ai / RunPod vLLM server
            self.client = OpenAI(
                base_url=os.getenv("VLLM_BASE_URL"),
                api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                timeout=15.0
            )
            self.model = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")
            log_event(agent_logger, "INFO", f"Agent initialized with GPU backend: {self.model}")
        else:
            # FALLBACK: OpenAI Cloud API
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=30.0
            )
            self.model = "gpt-4o-mini"
            log_event(agent_logger, "INFO", "Agent initialized with OpenAI fallback: gpt-4o-mini")
        
        # Load external system prompt and format dynamic years
        current_year = datetime.now().year
        last_year = current_year - 1
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_year=current_year, last_year=last_year)
        
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })
    
    def _trim_history(self, max_messages: int = None, max_chars: int = None):
        """
        Keeps history lean for high concurrency (2x+ on vLLM).
        """
        if not self.use_gpu:
            # OpenAI Fallback: GPT-4o-mini has 128k context and doesn't eat local VRAM.
            # Relax limits completely to retain deep conversation memory.
            max_messages = max_messages or PerfumeAgent.CLOUD_MAX_MESSAGES
            max_chars = max_chars or PerfumeAgent.CLOUD_MAX_CHARS
        else:
            # GPU Mode: Strict caps to stay in 'Sweet Spot' concurrency
            max_messages = max_messages or PerfumeAgent.GPU_MAX_MESSAGES
            max_chars = max_chars or PerfumeAgent.GPU_MAX_CHARS
        if len(self.conversation_history) <= 1:
            return

        # 1. Message Count Truncation (Preserve System at 0)
        while len(self.conversation_history) > max_messages:
            self.conversation_history.pop(1)
            # OpenAI requires 'tool' messages to have a preceding 'assistant' tool_call. 
            # If we popped the assistant, we MUST pop its orphaned tool outputs.
            while len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                self.conversation_history.pop(1)

        # 2. Token/Char-based Truncation (Heuristic)
        while len(self.conversation_history) > 2: # Keep at least System + Last User + Last AI
            total_chars = sum(len(str(m.get('content', ''))) for m in self.conversation_history[1:])
            if total_chars > max_chars:
                self.conversation_history.pop(1)
                # Ensure no orphaned tool sequences are left
                while len(self.conversation_history) > 1 and self.conversation_history[1].get("role") == "tool":
                    self.conversation_history.pop(1)
            else:
                break

    def _is_qwen(self) -> bool:
        """Check if current model is Qwen (needs special parameters)."""
        return "qwen" in self.model.lower()
    
    def _build_kwargs(self, streaming: bool = False, include_tools: bool = False) -> dict:
        """Build the kwargs dict for OpenAI API calls with model-specific parameters."""
        self._trim_history() # Keep history lean to fit in context window
        kwargs = {
            "model": self.model,
            "messages": self.conversation_history,
            "temperature": 0.0 if not streaming else 0.1,
            "max_tokens": 2048,
        }
        
        if streaming:
            kwargs["stream"] = True
        
        # Disable thinking mode for ALL self-hosted vLLM models (Qwen, Gemma, etc.)
        # OpenAI API ignores unknown params, so this only applies when use_gpu=True
        if self.use_gpu:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        if include_tools:
            kwargs["tools"] = TOOL_DEFINITIONS
            kwargs["tool_choice"] = "auto"
            if streaming:
                kwargs["parallel_tool_calls"] = False
        
        return kwargs
    
    def _switch_to_openai(self):
        """Runtime fallback: switch from GPU backend to OpenAI mid-session."""
        self.use_gpu = False
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0
        )
        self.model = "gpt-4o-mini"
        log_event(agent_logger, "WARNING", "Switched to OpenAI fallback for remainder of session.")
    
    @staticmethod
    def _clean_thinking(text: str) -> str:
        """
        Strip thinking-mode artifacts from model output.
        Some vLLM models ignore enable_thinking:False and still leak
        'thought' prefixes or <think>...</think> blocks into responses.
        """
        import re
        # Remove <think>...</think> blocks (Gemma/Qwen thinking tags)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        # Remove bare 'thought' prefix at the very start (Gemma 4 artifact)
        text = re.sub(r'^thought\s+', '', text, flags=re.IGNORECASE)
        return text
    
    def _trim_context(self, max_messages: int = 6):
        """Keep only last N messages plus system prompt. Ensures no orphan tool calls."""
        if len(self.conversation_history) <= max_messages + 1:
            return

        system_msg = self.conversation_history[0]
        # Initial slice
        recent = self.conversation_history[-max_messages:]
        
        # Safe Trim: Ensure we don't start with a 'tool' message (orphaned result)
        # or an assistant message that is ONLY a tool call (orphaned call without result context potentially)
        # Best strategy: Start from a 'user' message if possible
        while len(recent) > 0:
            first_msg = recent[0]
            role = first_msg.get("role") if isinstance(first_msg, dict) else first_msg.role
            
            # If it's a tool output, remove it (it lost its parent call)
            if role == "tool":
                recent.pop(0)
                continue
                
            # If it's an assistant message with tool_calls, but we aren't sure if we have the tools...
            # Actually, simply ensuring we start with 'user' is the safest bet for RAG chats.
            if role == "user":
                break
                
            # If it's assistant text, it's fine.
            # But if we are in the middle of a chain, better simplify.
            recent.pop(0)
            
        self.conversation_history = [system_msg] + recent
    
    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """
        Process a user message and return agent's response.
        Handles tool calling automatically.
        """
        # Trim context for token efficiency (Max 6 keeps enough conversation context without bloat)
        self._trim_context(max_messages=6)
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            
            # Build API call parameters (handles Qwen extra_body automatically)
            kwargs = self._build_kwargs(
                streaming=False,
                include_tools=(iterations == 1)
            )

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self.use_gpu:
                    # GPU failed mid-session → fall back to OpenAI
                    log_event(agent_logger, "WARNING", f"GPU backend failed: {e}. Falling back to OpenAI.")
                    self._switch_to_openai()
                    kwargs = self._build_kwargs(streaming=False, include_tools=(iterations == 1))
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            
            assistant_message = response.choices[0].message
            
            # If no tool calls, this is the final final answer!
            if not assistant_message.tool_calls:
                final_content = assistant_message.content
                if self.use_gpu and final_content:
                    final_content = self._clean_thinking(final_content)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })
                return final_content
            
            # 1. Add assistant's tool calls to history
            self.conversation_history.append(assistant_message)
            
            # 2. Process EVERY tool call in the message
            # This is crucial: OpenAI requires a response for every tool_call_id
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Inject user-selected gender filters
                if self.gender_filter and len(self.gender_filter) > 0 and function_name in ["search_perfumes", "recommend_similar"]:
                    if "filters" not in function_args or function_args["filters"] is None:
                        function_args["filters"] = {}
                    if "gender_score" in function_args["filters"]:
                        del function_args["filters"]["gender_score"]
                    
                    gender_map = {
                        "Masculine": {"gender_score": {"$gt": 0.6}},
                        "Feminine": {"gender_score": {"$lt": 0.4}},
                        "Unisex": {"$and": [
                            {"gender_score": {"$gte": 0.4}},
                            {"gender_score": {"$lte": 0.6}}
                        ]}
                    }
                    if len(self.gender_filter) == 1:
                        if self.gender_filter[0] in gender_map:
                            function_args["filters"].update(gender_map[self.gender_filter[0]])
                    elif len(self.gender_filter) > 1:
                        or_conditions = [gender_map[g] for g in self.gender_filter if g in gender_map]
                        if or_conditions:
                            function_args["filters"]["$or"] = or_conditions
                    log_event(agent_logger, "FILTER_INJECT", f"Auto-injected gender filter: {self.gender_filter}")

                # Execute tool with Pydantic validation
                function_to_call = TOOL_MAP[function_name]
                
                # Validate arguments through Pydantic schema
                schema_cls = TOOL_SCHEMAS.get(function_name)
                if schema_cls:
                    try:
                        validated = schema_cls(**function_args)
                        function_args = validated.model_dump(exclude_none=True)
                    except ValidationError as ve:
                        log_event(agent_logger, "VALIDATION_ERROR", f"Tool {function_name} args invalid: {ve}")
                        function_response = f"Error: Invalid arguments for {function_name}. Details: {ve}"
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(function_response)
                        })
                        continue
                
                function_response = function_to_call(**function_args)
                
                # Add response to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(function_response)
                })
        
        return "I've searched through several options but reached my limit. Based on what I found, what else can I help you with?"
    
    def reset_conversation(self):
        """Clear conversation history (keep system prompt)"""
        self.conversation_history = [self.conversation_history[0]]

    def chat_stream(self, user_message: str, max_iterations: int = 3):
        """
        Generator for streaming response.
        Tool calls are handled internally (synchronously), then the final response is streamed.
        
        Yields:
            str: Chunks of the final response content
        """
        start_time = time.time()  # Start global timer
        
        # Trim context for token efficiency
        self._trim_context(max_messages=6)
        
        log_event(agent_logger, "USER_QUERY", user_message[:500])
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            iter_start_time = time.time()
            log_event(agent_logger, "THOUGHT_PROCESS", f"Iteration {iterations}/{max_iterations}")
            
            try:
                # Build API call parameters (handles Qwen extra_body, tools, streaming automatically)
                kwargs = self._build_kwargs(
                    streaming=True,
                    include_tools=(iterations == 1)
                )
                stream = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if self.use_gpu:
                    # GPU failed mid-session → fall back to OpenAI and retry
                    log_event(agent_logger, "WARNING", f"GPU streaming failed: {e}. Falling back to OpenAI.")
                    self._switch_to_openai()
                    try:
                        kwargs = self._build_kwargs(streaming=True, include_tools=(iterations == 1))
                        stream = self.client.chat.completions.create(**kwargs)
                    except Exception as e2:
                        log_event(agent_logger, "ERROR", f"OpenAI fallback also failed: {str(e2)}")
                        yield f"⚠️ Sorry, I encountered a technical issue. Please try again in a moment."
                        return
                else:
                    log_event(agent_logger, "ERROR", f"API call failed: {str(e)}")
                    yield f"⚠️ Sorry, I encountered a technical issue. Please try rephrasing your question or try again in a moment."
                    return
            
            # Collect streaming response
            collected_content = ""
            collected_tool_calls = []
            current_tool_call = None
            first_token_yielded = False
            
            # Anti-thinking buffer for streaming
            initial_buffer = ""
            thinking_stripped = False
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle content chunks - YIELD IMMEDIATELY (with anti-thought buffer)
                if delta.content:
                    if not first_token_yielded:
                        llm_ttft = time.time() - iter_start_time
                        global_ttft = time.time() - start_time
                        log_event(agent_logger, "PERFORMANCE", "LLM Text Gen TTFT", {"seconds": round(llm_ttft, 3)})
                        log_event(agent_logger, "PERFORMANCE", "Global UX TTFT (Click to First Token)", {"seconds": round(global_ttft, 3)})
                        first_token_yielded = True
                    
                    if not thinking_stripped and self.use_gpu:
                        initial_buffer += delta.content
                        if len(initial_buffer) < 12:
                            # Start checking if it begins with "thought" (ignoring whitespace)
                            if not "thought".startswith(initial_buffer.lstrip().lower()[:7]):
                                # Not matching "thought", release buffer
                                yield initial_buffer
                                collected_content += initial_buffer
                                thinking_stripped = True
                            continue
                        else:
                            # We have enough chars to safely strip it once
                            import re
                            cleaned = re.sub(r'^\s*thought\s+', '', initial_buffer, flags=re.IGNORECASE)
                            yield cleaned
                            collected_content += cleaned
                            thinking_stripped = True
                            continue
                    
                    collected_content += delta.content
                    yield delta.content  # Stream to user immediately!
                
                # Handle tool call chunks (accumulate, don't yield)
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        if tc_chunk.index is not None:
                            # New tool call or continuation?
                            while len(collected_tool_calls) <= tc_chunk.index:
                                collected_tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            current_tc = collected_tool_calls[tc_chunk.index]
                            
                            if tc_chunk.id:
                                current_tc["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    current_tc["function"]["name"] += tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    current_tc["function"]["arguments"] += tc_chunk.function.arguments
            
            # Flush any unyielded buffer (if the entire response was < 12 chars)
            if self.use_gpu and not thinking_stripped and initial_buffer:
                import re
                cleaned = re.sub(r'^\s*thought\s+', '', initial_buffer, flags=re.IGNORECASE)
                yield cleaned
                collected_content += cleaned

            # Stream finished - analyze what we got
            has_tool_calls = len(collected_tool_calls) > 0 and collected_tool_calls[0]["id"] != ""
            
            # If NO tool calls, we already yielded the content - just finalize
            if not has_tool_calls:
                log_event(agent_logger, "DECISION", "Final answer ready (no tools called)")
                
                # Append to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": collected_content
                })
                
                # Log Total Performance
                total_time = time.time() - start_time
                log_event(agent_logger, "RESPONSE_COMPLETE", "End-to-End Task Complete", {
                    "total_time_sec": round(total_time, 2),
                    "length": len(collected_content)
                })
                return
            else:
                tool_llm_time = time.time() - iter_start_time
                log_event(agent_logger, "PERFORMANCE", "LLM Tool-Calling JSON Gen Time", {"seconds": round(tool_llm_time, 3)})
            
            # If there ARE tool calls, execute them (not streamed)
            # Use collected_tool_calls from streaming
            assistant_msg_dict = {
                "role": "assistant",
                "content": collected_content,
                "tool_calls": collected_tool_calls
            }
            self.conversation_history.append(assistant_msg_dict)
            
            for tool_call in collected_tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                # Inject gender filter if user selected one or more
                if self.gender_filter and len(self.gender_filter) > 0 and tool_name in ["search_perfumes", "recommend_similar"]:
                    if "filters" not in tool_args or tool_args["filters"] is None:
                        tool_args["filters"] = {}
                    # CRITICAL: Remove any existing gender_score filter that LLM inferred
                    # We want the UI filter to be the strict source of truth
                    if "gender_score" in tool_args["filters"]:
                        del tool_args["filters"]["gender_score"]
                    
                    # Map UI selection to database filter
                    gender_map = {
                        "Masculine": {"gender_score": {"$gt": 0.6}},
                        "Feminine": {"gender_score": {"$lt": 0.4}},
                        "Unisex": {"$and": [  # Unisex needs $and wrapper for range
                            {"gender_score": {"$gte": 0.4}},
                            {"gender_score": {"$lte": 0.6}}
                        ]}
                    }
                    
                    # If multiple genders selected, use $or logic
                    if len(self.gender_filter) == 1:
                        # Single selection - direct filter
                        if self.gender_filter[0] in gender_map:
                            tool_args["filters"].update(gender_map[self.gender_filter[0]])
                    elif len(self.gender_filter) > 1:
                        # Multiple selections - OR logic
                        or_conditions = []
                        for gender in self.gender_filter:
                            if gender in gender_map:
                                or_conditions.append(gender_map[gender])
                        if or_conditions:
                            tool_args["filters"]["$or"] = or_conditions
                    
                    log_event(agent_logger, "FILTER_INJECT", f"Auto-injected gender filter: {self.gender_filter}")
                
                log_event(agent_logger, "TOOL_CALL", tool_name, tool_args)
                
                # Execute tool with Safe Handling
                try:
                    t_tool_start = time.time()
                    
                    # Validate arguments through Pydantic schema
                    schema_cls = TOOL_SCHEMAS.get(tool_name)
                    if schema_cls:
                        try:
                            validated = schema_cls(**tool_args)
                            tool_args = validated.model_dump(exclude_none=True)
                        except ValidationError as ve:
                            log_event(agent_logger, "VALIDATION_ERROR", f"Tool {tool_name} args invalid: {ve}")
                            result = f"Error: Invalid arguments for {tool_name}. Please fix: {ve}"
                            t_tool_end = time.time()
                            log_event(agent_logger, "PERFORMANCE", "Tool Execution Time", {"seconds": round(t_tool_end - t_tool_start, 3), "tool": tool_name})
                            # Add error to history so LLM can self-correct
                            self.conversation_history.append({
                                "tool_call_id": tool_call_id,
                                "role": "tool",
                                "name": tool_name,
                                "content": str(result)
                            })
                            continue
                    
                    if tool_name == "search_perfumes":
                        result = search_perfumes(**tool_args)
                    elif tool_name == "get_perfume_details":
                        result = get_perfume_details(**tool_args)
                    elif tool_name == "compare_perfumes":
                        result = compare_perfumes(**tool_args)
                    elif tool_name == "recommend_similar":
                        result = recommend_similar(**tool_args)
                    else:
                        result = f"Error: Unknown tool {tool_name}"
                    t_tool_end = time.time()
                    log_event(agent_logger, "PERFORMANCE", "Tool Execution Time", {"seconds": round(t_tool_end - t_tool_start, 3), "tool": tool_name})
                except Exception as e:
                    log_event(agent_logger, "TOOL_ERROR", f"Tool {tool_name} failed: {e}")
                    result = f"Error executing tool {tool_name}: {e}"
                
                # Log result (Full result for debugging AI Match scores)
                result_str = str(result)
                log_event(agent_logger, "TOOL_RESULT", f"Result for {tool_name}", {"result": result_str})
                
                # Add tool result to history
                self.conversation_history.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str
                })
        
        # Fallback if max iterations reached
        log_event(agent_logger, "WARNING", "Max iterations reached")
        fallback_msg = "I've searched extensively but couldn't finalize the answer. Let me know if you'd like to narrow down the criteria."
        yield fallback_msg
        self.conversation_history.append({
            "role": "assistant",
            "content": fallback_msg
        })
