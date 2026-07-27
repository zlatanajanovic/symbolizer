# Below is the full updated code in one file, with added code and comments where changes were made.
# Changes are marked with "### MODIFICATION FOR GOALS START" and "### MODIFICATION FOR GOALS END"

# Imports
import os
import json
import random
import base64
import hashlib
import sys
from pathlib import Path
import requests
import time
from typing import List, Dict, Any, Literal, Optional
import threading
from types import SimpleNamespace
from dotenv import load_dotenv
from PIL import Image
import tiktoken
from pydantic import ValidationError, BaseModel
from itertools import permutations
from functools import wraps
from collections import defaultdict
import statistics

# Ensure that utils modules are accessible
from .parsing_utils.general_utils import load_env, next_letter, load_config
from .parsing_utils.generate_scenes_utils import get_atom_schema_from_chatgpt_objects
from .parsing_utils.chatgpt_utils import (
    PDDLDomainWrapper,
    get_object_schema_from_pddl,
    create_dynamic_models,
    encode_image_base64,
)
from .parsing_utils.chatgpt_utils import parse_domain
# OpenAI Imports
from openai import RateLimitError,LengthFinishReasonError,OpenAI
from mistralai import Mistral

# Gemini imports (optional - will be None if not installed)
try:
    from .gemini_client import GeminiClient, is_gemini_available, create_gemini_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiClient = None
    is_gemini_available = lambda: False
    create_gemini_client = lambda: None

# Load configuration from an absolute path so imports do not depend on cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = Path(os.getenv("SYMBOLIZER_CONFIG_PATH", str(_REPO_ROOT / "config.yaml"))).resolve()
config = load_config(str(_CONFIG_PATH))

# Global seed used for reproducible per-problem few-shot example selection.
# Override by passing --seed at the CLI; set directly when using as a library.
_GLOBAL_SEED = 42  # type: int

# NOTE: Module-level env loading uses the default config/.env path.
# This is overridden when run_eval_on_dataset.py calls reinitialize_client()
# with the correct per-model config.
_ENV_FILE = Path(config["env_file"])
if not _ENV_FILE.is_absolute():
    _ENV_FILE = (_CONFIG_PATH.parent / _ENV_FILE).resolve()
load_env(str(_ENV_FILE))

# OpenAI API settings — these are DEFAULTS, will be overwritten by reinitialize_client()
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_TO_USE = os.getenv("MODEL_TO_USE", "None Set")
SELF_HOSTED_ENDPOINT = os.getenv("SELF_HOSTED_ENDPOINT", "None Set")
USE_LOGPROBS = os.getenv("USE_LOGPROBS", "false").lower() == "true"
DEFAULT_TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5" if USE_LOGPROBS else "0"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")
CODE_INTERPRETER_ENABLED = os.getenv("CODE_INTERPRETER_ENABLED", "false").lower() == "true"
LOG_EXAMPLES = os.getenv("SSR_LOG_EXAMPLES", "false").lower() == "true"
INCLUDE_INSTRUCTION_IN_EXAMPLES = (
    os.getenv("SSR_INCLUDE_INSTRUCTION_IN_EXAMPLES", "false").lower() == "true"
)
EXAMPLE_TEMPLATE_VARIANTS = max(1, int(os.getenv("SSR_EXAMPLE_TEMPLATE_VARIANTS", "1") or 1))
EXAMPLE_TEMPLATE_MODE = os.getenv("SSR_EXAMPLE_TEMPLATE_MODE", "random").strip().lower()
EXAMPLE_TEMPLATE_FIXED = os.getenv("SSR_EXAMPLE_TEMPLATE_FIXED", "").strip()

# --- OpenAI-compatible endpoint support (vLLM / proxies / other providers) ---
def _openai_base_url():
    """Custom base_url for any OpenAI-compatible server (vLLM, LiteLLM, OpenRouter, ...).
    Set OPENAI_BASE_URL (or OPENAI_API_BASE) to route the `openai` provider elsewhere."""
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None


def _make_openai_client(api_key, base_url=None):
    """Build an OpenAI SDK client, optionally pointed at a custom base_url.
    vLLM and most OpenAI-compatible servers ignore the key, so a placeholder is fine."""
    kwargs = {"api_key": api_key or "EMPTY"}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


# --- Client initialization (auto-init with override support) ---
# Auto-initialize client from module-level env vars (.env loaded above).
# reinitialize_client() can be called later to switch models (Phase 1 config switching).
# This supports both Phase 1 (explicit reinit per config) and Phase 2/3 (default model).
_MODULE_INIT_MODEL = MODEL_TO_USE
client = None

def _auto_init_client():
    """Auto-initialize client from current env vars if possible."""
    global client
    try:
        if MODEL_TO_USE == "mistral" and MISTRAL_API_KEY:
            client = Mistral(api_key=MISTRAL_API_KEY)
        elif MODEL_TO_USE == "openai" and (OPENAI_API_KEY or _openai_base_url()):
            client = _make_openai_client(OPENAI_API_KEY, _openai_base_url())
        elif MODEL_TO_USE == "selfhosted" and SELF_HOSTED_ENDPOINT != "None Set":
            client = _make_openai_client(OPENAI_API_KEY, SELF_HOSTED_ENDPOINT)
        elif MODEL_TO_USE == "gemini" and GEMINI_AVAILABLE and (GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY") or (os.getenv("PROJECT") and os.getenv("LOCATION"))):
            client = create_gemini_client()
        if client is not None:
            print(f"[run_eval_vlm] Auto-initialized {MODEL_TO_USE} client.")
        else:
            print(f"[run_eval_vlm] Module loaded, MODEL_TO_USE={MODEL_TO_USE}, client=None (will need reinitialize_client()).")
    except Exception as e:
        print(f"[run_eval_vlm] Auto-init failed: {e}. Call reinitialize_client() explicitly.")
        client = None

_auto_init_client()


def reinitialize_client():
    """
    Reinitialize the API client after environment variables have been loaded.
    Call this function after loading a new config file with different env settings.
    
    This is the ONLY safe way to initialize the VLM client. Module-level
    initialization is deferred to prevent model/config mismatch.
    """
    global client, MODEL_TO_USE, MODEL_NAME, USE_LOGPROBS, DEFAULT_TOP_LOGPROBS
    global INCLUDE_INSTRUCTION_IN_EXAMPLES, EXAMPLE_TEMPLATE_VARIANTS, EXAMPLE_TEMPLATE_MODE, EXAMPLE_TEMPLATE_FIXED
    global OPENAI_API_KEY, MISTRAL_API_KEY, SELF_HOSTED_ENDPOINT
    
    # Re-read environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MODEL_TO_USE = os.getenv("MODEL_TO_USE", "None Set")
    SELF_HOSTED_ENDPOINT = os.getenv("SELF_HOSTED_ENDPOINT", "None Set")
    USE_LOGPROBS = os.getenv("USE_LOGPROBS", "false").lower() == "true"
    DEFAULT_TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5" if USE_LOGPROBS else "0"))
    INCLUDE_INSTRUCTION_IN_EXAMPLES = (
        os.getenv("SSR_INCLUDE_INSTRUCTION_IN_EXAMPLES", "false").lower() == "true"
    )
    EXAMPLE_TEMPLATE_VARIANTS = max(1, int(os.getenv("SSR_EXAMPLE_TEMPLATE_VARIANTS", "1") or 1))
    EXAMPLE_TEMPLATE_MODE = os.getenv("SSR_EXAMPLE_TEMPLATE_MODE", "random").strip().lower()
    EXAMPLE_TEMPLATE_FIXED = os.getenv("SSR_EXAMPLE_TEMPLATE_FIXED", "").strip()
    
    # Re-read Gemini API key (check all three possible env var names)
    global GOOGLE_API_KEY
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    
    print(f"[reinitialize_client] MODEL_TO_USE={MODEL_TO_USE}, MODEL_NAME={MODEL_NAME}, USE_LOGPROBS={USE_LOGPROBS}")
    
    if MODEL_TO_USE == "openai":
        _base = _openai_base_url()
        client = _make_openai_client(OPENAI_API_KEY, _base)
        if _base:
            print(f"Using OpenAI-compatible endpoint: {_base} (model={MODEL_NAME})")
    elif MODEL_TO_USE == "mistral":
        client = Mistral(api_key=MISTRAL_API_KEY)
    elif MODEL_TO_USE == "selfhosted":
        client = _make_openai_client(OPENAI_API_KEY, SELF_HOSTED_ENDPOINT)
        print(f"Using self-hosted endpoint: {SELF_HOSTED_ENDPOINT}")
    elif MODEL_TO_USE == "gemini":
        _has_gemini_creds = GOOGLE_API_KEY or (os.getenv("PROJECT") and os.getenv("LOCATION"))
        if GEMINI_AVAILABLE and _has_gemini_creds:
            client = create_gemini_client()
            print(f"Using Gemini model: {MODEL_NAME}")
        else:
            raise ValueError(f"Gemini not available. GEMINI_AVAILABLE={GEMINI_AVAILABLE}, API key set={bool(GOOGLE_API_KEY)}, Vertex SA={bool(os.getenv('PROJECT'))}. Install google-genai: pip install google-genai")
    else:
        raise ValueError(f"Invalid model type: {MODEL_TO_USE}")
    
    return client

# Configuration Parameters
MAX_TOKENS = config['max_tokens']
DETAIL_LEVEL_LOW = config['detail_level_low']

def is_gpt5_model(model_name: str) -> bool:
    """Check if the model is a GPT-5 model which requires max_completion_tokens."""
    return model_name.startswith("gpt-5") or "gpt-5" in model_name.lower()

def get_openai_params() -> dict:
    """Get the correct parameters for OpenAI API based on model.
    GPT-5+ models require 'max_completion_tokens' (not 'max_tokens'),
    and only support temperature=1 (so we omit it).
    """
    if is_gpt5_model(MODEL_NAME):
        return {"max_completion_tokens": MAX_TOKENS}
    else:
        # Non-GPT-5 OpenAI chat models cap completion tokens well below our
        # config default (e.g. gpt-4o/4o-mini = 16384), so a 64000 request
        # 400s. Clamp to that ceiling -- grounding outputs are a few hundred
        # tokens, so this never truncates. Other providers (Gemini, used for
        # the paper, supports 64000) keep the full configured MAX_TOKENS.
        return {"max_tokens": min(MAX_TOKENS, 16384), "temperature": 0.2}

DATASET_PATH = config['dataset_path']
NUM_EXAMPLES = config['num_examples']
NUM_SAMPLES = config['num_tests']
DOMAIN_NAME = config['domain_name']
OUTPUT_FOLDER = config['output_folder']
VERBOSE = config['verbose']

FULL_RESPONSES_OBJECTS_FILE = config["objects_full_responses_path"]
FULL_RESPONSES_ATOMS_FILE = config["atoms_full_responses_path"]
FULL_RESPONSES_GOALS_FILE = config["goals_full_responses_path"]
# Global variables
object_schema = None
types_available_classes_mapping = {}
predicates_with_types = {}

# Rate-Limiting (Adjust according to rate limits)
#TODO: Remove or lower 
api_key_last_called = defaultdict(lambda: 0.0)
api_key_lock = threading.Lock()

# Function to create messages with image for OpenAI
def make_messages_image(prompt: str, image_base64: str) -> List[Dict[str, Any]]:
    """Creates a message containing a prompt and an image for OpenAI."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]

# Function to create text messages for OpenAI
def make_message_text(prompt: str) -> List[Dict[str, Any]]:
    """Creates a text message for OpenAI."""
    return [{"role": "user", "content": prompt}]

# Function to create assistant's text answer
def make_answer_text(answer: Any) -> Dict[str, Any]:
    """Creates an assistant's text answer."""
    return {"role": "assistant", "content": str(answer)}


def _extract_message_text(content: Any) -> str:
    """Normalize completion message content to a json string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            else:
                maybe_text = getattr(part, "text", None)
                if maybe_text:
                    text_parts.append(maybe_text)
        return "".join(text_parts)
    return str(content)


def _parse_choices_with_schema(schema_response: Any, completion_choices: List[Any]) -> List[Any]:
    """Parse each choice's content into the provided schema."""
    parsed_payloads = []
    for choice in completion_choices:
        content_str = _extract_message_text(choice.message.content).strip()
        parsed_payloads.append(schema_response.model_validate_json(content_str))
    return parsed_payloads


def _wrap_completion_with_parsed(raw_completion: Any, parsed_payloads: List[Any]) -> SimpleNamespace:
    """Attach parsed payloads to a lightweight completion wrapper."""
    wrapped_choices = []
    for choice, parsed in zip(raw_completion.choices, parsed_payloads):
        message_ns = SimpleNamespace(
            parsed=parsed,
            content=choice.message.content,
            role=choice.message.role,
        )
        choice_ns = SimpleNamespace(
            index=getattr(choice, "index", None),
            message=message_ns,
            logprobs=getattr(choice, "logprobs", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )
        wrapped_choices.append(choice_ns)

    def _to_dict():
        return raw_completion.to_dict()

    return SimpleNamespace(
        choices=wrapped_choices,
        raw_completion=raw_completion,
        to_dict=_to_dict,
    )


def _transform_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Augment chat messages so the Responses API accepts text payloads."""

    def _transform_content_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        transformed = dict(item)
        content_type = transformed.get("type")
        if content_type == "text":
            transformed["type"] = "input_text"
        elif content_type == "image_url":
            image_url = transformed.get("image_url")
            if isinstance(image_url, dict):
                # Responses API expects a bare data URL string
                transformed["image_url"] = image_url.get("url")
            transformed["type"] = "input_image"
        return transformed

    transformed_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            transformed_messages.append(message)
            continue

        updated_message = dict(message)
        content = updated_message.get("content")

        if isinstance(content, list):
            updated_message["content"] = [_transform_content_item(entry) for entry in content]
        elif isinstance(content, dict):
            updated_message["content"] = _transform_content_item(content)

        transformed_messages.append(updated_message)

    return transformed_messages


def _gemini_logprobs_to_openai_like(raw_dict: Dict[str, Any]) -> Any:
    """Convert Gemini logprobs payload into an OpenAI-like object shape.

    Existing SSR parsers expect:
      completion.choices[i].logprobs.content[j].token
      completion.choices[i].logprobs.content[j].logprob
      completion.choices[i].logprobs.content[j].top_logprobs
    """
    if not isinstance(raw_dict, dict):
        return None

    gemini_lp = raw_dict.get("_gemini_logprobs")
    if not isinstance(gemini_lp, dict) or not gemini_lp.get("available"):
        return None

    token_entries = gemini_lp.get("tokens") or []
    top_entries = gemini_lp.get("top_candidates") or []
    if not isinstance(token_entries, list) or not token_entries:
        return None

    content = []
    for idx, token_obj in enumerate(token_entries):
        token = str((token_obj or {}).get("token", ""))
        logprob = float((token_obj or {}).get("log_probability", 0.0))

        top_logprobs = []
        if isinstance(top_entries, list) and idx < len(top_entries) and isinstance(top_entries[idx], list):
            for cand in top_entries[idx]:
                cand_token = str((cand or {}).get("token", ""))
                cand_logprob = float((cand or {}).get("log_probability", 0.0))
                top_logprobs.append(
                    SimpleNamespace(token=cand_token, logprob=cand_logprob)
                )

        content.append(
            SimpleNamespace(token=token, logprob=logprob, top_logprobs=top_logprobs)
        )

    return SimpleNamespace(content=content)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 10.0,
    max_delay: float = 120.0,
    backoff_factor: float = 2.0
):
    """
    Decorator that retries a function with exponential backoff on rate limit errors
    and transient server errors (503, 500, unreachable_backend, etc.).
    Works for Mistral (SDKError), OpenAI (RateLimitError), and other API errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limit indicators in the error
                    is_rate_limit = (
                        "429" in str(e) or 
                        "rate" in error_str or 
                        "capacity" in error_str or
                        "quota" in error_str or
                        "tier" in error_str
                    )
                    # Check for transient server errors that should be retried
                    is_transient_error = (
                        "503" in str(e) or
                        "500" in str(e) or
                        "502" in str(e) or
                        "504" in str(e) or
                        "unreachable" in error_str or
                        "internal server error" in error_str or
                        "service unavailable" in error_str or
                        "backend" in error_str or
                        "timeout" in error_str or
                        "connection" in error_str
                    )
                    
                    if (is_rate_limit or is_transient_error) and attempt < max_retries:
                        error_type = "Rate limit" if is_rate_limit else "Transient server error"
                        print(f"[Retry {attempt + 1}/{max_retries}] {error_type} hit: {str(e)[:100]}. Waiting {delay:.1f}s before retry...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                        last_exception = e
                    else:
                        raise e
            
            # If we exhausted retries
            raise last_exception
        return wrapper
    return decorator


# Function to send messages to OpenAI and parse the structured output
@retry_with_exponential_backoff(max_retries=10, initial_delay=30.0, max_delay=300.0)
def get_structured_output_parsed(messages: List[Dict[str, Any]], schema_response: Any, top_logprobs: int = None, number_answers: int = 1) -> (Dict[str, Any], Any):
    """Sends messages to OpenAI and parses the structured output."""
    if top_logprobs is None:
        top_logprobs = DEFAULT_TOP_LOGPROBS
    try:
        # Rate limit per API key (Adjust as necessary)
        with api_key_lock:
            current_time = time.time()
            elapsed = current_time - api_key_last_called[OPENAI_API_KEY]
            #only invoke if rate issues
            # wait_time = 2.0 - elapsed
            # if wait_time > 0:
            #     time.sleep(wait_time)
            # api_key_last_called[OPENAI_API_KEY] = time.time()
        
        print(f"Sending API request to provider={MODEL_TO_USE} model={MODEL_NAME}")
        print("len messages: ",len(messages))
        schema_payload = (
            schema_response.model_json_schema()
            if hasattr(schema_response, "model_json_schema")
            else str(schema_response)
        )
        _req_path = os.environ.get('REQUESTS_LOG_PATH', 'requests.jsonl')
        with open(_req_path, 'a') as f:
            f.write(
                json.dumps(
                    {
                        "provider": MODEL_TO_USE,
                        "model": MODEL_NAME,
                        "max_tokens": MAX_TOKENS,
                        "use_logprobs": USE_LOGPROBS,
                        "top_logprobs": top_logprobs,
                        "number_answers": number_answers,
                        "messages": messages,
                        "response_schema": schema_payload,
                    },
                    default=str,
                )
            )
            f.write('\n')
        # Send the request to OpenAI
        parsed_content = None
        _raw_response = None
        # Get the correct parameters for OpenAI (GPT-5+ uses different params)
        openai_params = get_openai_params() if MODEL_TO_USE == "openai" else {}
        if MODEL_TO_USE == "openai" and USE_LOGPROBS and top_logprobs > 0:
            completion = client.beta.chat.completions.parse(	
                model=MODEL_NAME,	
                messages=messages,	
                response_format=schema_response,	
                **openai_params,	
                logprobs=True,
                top_logprobs=top_logprobs,
                n = number_answers
            )
            parsed_content = completion.choices[0].message.parsed
        elif MODEL_TO_USE == "openai":
            # Use beta.chat.completions.parse for structured outputs with logprobs
            print(f"Using OpenAI structured output API with model: {MODEL_NAME}")
            completion = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=messages,
                response_format=schema_response,
                **openai_params,
                logprobs=USE_LOGPROBS,
                top_logprobs=top_logprobs if USE_LOGPROBS else None,
                n=number_answers
            )
            parsed_content = completion.choices[0].message.parsed
        elif MODEL_TO_USE == "mistral" and USE_LOGPROBS and top_logprobs > 0:
            completion = client.chat.parse(
                model=MODEL_NAME,
                messages=messages,
                response_format=schema_response,
                max_tokens=min(MAX_TOKENS, 16384),
                temperature=0.2,
                n = number_answers,
                # logprobs=True,
                # top_logprobs=top_logprobs
            )
            parsed_content = completion.choices[0].message.parsed
        elif MODEL_TO_USE == "mistral":
            completion = client.chat.parse(
                model=MODEL_NAME,
                messages=messages,
                response_format=schema_response,
                max_tokens=min(MAX_TOKENS, 16384),
                temperature=0.2,
                n = number_answers
            )
            parsed_content = completion.choices[0].message.parsed
        elif MODEL_TO_USE == "selfhosted" and USE_LOGPROBS and top_logprobs > 0:
            completion_raw = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=min(MAX_TOKENS, 16384),
                temperature=0.2,
                logprobs=True,
                top_logprobs=top_logprobs,
                n=number_answers,
            )
            parsed_payloads = _parse_choices_with_schema(schema_response, completion_raw.choices)
            completion = _wrap_completion_with_parsed(completion_raw, parsed_payloads)
            parsed_content = parsed_payloads[0]
        elif MODEL_TO_USE == "selfhosted":
            completion_raw = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=min(MAX_TOKENS, 16384),
                temperature=0.2,
                n=number_answers,
            )
            parsed_payloads = _parse_choices_with_schema(schema_response, completion_raw.choices)
            completion = _wrap_completion_with_parsed(completion_raw, parsed_payloads)
            parsed_content = parsed_payloads[0]
        # -------------------- GEMINI (new google-genai SDK) -----------------------
        elif MODEL_TO_USE == "gemini":
            if not GEMINI_AVAILABLE or client is None:
                raise ValueError("Gemini client not available. Check GOOGLE_API_KEY/GEMINI_API_KEY and google-genai installation.")
            # Use the new GeminiClient.send_structured_request_from_messages()
            # When USE_LOGPROBS is true, always enable logprobs (minimum count=1)
            gemini_logprobs_count = max(1, top_logprobs) if top_logprobs else 1
            gemini_enable_logprobs = bool(USE_LOGPROBS)
            parsed_content, raw_dict, _raw_response = client.send_structured_request_from_messages(
                messages,
                schema_response,
                max_tokens=MAX_TOKENS,
                enable_logprobs=gemini_enable_logprobs,
                logprobs_count=gemini_logprobs_count,
            )
            gemini_logprobs = _gemini_logprobs_to_openai_like(raw_dict)
            if USE_LOGPROBS and gemini_logprobs is None:
                print("[gemini] USE_LOGPROBS=true but token-level logprobs were not returned by the API response.")
            gemini_raw = _raw_response.to_dict() if hasattr(_raw_response, "to_dict") else str(_raw_response)
            # Create a mock completion object for consistency with other backends
            # TODO: Make sure it does NOT use a MOCK FUNCTION!
            completion = type('GeminiCompletion', (), {
                'choices': [type('Choice', (), {
                    'message': type('Message', (), {'parsed': parsed_content})(),
                    'logprobs': gemini_logprobs,
                })()],
                'to_dict': lambda self: {
                    'model': MODEL_NAME,
                    'parsed': str(parsed_content),
                    '_gemini_logprobs': raw_dict.get('_gemini_logprobs', {}) if isinstance(raw_dict, dict) else {},
                    '_gemini_usage': raw_dict.get('_gemini_usage', {}) if isinstance(raw_dict, dict) else {},
                    '_gemini_context_cache': raw_dict.get('_gemini_context_cache', {}) if isinstance(raw_dict, dict) else {},
                    '_gemini_raw_response': gemini_raw,
                }
            })()
        # -------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported MODEL_TO_USE '{MODEL_TO_USE}' in structured output parser.")

        if parsed_content is None:
            if hasattr(completion, "output_parsed"):
                parsed_content = completion.output_parsed
            else:
                parsed_content = completion.choices[0].message.parsed
        # Save the response to a file for debugging
        _resp_path = os.environ.get('RESPONSES_LOG_PATH', 'responses.json')
        with open(_resp_path, 'a') as f:
            if MODEL_TO_USE == "openai":
                f.write(json.dumps(completion.to_dict()))
                f.write('\n')
            elif MODEL_TO_USE == "selfhosted":
                f.write(json.dumps(completion.to_dict()))
                f.write('\n')
            elif MODEL_TO_USE == "mistral":
                f.write(parsed_content.json())
                f.write('\n')
            elif MODEL_TO_USE == "gemini":
                f.write(json.dumps(completion.to_dict()))
                f.write('\n')

        _resp_content_path = os.environ.get('RESPONSES_CONTENT_LOG_PATH', 'responses_content.json')
        with open(_resp_content_path, 'a') as f:
            if MODEL_TO_USE == "openai":
                f.write(json.dumps(str(parsed_content)))
                f.write('\n')
            if MODEL_TO_USE == "selfhosted":
                f.write(json.dumps(str(parsed_content)))
                f.write('\n')
            elif MODEL_TO_USE == "mistral":
                f.write(parsed_content.json())
                f.write('\n')
            elif MODEL_TO_USE == "gemini":
                gemini_payload = {
                    "parsed": str(parsed_content),
                    "_gemini_usage": raw_dict.get("_gemini_usage", {}) if isinstance(raw_dict, dict) else {},
                    "_gemini_context_cache": raw_dict.get("_gemini_context_cache", {}) if isinstance(raw_dict, dict) else {},
                }
                f.write(json.dumps(gemini_payload))
                f.write('\n')
        print(f"Response from Model: {parsed_content}")
        #.to_dict() before
        return completion, parsed_content
    
    except ValidationError as ve:
        print(f"Validation error while parsing response: {ve}")
        return None, None
    except json.decoder.JSONDecodeError:
        print("JSON decode error in response.")
        return None, None
    except KeyboardInterrupt:
        sys.exit()
    except RateLimitError as rle:
        # Re-raise so the retry decorator can handle it
        print(f"Rate limit error (will retry): {rle}")
        raise
    except LengthFinishReasonError as lf:
        print("Length finish reason error:", lf)
        print("Sleeping for 20 seconds.")
        time.sleep(20)
        return None, None

def get_predicates_structured(object_list_instance: Any, pddl_domain_file_path: str) -> Dict[str, Any]:
    """Gets structured predicates based on the object list and PDDL domain."""
    types_available_classes_mapping.clear()
    for object_instance in object_list_instance.objects:
        if object_instance.type not in types_available_classes_mapping:
            types_available_classes_mapping[object_instance.type] = []
        types_available_classes_mapping[object_instance.type].append(
            object_instance.name
        )
        
    domain_0 = parse_domain(pddl_domain_file_path)

    # Creating the inverted dictionary
    domain_types = {}
    for key, value in domain_0.types.items():
        if value not in domain_types:
            domain_types[value] = []
        domain_types[value].append(key)
    def _collect_objects_for_type(type_name):
        """Recursively collect objects for a type, resolving multi-level hierarchies."""
        if type_name in types_available_classes_mapping:
            return types_available_classes_mapping[type_name]
        if type_name in domain_types:
            types_available_classes_mapping[type_name] = []
            for subclass in domain_types[type_name]:
                types_available_classes_mapping[type_name].extend(
                    _collect_objects_for_type(subclass)
                )
            return types_available_classes_mapping[type_name]
        return []

    for superclass in domain_types.keys():
        if superclass is not None and "None" not in superclass:
            _collect_objects_for_type(superclass)

    # Ensure 'object' (PDDL root type) includes ALL objects
    if "object" not in types_available_classes_mapping or not types_available_classes_mapping["object"]:
        all_obj_names = []
        for names in types_available_classes_mapping.values():
            for n in names:
                if n not in all_obj_names:
                    all_obj_names.append(n)
        if all_obj_names:
            types_available_classes_mapping["object"] = all_obj_names

    
    classes_with_objects = {k for k, v in types_available_classes_mapping.items() if v}
    domain_wrapper = PDDLDomainWrapper(pddl_domain_file_path)

    predicates = {}
    for gen_pred in domain_wrapper.predicates:
        pred = {
            "name": gen_pred.name,
            "arity": gen_pred.arity,
            "types": [t.type_tags for t in gen_pred.terms],
        }
        predicates[gen_pred.name] = pred

    def transform_dict(original_dict):
        simplified_dict = {
            k: [list(t)[0] if t else "default" for t in v["types"]] for k, v in original_dict.items()
        }
        return simplified_dict

    predicates = transform_dict(predicates)

    predicates_structured = {}
    for pred_name, pred in predicates.items():
        next_char = "x"
        dict_how_are_they_named = {}
        for type_name in pred:
            dict_how_are_they_named[next_char] = type_name
            next_char = next_letter(next_char)
        all_classes_used = set(dict_how_are_they_named.values())
        if all_classes_used - classes_with_objects == set():
            predicates_structured[pred_name] = dict_how_are_they_named
        else:
            missing_classes = all_classes_used - classes_with_objects
            print(
                f"No object for classes: {pred_name} because an object of class {missing_classes} is missing 2"
            )
    return predicates_structured

def load_dataset(file_path: str) -> Dict[str, Any]:
    """Loads the dataset from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def _image_content_hash(image_path: str) -> Optional[str]:
    """Return MD5 hex digest of file contents, or None if unreadable."""
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except (OSError, IOError):
        return None


def get_random_examples(dataset: Dict[str, Any], problem_num: int, num_examples: int) -> List[tuple]:
    """Gets random examples from the dataset.

    Excludes the current problem_num AND any problem whose state images are
    identical to the test problem's images — checked both by file path AND by
    content hash (MD5).  This prevents the model from being shown the correct
    answer during few-shot when different file paths point to byte-identical
    images (common in ViLaIn, PDDLGym, etc.).

    If there are fewer unique eligible examples than requested, duplicates of
    already-selected examples are allowed so that the model still receives the
    requested number of few-shot demonstrations.

    Uses a per-problem local RNG seeded from (_GLOBAL_SEED, problem_num) so
    the selected examples are always identical for a given problem number,
    regardless of the order in which problems are processed.
    """
    # Template variant allows controlled prompt variation while still reusing caches.
    if EXAMPLE_TEMPLATE_FIXED.isdigit():
        template_variant = max(0, min(EXAMPLE_TEMPLATE_VARIANTS - 1, int(EXAMPLE_TEMPLATE_FIXED)))
    elif EXAMPLE_TEMPLATE_MODE == "stable":
        template_variant = (int(problem_num) + int(_GLOBAL_SEED)) % EXAMPLE_TEMPLATE_VARIANTS
    else:
        template_variant = random.randint(0, EXAMPLE_TEMPLATE_VARIANTS - 1)

    # Local RNG: seed includes template variant.
    rng = random.Random(f"{_GLOBAL_SEED}_{problem_num}_v{template_variant}")  # str seed: Python 3.12+ compatible
    # Collect image paths AND content hashes for the test problem.
    test_image_paths: set = set()
    test_image_hashes: set = set()
    for state in dataset["problems"][problem_num].get("states", []):
        img = state.get("image_path", "")
        if img:
            test_image_paths.add(img)
            h = _image_content_hash(img)
            if h:
                test_image_hashes.add(h)

    # Build the pool of eligible (problem, state) pairs — same domain, different
    # image content from the test.
    eligible = []
    for pid, prob in enumerate(dataset["problems"]):
        if pid == problem_num:
            continue
        if prob["domain_file"] != dataset["problems"][problem_num]["domain_file"]:
            continue
        for sid, st in enumerate(prob.get("states", [])):
            cand_img = st.get("image_path", "")
            if cand_img in test_image_paths:
                continue
            if cand_img and test_image_hashes:
                cand_hash = _image_content_hash(cand_img)
                if cand_hash and cand_hash in test_image_hashes:
                    continue
            eligible.append((pid, sid))

    if not eligible:
        return []

    # Deduplicate by image content so we prefer variety first.
    seen_hashes: set = set()
    unique_eligible = []
    for pid, sid in eligible:
        img = dataset["problems"][pid]["states"][sid].get("image_path", "")
        h = _image_content_hash(img) if img else None
        if h and h in seen_hashes:
            continue
        if h:
            seen_hashes.add(h)
        unique_eligible.append((pid, sid))

    # If enough unique examples, sample without replacement.
    # Otherwise, use all unique ones and fill remaining with duplicates.
    chain_examples = []
    if len(unique_eligible) >= num_examples:
        chain_examples = rng.sample(unique_eligible, num_examples)
    else:
        chain_examples = list(unique_eligible)
        remaining = num_examples - len(chain_examples)
        for _ in range(remaining):
            chain_examples.append(rng.choice(eligible))

    if LOG_EXAMPLES:
        print(
            f"[TEMPLATE] problem={problem_num} variant={template_variant} "
            f"variants={EXAMPLE_TEMPLATE_VARIANTS} mode={EXAMPLE_TEMPLATE_MODE}"
        )

    return chain_examples

def get_example_data(dataset: Dict[str, Any], problem_num: int, state_num: int) -> Dict[str, Any]:
    """Retrieves example data from the dataset."""
    problem = dataset["problems"][problem_num]
    state = problem["states"][state_num]
    return {
        "current_state": state,
        "atoms_schema_json": state["atoms_schema"],
        "atoms_json_curr": state["atoms"],
        "objects_json_curr": state["all_objects"],
        "image_path_curr": state["image_path"],
        ### MODIFICATION FOR GOALS START
        "goal_instruction_path": state.get("goal_instruction_path", None),
        "goal_predicates": state.get("goal_predicates", None),
        ### MODIFICATION FOR GOALS END
    }

def get_message_example_from_iteration(
    domain_file: str,
    current_state: Dict[str, Any],
    atoms_schema_json: str,
    atoms_json_curr: str,
    objects_json_curr: str,
    image_path_curr: str,
    ### MODIFICATION FOR GOALS START
    goal_instruction_path: str = None,
    goal_predicates: str = None,
    ### MODIFICATION FOR GOALS END
) -> tuple:
    """Generates example messages from a given iteration."""
    image_base64_curr = encode_image_base64(image_path=image_path_curr)

    object_schema_curr, list_types_curr = get_object_schema_from_pddl(domain_file)

    # Rebuild schema if ground-truth objects use types not in the domain
    gt_objects_data = json.loads(objects_json_curr)
    gt_types = {o["type"] for o in gt_objects_data.get("objects", [])}
    extra_types = gt_types - set(list_types_curr)
    if extra_types:
        all_types = tuple(sorted(set(list_types_curr) | gt_types))
        list_types_curr = all_types
        from pydantic import Field as _Field
        class _Object(BaseModel):
            name: str
            type: Literal[all_types] = _Field(..., description="Must be one of the valid types")
        class _ObjectList(BaseModel):
            objects: List[_Object]
        object_schema_curr = _ObjectList

    object_list_curr = object_schema_curr(**gt_objects_data)
    object_schema_json_str = object_schema_curr.schema_json(indent=2)
    prompt_get_objects_curr = get_all_objects_prompt(list_types_curr, object_schema_json_str)

    instruction_text = None
    if (
        INCLUDE_INSTRUCTION_IN_EXAMPLES
        and goal_instruction_path
        and os.path.exists(goal_instruction_path)
    ):
        with open(goal_instruction_path, "r") as f:
            instruction_text = f.read().strip() or None
    if instruction_text:
        prompt_get_objects_curr += (
            "\nTask instruction (STRICT: include ONLY task-relevant objects):\n"
            f"{instruction_text}"
        )
    predicates_structured_curr = get_predicates_structured(object_list_curr, domain_file)
    Atoms_schema = get_atom_schema_from_chatgpt_objects(object_list_curr, domain_file)
    atoms_schema_json_str = Atoms_schema.schema_json(indent=2)
    prompt_get_atoms_curr = get_all_grounded_atoms_prompt(
        predicates_structured_curr, object_list_curr, atoms_schema_json_str
    )
    if instruction_text:
        prompt_get_atoms_curr += (
            "\nTask instruction (STRICT: include ONLY task-relevant grounded predicates/atoms):\n"
            f"{instruction_text}"
        )

    message_get_object_curr = make_messages_image(prompt_get_objects_curr, image_base64_curr)
    message_original_get_atoms_curr = make_messages_image(prompt_get_atoms_curr, image_base64_curr)

    objects_curr = object_schema_curr(**json.loads(objects_json_curr)).json()
    atoms_curr = Atoms_schema(**json.loads(atoms_json_curr)).json() 

    answer_objects_curr = make_answer_text(objects_curr)
    answer_atoms_curr = make_answer_text(atoms_curr)

    ### MODIFICATION FOR GOALS START
    # For examples with goals, if present:
    goal_example_messages = []
    #check if goal_instruction_path actually exists
    exists_goal_instruction_path = os.path.exists(goal_instruction_path)
    if goal_instruction_path and goal_predicates and exists_goal_instruction_path:
        # Parse goal_predicates into a goal-specific schema that supports is_negated
        # so that the few-shot example demonstrates the negation field format.
        Goals_schema = get_atom_schema_from_chatgpt_objects(
            object_list_curr, domain_file, include_negation=True
        )
        goals_curr = Goals_schema(**json.loads(goal_predicates))
        goals_schema_json_str = Goals_schema.schema_json(indent=2)
        with open(goal_instruction_path, 'r') as f:
            instruction_text = f.read()
        prompt_get_goals_curr = get_all_goal_predicates_prompt(instruction_text, goals_schema_json_str)
        # Goals are from text, not from image, so use make_message_text
        message_get_goals_curr = make_message_text(prompt_get_goals_curr)
        answer_goals_curr = make_answer_text(goals_curr)
        goal_example_messages = ([message_get_goals_curr, answer_goals_curr])
    ### MODIFICATION FOR GOALS END

    # Return tuple including goals if available
    return (
        [message_get_object_curr, answer_objects_curr],
        [message_original_get_atoms_curr, answer_atoms_curr],
        ### MODIFICATION FOR GOALS START
        goal_example_messages
        ### MODIFICATION FOR GOALS END
    )

def get_example_history(dataset, domain_file, chain_examples):
    """Creates an example history for few-shot learning."""
    example_history = []
    for problem_num, state_num in chain_examples:
        example_data = get_example_data(dataset, problem_num, state_num)
        example_message = get_message_example_from_iteration(
            domain_file, **example_data
        )
        example_history.append(example_message)
    return example_history

def evaluate_predicates(results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the predicates against the ground truth."""
    def _lower_arg_value(v):
        """Lowercase dict values (e.g. object names) for case-insensitive matching."""
        if isinstance(v, dict):
            return tuple((dk, dv.lower() if isinstance(dv, str) else dv) for dk, dv in v.items())
        if isinstance(v, str):
            return v.lower()
        return v

    def predicate_to_tuple(predicate):
        p_type = predicate["predicate_type"].lower()
        # is_negated: treat missing or False as positive (False), True as negated.
        # Exclude is_negated from args so it can be handled uniformly.
        is_neg = bool(predicate.get("is_negated", False))
        args = tuple(
            sorted(
                (k, _lower_arg_value(v))
                for k, v in predicate.items()
                if k != "predicate_type" and k != "is_negated"
            )
        )
        return (p_type, is_neg, args)

    def get_predicate_set(predicates):
        result = set()
        for predicate in predicates:
            if isinstance(predicate, dict):
                result.add(predicate_to_tuple(predicate))
            # skip non-dict predicates (e.g. stringified Pydantic objects)
        return result

    results_set = get_predicate_set(results["grounded_predicates"])
    ground_set = get_predicate_set(ground_truth["grounded_predicates"])

    # False Positives and Negatives
    false_positives = results_set - ground_set
    false_negatives = ground_set - results_set

    # Check for permutations
    permutations_set = set()
    for fn in false_negatives.copy():
        p_type, is_neg, args = fn
        vars = [
            dict(arg[1])["name"]
            for arg in args
            if isinstance(arg[1], tuple) and dict(arg[1]).get("name")
        ]
        for perm in permutations(vars):
            perm_args = tuple(
                (
                    k,
                    tuple({"name": perm[vars.index(dict(v)["name"])]}.items())
                    if k != "predicate_type" and isinstance(v, tuple)
                    else v,
                )
                for k, v in args
            )
            perm_pred = (p_type, is_neg, perm_args)
            if perm_pred in results_set:
                permutations_set.add(fn)
                false_negatives.remove(fn)
                break

    def tuple_to_predicate(t):
        p_type, is_neg, args = t
        predicate = {"predicate_type": p_type, "is_negated": is_neg}
        for k, v in args:
            if isinstance(v, tuple):
                predicate[k] = dict(v)
            else:
                predicate[k] = v
        return predicate

    num_total = len(ground_set)
    num_correct = num_total - len(false_negatives)

    # Handle empty ground truth: if both GT and prediction are empty → perfect
    if num_total == 0:
        if len(false_positives) == 0:
            accuracy, precision, recall, f1 = 1.0, 1.0, 1.0, 1.0
        else:
            accuracy, precision, recall, f1 = 0.0, 0.0, 1.0, 0.0
    else:
        accuracy = num_correct / num_total
        precision = (
            num_correct / (num_correct + len(false_positives))
            if (num_correct + len(false_positives)) > 0
            else 0
        )
        recall = num_correct / num_total
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

    return {
        "false_positives": [tuple_to_predicate(fp) for fp in false_positives],
        "false_negatives": [tuple_to_predicate(fn) for fn in false_negatives],
        "permutations": [tuple_to_predicate(p) for p in permutations_set],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def evaluate_objects(results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the objects against the ground truth."""
    def objects_to_set(objects_list):
        return set((obj['name'].lower(), obj['type'].lower()) for obj in objects_list)

    results_set = objects_to_set(results['objects'])
    ground_set = objects_to_set(ground_truth['objects'])

    # Calculate true positives, false positives, and false negatives
    true_positives = results_set & ground_set
    false_positives = results_set - ground_set
    false_negatives = ground_set - results_set

    # Calculate metrics
    num_total = len(ground_set)
    num_correct = len(true_positives)

    # Handle empty ground truth: if both GT and prediction are empty → perfect
    if num_total == 0:
        if len(false_positives) == 0:
            accuracy, precision, recall, f1 = 1.0, 1.0, 1.0, 1.0
        else:
            accuracy, precision, recall, f1 = 0.0, 0.0, 1.0, 0.0
    else:
        accuracy = num_correct / num_total
        precision = (
            num_correct / (num_correct + len(false_positives))
            if (num_correct + len(false_positives)) > 0
            else 0
        )
        recall = num_correct / num_total
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

    return {
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def calculate_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates aggregate metrics from evaluation results."""
    if not evaluation_results:
        return {
            "avg_accuracy": 0,
            "avg_precision": 0,
            "avg_recall": 0,
            "avg_f1": 0,
            "avg_permutation_percentage": 0
        }

    accuracy_list = [res['accuracy'] for res in evaluation_results]
    precision_list = [res['precision'] for res in evaluation_results]
    recall_list = [res['recall'] for res in evaluation_results]
    f1_list = [res['f1'] for res in evaluation_results]
    permutation_percentages = []

    for result in evaluation_results:
        total_predicates = (
            len(result.get('false_positives', [])) +
            len(result.get('false_negatives', [])) +
            len(result.get('permutations', []))
        )
        permutation_percentage = (
            len(result.get('permutations', [])) / total_predicates
            if total_predicates > 0 else 0
        )
        permutation_percentages.append(permutation_percentage)

    metrics = {
        "avg_accuracy": statistics.mean(accuracy_list) if accuracy_list else 0,
        "avg_precision": statistics.mean(precision_list) if precision_list else 0,
        "avg_recall": statistics.mean(recall_list) if recall_list else 0,
        "avg_f1": statistics.mean(f1_list) if f1_list else 0,
        "avg_permutation_percentage": statistics.mean(permutation_percentages) if permutation_percentages else 0,
    }

    return metrics

def calculate_metrics_objects(evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates aggregate metrics for objects."""
    if not evaluation_results:
        return {
            "avg_accuracy": 0,
            "avg_precision": 0,
            "avg_recall": 0,
            "avg_f1": 0,
        }
    accuracy_list = [res['accuracy'] for res in evaluation_results]
    precision_list = [res['precision'] for res in evaluation_results]
    recall_list = [res['recall'] for res in evaluation_results]
    f1_list = [res['f1'] for res in evaluation_results]

    metrics = {
        "avg_accuracy": statistics.mean(accuracy_list) if accuracy_list else 0,
        "avg_precision": statistics.mean(precision_list) if precision_list else 0,
        "avg_recall": statistics.mean(recall_list) if recall_list else 0,
        "avg_f1": statistics.mean(f1_list) if f1_list else 0,
    }

    return metrics

def _extract_domain_identifier(result: Dict[str, Any]) -> str:
    """Returns a domain identifier parsed from supported fields."""
    batch_or_request = result.get("request_id") or result.get("request_id") or ""
    if "_domain_" in batch_or_request:
        return batch_or_request.split("_domain_", 1)[1]

    domain_file = result.get("domain_file") or result.get("domain")
    if isinstance(domain_file, str) and domain_file:
        return os.path.splitext(os.path.basename(domain_file))[0]

    return "unknown_domain"

def _aggregate_metrics_per_domain(
    evaluation_results: List[Dict[str, Any]],
    aggregate_fn,
) -> Dict[str, Dict[str, float]]:
    """Groups evaluation results per domain and applies the selected aggregation."""
    if not evaluation_results:
        return {}

    grouped_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in evaluation_results:
        domain = _extract_domain_identifier(result)
        grouped_results[domain].append(result)

    metrics_per_domain: Dict[str, Dict[str, float]] = {}
    for domain, results in grouped_results.items():
        domain_metrics = dict(aggregate_fn(results))
        domain_metrics["num_samples"] = len(results)
        metrics_per_domain[domain] = domain_metrics

    return dict(sorted(metrics_per_domain.items()))

def calculate_metrics_per_domain(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Calculates aggregate metrics per domain for atom/goal style evaluations."""
    return _aggregate_metrics_per_domain(evaluation_results, calculate_metrics)

def calculate_metrics_objects_per_domain(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Calculates aggregate metrics per domain for object evaluations."""
    return _aggregate_metrics_per_domain(evaluation_results, calculate_metrics_objects)

def get_model_encodings(model_name: str):
    """Gets the encoding for a given model."""
    return tiktoken.get_encoding("cl100k_base")

def get_image_size(image_path: str) -> tuple:
    """
    Retrieves the size of an image.
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error retrieving image size: {e}")
        return None

def resize_image(image_path: str, max_width: int = 800, max_height: int = 800) -> str:
    """
    Resizes an image while maintaining aspect ratio.
    """
    try:
        with Image.open(image_path) as img:
            original_size = img.size
            img.thumbnail((max_width, max_height), Image.ANTIALIAS)
            resized_size = img.size

            if resized_size == original_size:
                print(f"No resizing needed for {image_path}.")
                return image_path

            base, ext = os.path.splitext(image_path)
            resized_image_path = f"{base}_resized{ext}"
            img.save(resized_image_path)
            print(f"Image resized from {original_size} to {resized_size} and saved as {resized_image_path}.")
            return resized_image_path

    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return image_path
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_path

def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            print("Image Size:", get_image_size(image_path))
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_and_create_example_from_dataset(dataset_path: str, problem_num: int, num_examples: int) -> tuple:
    """Loads and creates example messages from the dataset."""
    dataset = load_dataset(dataset_path)
    domain_file = dataset["problems"][problem_num]["domain_file"]
    chain_examples = get_random_examples(dataset, problem_num, num_examples)
    if LOG_EXAMPLES:
        _example_meta = []
        for ex_problem_num, ex_state_num in chain_examples:
            ex_state = dataset["problems"][ex_problem_num]["states"][ex_state_num]
            _example_meta.append(
                {
                    "example_problem": ex_problem_num,
                    "example_state": ex_state_num,
                    "example_image_path": ex_state.get("image_path"),
                }
            )
        print(
            f"[EXAMPLES] target_problem={problem_num} num_examples={num_examples} "
            f"selected_examples={json.dumps(_example_meta)}"
        )
    example_history = get_example_history(dataset, domain_file, chain_examples)
    example_history_objects = [example[0] for example in example_history]
    example_history_objects_final = [
        item for sublist in example_history_objects for item in sublist
    ]
    example_history_atoms = [example[1] for example in example_history]
    example_history_atoms_final = [
        item for sublist in example_history_atoms for item in sublist
    ]
    ### MODIFICATION FOR GOALS START
    # Extract goal examples if any exist
    example_history_goals = [example[2] for example in example_history if len(example) > 2 and example[2]]
    example_history_goals_final = []
    for sublist in example_history_goals:
        for item in sublist:
            example_history_goals_final.append(item)
    ### MODIFICATION FOR GOALS END
    return example_history_objects_final, example_history_atoms_final, example_history_goals_final

def process_sample(
    dataset,
    dataset_path,
    problem_id,
    state_num,
    num_examples,
    id_counter,
    evaluation_results_objects,
    evaluation_results_atoms,
    ### MODIFICATION FOR GOALS START
    evaluation_results_goals,
    ### MODIFICATION FOR GOALS END
    api_key,
):
    """
    Processes a single sample by performing the steps for objects, atoms,
    and now also goals if available.
    """
    # Load and prepare data
    example_history_objects, example_history_atoms, example_history_goals = load_and_create_example_from_dataset(
        dataset_path, problem_id, num_examples
    )
    print("Processing Sample: Problem id:", problem_id, "State number:", state_num)

    problem_json = dataset["problems"][problem_id]
    domain_file = problem_json["domain_file"]
    random_state = problem_json["states"][state_num]
    atoms_schema_json = random_state["atoms_schema"]
    atoms_json_curr = random_state["atoms"]
    objects_json_curr_correct = random_state["all_objects"]
    image_path_curr = random_state["image_path"]

    ### MODIFICATION FOR GOALS START
    goal_instruction_path = random_state.get("goal_instruction_path", None)
    goal_predicates = random_state.get("goal_predicates", None)
    ### MODIFICATION FOR GOALS END

    image_base64_curr = encode_image(image_path=image_path_curr)
    object_schema_curr, list_types_curr = get_object_schema_from_pddl(domain_file)

    # Get JSON schema as string
    object_schema_json_str = object_schema_curr.schema_json(indent=2)
    prompt_get_objects_curr = get_all_objects_prompt(list_types_curr, object_schema_json_str)

    # Create messages for objects
    message_get_object_curr = make_messages_image(
        prompt_get_objects_curr, image_base64_curr
    )

    # Combine with example history
    messages_get_object = [*example_history_objects, *message_get_object_curr]

    object_correct_instance_curr = object_schema_curr(**json.loads(objects_json_curr_correct))

    response_format_objects = object_schema_curr

    request_id = f"{id_counter}_problem_{problem_id}_state_{state_num}_domain_{problem_json['problem_name']}"

    # Step: Get Objects
    messages_get_object_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_get_object
    ]

    print(f"Making API call for Objects with ID: {request_id}")
    response_objects, object_list_instance = get_structured_output_parsed(
        messages=messages_get_object_together,
        schema_response=response_format_objects,
    )
    
    if object_list_instance is not None:
        full_obj_resp = {
            "request_id": request_id,
            "objects": object_list_instance.dict().get('objects', []),
            "domain_file": domain_file
        }
        with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_OBJECTS_FILE), 'a') as f_obj_full:
            json.dump(full_obj_resp, f_obj_full)
            f_obj_full.write('\n')

    # Evaluate objects
    if object_list_instance is not None:
        answer_objects = {
            "objects": object_list_instance.dict().get('objects', []),
        }
        ground_truth_objects = object_correct_instance_curr.dict()
        eval_result_object = evaluate_objects(answer_objects, ground_truth_objects)
        eval_result_object.update({"request_id": request_id})
        evaluation_results_objects.append(eval_result_object)
    else:
        print(f"Failed to parse Objects response for ID: {request_id}")

    # Step: Get Atoms
    Atoms_schema_correct = get_atom_schema_from_chatgpt_objects(
        object_correct_instance_curr, domain_file
    )
    Atoms_schema_chatGPT = get_atom_schema_from_chatgpt_objects(
        object_list_instance, domain_file
    )
    atoms_schema_json_str = Atoms_schema_chatGPT.schema_json(indent=2)
    # Now that we have objects, we re-check the structured predicates for them
    predicates_structured_updated = get_predicates_structured(object_list_instance, domain_file)
    prompt_get_atoms_curr = get_all_grounded_atoms_prompt(
        predicates_structured_updated, object_list_instance, atoms_schema_json_str
    )
    message_get_atoms_curr = make_messages_image(
        prompt_get_atoms_curr, image_base64_curr
    )
    messages_get_atoms = [*example_history_atoms, *message_get_atoms_curr]

    response_format_atoms = Atoms_schema_chatGPT
    
    messages_get_atoms_together = [
        message[0] if isinstance(message, list) else message
        for message in messages_get_atoms
    ]

    print(f"Making API call for Atoms with ID: {request_id}")
    response_atoms, grounded_atoms_instance = get_structured_output_parsed(
        messages=messages_get_atoms_together,
        schema_response=response_format_atoms,
    )
    # Save full response to file
    if grounded_atoms_instance is not None:
        full_atoms_resp = {
            "request_id": request_id,
            "grounded_predicates": grounded_atoms_instance.dict().get("grounded_predicates", []),
            "domain_file": domain_file
        }
        with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_ATOMS_FILE), 'a') as f_atoms_full:
            json.dump(full_atoms_resp, f_atoms_full)
            f_atoms_full.write('\n')
            
    atom_correct_instance_curr = Atoms_schema_correct(**json.loads(atoms_json_curr))

    # Evaluate atoms
    if grounded_atoms_instance is not None:
        answer_atoms = {
            "grounded_predicates": grounded_atoms_instance.dict().get("grounded_predicates", [])
        }
        ground_truth_atoms = atom_correct_instance_curr.dict()
        eval_result_atoms = evaluate_predicates(answer_atoms, ground_truth_atoms)
        eval_result_atoms.update({"request_id": request_id})
        evaluation_results_atoms.append(eval_result_atoms)
    else:
        print(f"Failed to parse Atoms response for ID: {request_id}")

    ### MODIFICATION FOR GOALS START
    # Step: Get Goals (if available)
    # If goal_instruction_path and goal_predicates are present, we do the same as atoms:
    if goal_instruction_path and goal_predicates:
        # Build a goal-specific schema with is_negated support (only for goals)
        Goals_schema_correct = get_atom_schema_from_chatgpt_objects(
            object_correct_instance_curr, domain_file, include_negation=True
        )
        Goals_schema_chatGPT = get_atom_schema_from_chatgpt_objects(
            object_list_instance, domain_file, include_negation=True
        )
        # Parse ground truth using the negation-aware goal schema
        goal_correct_instance_curr = Goals_schema_correct(**json.loads(goal_predicates))
        goals_schema_json_str = Goals_schema_chatGPT.schema_json(indent=2)

        # Read the instruction text
        with open(goal_instruction_path, 'r') as f:
            instruction_text = f.read()

        prompt_get_goals_curr = get_all_goal_predicates_prompt(instruction_text, goals_schema_json_str)
        
        # Combine with example history for goals
        # We have example_history_goals if there were previous examples with goals
        messages_get_goals = [*example_history_goals, *make_message_text(prompt_get_goals_curr)]

        messages_get_goals_together = [
            message[0] if isinstance(message, list) else message
            for message in messages_get_goals
        ]

        print(f"Making API call for Goals with ID: {request_id}")
        response_goals, goals_instance = get_structured_output_parsed(
            messages=messages_get_goals_together,
            schema_response=Goals_schema_chatGPT,  # goal-specific schema with is_negated
        )

        
        
        if goal_instruction_path and goal_predicates and goals_instance is not None:
            full_goals_resp = {
                "request_id": request_id,
                "goal_predicates": goals_instance.dict().get("grounded_predicates", []),
                "domain_file": domain_file
            }
            with open(os.path.join(OUTPUT_FOLDER, FULL_RESPONSES_GOALS_FILE), 'a') as f_goals_full:
                json.dump(full_goals_resp, f_goals_full)
                f_goals_full.write('\n')
        
        
        
        
        if goals_instance is not None:
            answer_goals = {
                "grounded_predicates": goals_instance.dict().get("grounded_predicates", [])
            }
            ground_truth_goals = goal_correct_instance_curr.dict()
            eval_result_goals = evaluate_predicates(answer_goals, ground_truth_goals)
            eval_result_goals.update({"request_id": request_id})
            evaluation_results_goals.append(eval_result_goals)
        else:
            print(f"Failed to parse Goals response for ID: {request_id}")
    else:
        print(f"No goal information present for ID: {request_id}, skipping goal step.")
    ### MODIFICATION FOR GOALS END

    if VERBOSE:
        print("\nProcessed:", request_id)

def create_sequential_calls_from_dataset_for_domain(
    dataset: Dict[str, Any],
    dataset_path: str,
    domain_name: str = "all",
    num_examples: int = 1,
    num_tests: int = 10,
    output_folder: str = None,
    output_filename_objects: str = "objects_evaluation_results.jsonl",
    output_filename_atoms: str = "atoms_evaluation_results.jsonl",
    ### MODIFICATION FOR GOALS START
    output_filename_goals: str = "goals_evaluation_results.jsonl",
    ### MODIFICATION FOR GOALS END
    api_keys: List[str] = None,
):
    """
    Processes the dataset and creates sequential API calls for the specified domain.
    """
    if output_folder is None:
        output_folder = OUTPUT_FOLDER
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize evaluation result lists
    evaluation_results_objects = []
    evaluation_results_atoms = []
    ### MODIFICATION FOR GOALS START
    evaluation_results_goals = []
    ### MODIFICATION FOR GOALS END

    # Collect all domains and their states
    domains = {}
    for problem_num, problem in enumerate(dataset["problems"]):
        problem_domain = problem["problem_name"]
        if domain_name == "all" or problem_domain == domain_name:
            if problem_domain not in domains:
                domains[problem_domain] = []
            for state_num in range(len(problem["states"])):
                domains[problem_domain].append((problem_num, state_num))

    # Calculate number of samples per domain
    num_domains = len(domains)
    samples_per_domain = num_tests // num_domains if num_domains > 0 else 0
    remainder = num_tests % num_domains if num_domains > 0 else 0

    sampled_states = []

    for i, (domain, state_list) in enumerate(domains.items()):
        num_samples = samples_per_domain
        if i < remainder:
            num_samples += 1  # Distribute the remainder

        # Adjust if domain has fewer states than needed
        num_samples = min(num_samples, len(state_list))

        sampled_states.extend(random.sample(state_list, num_samples))

    # Shuffle the sampled states to mix them
    random.shuffle(sampled_states)
    print("List of sampled states: ", sampled_states)
    print("Number of actual samples (Lower due to equal distribution over domains): ", len(sampled_states))

    id_counter = 0

    # Open output files in append mode
    evaluation_output_file_objects = os.path.join(output_folder, output_filename_objects)
    evaluation_output_file_atoms = os.path.join(output_folder, output_filename_atoms)
    ### MODIFICATION FOR GOALS START
    evaluation_output_file_goals = os.path.join(output_folder, output_filename_goals)
    ### MODIFICATION FOR GOALS END

    # Initialize API keys
    if api_keys is None or not api_keys:
        api_keys = [OPENAI_API_KEY]

    api_key_index = 0
    ### MODIFICATION FOR  is adding the fgoals thing

    with open(evaluation_output_file_objects, 'a') as f_objects, \
         open(evaluation_output_file_atoms, 'a') as f_atoms, \
         open(evaluation_output_file_goals, 'a') as f_goals:
        for problem_id, state_num in sampled_states:
            # Assign API key in round-robin fashion (if multiple keys are used)
            api_key = api_keys[api_key_index % len(api_keys)]
            api_key_index += 1

            # Process sample
            process_sample(
                dataset,
                dataset_path,
                problem_id,
                state_num,
                num_examples,
                id_counter,
                evaluation_results_objects,
                evaluation_results_atoms,
                ### MODIFICATION FOR GOALS START
                evaluation_results_goals,
                ### MODIFICATION FOR GOALS END
                api_key=api_key,
            )
            id_counter += 1
            # Write evaluation results to files
            for eval_result in evaluation_results_objects:
                json.dump(eval_result, f_objects)
                f_objects.write('\n')
            f_objects.flush()
            evaluation_results_objects.clear()

            for eval_result in evaluation_results_atoms:
                json.dump(eval_result, f_atoms)
                f_atoms.write('\n')
            f_atoms.flush()
            evaluation_results_atoms.clear()

            ### MODIFICATION FOR GOALS START
            for eval_result in evaluation_results_goals:
                json.dump(eval_result, f_goals)
                f_goals.write('\n')
            f_goals.flush()
            evaluation_results_goals.clear()
            ### MODIFICATION FOR GOALS END

            # Sleep for a small amount to avoid overlapping requests
            time.sleep(0.01)

    # Calculate and print metrics
    evaluation_results_objects_all = []
    evaluation_results_atoms_all = []
    ### MODIFICATION FOR GOALS START
    evaluation_results_goals_all = []
    ### MODIFICATION FOR GOALS END

    ### MODIFICATION FOR GOALS in f_goals

    with open(evaluation_output_file_objects, 'r') as f_objects, \
         open(evaluation_output_file_atoms, 'r') as f_atoms, \
         open(evaluation_output_file_goals, 'r') as f_goals:
        for line in f_objects:
            evaluation_results_objects_all.append(json.loads(line))
        for line in f_atoms:
            evaluation_results_atoms_all.append(json.loads(line))
        ### MODIFICATION FOR GOALS START
        for line in f_goals:
            evaluation_results_goals_all.append(json.loads(line))
        ### MODIFICATION FOR GOALS END

    metrics_objects = calculate_metrics_objects(evaluation_results_objects_all)
    print("Objects Evaluation Metrics:")
    print(metrics_objects)

    metrics_atoms = calculate_metrics(evaluation_results_atoms_all)
    print("Atoms Evaluation Metrics:")
    print(metrics_atoms)

    ### MODIFICATION FOR GOALS START
    metrics_goals = calculate_metrics(evaluation_results_goals_all)
    print("Goals Evaluation Metrics:")
    print(metrics_goals)
    ### MODIFICATION FOR GOALS END

def get_all_objects_prompt(list_types: List[str], schema_json: str) -> str:
    """Generates a prompt to get all objects from the image."""
    return (
        f"Analyze the image and identify objects that match the following types: {list_types}. "
        "Name consistently. Keep the names for same objects. USE THE SAME NAMES AS YOU DID BEFORE."
    )

def get_all_grounded_atoms_prompt(predicates_structured: Dict[str, Any], object_list: Any, schema_json: str) -> str:
    """Generates a prompt to get all grounded atoms from the image."""
    grounded_objects = [o.dict() for o in object_list.objects]
    allowed_predicates = list(predicates_structured.keys())
    prompt_get_atoms_curr = (
        "Infer the current symbolic state from the image.\n"
        f"Grounded objects (use these exact symbols, no renaming): {grounded_objects}\n"
        f"Allowed predicate schemas: {allowed_predicates}\n"
        "Rules:\n"
        "1) Return the complete set of TRUE grounded predicates in the current state.\n"
        "2) Use only the provided grounded objects and allowed predicate schemas.\n"
        "3) Do not invent objects, predicates, aliases, or abbreviations.\n"
        "4) Do not include duplicates or contradictory facts.\n"
        "5) Be exhaustive: include all true relations, not only the most salient ones.\n"
        "Return only JSON matching the required schema."
    )
    return prompt_get_atoms_curr
    

### MODIFICATION FOR GOALS START
def get_all_goal_predicates_prompt(instruction_text: str, schema_json: str) -> str:
    """Generates a prompt to get all goal grounded predicates from the given instruction text."""
    return (
        "Given the following instructions:\n"
        f"{instruction_text}\n"
        "Identify the goal states as a set of grounded predicates that would match the described goal, and return them in the given schema."
    )
### MODIFICATION FOR GOALS END

def make_messages_text(prompt: str) -> List[Dict[str, Any]]:
    """Creates a message containing a prompt for OpenAI."""
    return [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
### PRediction functions

# Function to get random examples from the dataset
def get_random_examples_prediction(dataset: Dict[str, Any], problem_num: int, num_examples: int) -> List[tuple]:
    """Gets random examples from the dataset.

    Uses a per-problem local RNG seeded from (_GLOBAL_SEED, problem_num) so
    the selected examples are always identical for a given problem number,
    regardless of the order in which problems are processed.
    """
    # Local RNG: seeded per-problem so results are order-independent.
    rng = random.Random(f"{_GLOBAL_SEED}_{problem_num}")  # str seed: Python 3.12+ compatible
    chain_examples = []
    while len(chain_examples) < num_examples:
        random_problem_number = rng.randint(0, len(dataset["problems"]) - 1)
        if random_problem_number != problem_num:
            #ERROR - The indexes are not matching form the datset fdomain wise
            if (
                dataset["problems"][problem_num]["domain_file"]
                == dataset["problems"][random_problem_number]["domain_file"]
            ):
                random_triplet_number = rng.randint(
                    0, len(dataset["problems"][random_problem_number]["triplets"]) - 1
                )
                chain_examples.append((random_problem_number, random_triplet_number))

    return chain_examples








# Function to generate example messages from a given triplet
def get_message_example_from_triplet(
    domain_file: str,
    triplet: Dict[str, Any],
) -> tuple:
    """Generates example messages from a given triplet."""
    # Prepare data for current state
    objects_json_curr = triplet["state"]["all_objects"]
    atoms_json_curr = triplet["state"]["atoms"]
    action = triplet["action"]

    # Prepare data for next state
    objects_json_next = triplet["next_state"]["all_objects"]
    atoms_json_next = triplet["next_state"]["atoms"]

    # Get object schema
    object_schema_curr, list_types_curr = get_object_schema_from_pddl(domain_file)
    object_list_curr = object_schema_curr(**json.loads(objects_json_curr))
    object_schema_json_str = object_schema_curr.schema_json(indent=2)

    # Get predicates structured
    predicates_structured_curr = get_predicates_structured(object_list_curr, domain_file)
    Atoms_schema = get_atom_schema_from_chatgpt_objects(object_list_curr, domain_file)
    atoms_schema_json_str = Atoms_schema.schema_json(indent=2)
    
    #make the atom object
    atoms_curr = Atoms_schema(**json.loads(atoms_json_curr))

    # Generate prompts
    prompt_predict_next_state = get_predict_next_state_prompt(
        list_types_curr,
        object_schema_json_str,
        predicates_structured_curr,
        object_list_curr,
        atoms_schema_json_str,
        action,
        atoms_curr
    )

    message_predict_next_state = make_messages_text(prompt_predict_next_state)
    answer_next_state = make_answer_text(atoms_json_next)

    return [message_predict_next_state, answer_next_state]

def get_example_data_prediction(dataset: Dict[str, Any], problem_num: int, triplet_num: int) -> Dict[str, Any]:
    """Retrieves example data from the dataset."""
    problem = dataset["problems"][problem_num]
    triplet = problem["triplets"][triplet_num]
    return triplet

# Function to create example history for few-shot learning
def get_example_history_prediction(dataset, domain_file, chain_examples):
    """Creates an example history for few-shot learning."""
    example_history = []
    for problem_num, triplet_num in chain_examples:
        triplet = get_example_data_prediction(dataset, problem_num, triplet_num)
        example_message = get_message_example_from_triplet(
            domain_file, triplet
        )
        example_history.append(example_message)
    return example_history

# Function to load and create example messages from the dataset
def load_and_create_example_from_dataset_prediction(dataset_path: str, problem_num: int, num_examples: int) -> List[Dict[str, Any]]:
    """Loads and creates example messages from the dataset."""
    print("Loading dataset from:", dataset_path)
    dataset = load_dataset(dataset_path)
    print("Dataset loaded successfully.")
    domain_file = dataset["problems"][problem_num]["domain_file"]
    chain_examples = get_random_examples_prediction(dataset, problem_num, num_examples)
    print("Random examples selected")
    example_history = get_example_history_prediction(dataset, domain_file, chain_examples)
    example_history_final = [
        item for sublist in example_history for item in sublist
    ]
    print("Example history created")
    return example_history_final
    
# Function to generate prompts for predicting the next state
def get_predict_next_state_prompt(
    list_types: List[str],
    object_schema_json: str,
    predicates_structured: Dict[str, Any],
    object_list: Any,
    atoms_schema_json: str,
    action: str,
    atoms_curr: Any
) -> str:
    """Generates a prompt to predict the next state based on the current state and action."""
    return (
        f"You are given the current state's grounded predicates: {atoms_curr}, "
        f"and an action to be performed: {action}. "
        "Predict the next state's grounded predicates after applying the action. "
        f"Use the following predicates: {predicates_structured}, and objects: {object_list.objects}. "
        f"Return the predicted next state's grounded predicates."
    )
    
    
    
    
    
    
def reload_config(config_path: str):
    """
    Reload configuration from a specified config file.
    Updates all global configuration variables.
    """
    global config, DATASET_PATH, NUM_EXAMPLES, NUM_SAMPLES, DOMAIN_NAME
    global OUTPUT_FOLDER, VERBOSE, MAX_TOKENS, DETAIL_LEVEL_LOW
    global FULL_RESPONSES_OBJECTS_FILE, FULL_RESPONSES_ATOMS_FILE, FULL_RESPONSES_GOALS_FILE
    
    # Load new config
    config = load_config(config_path)
    
    # Reload environment from new config
    load_env(config['env_file'])
    
    # Update all configuration parameters
    MAX_TOKENS = config['max_tokens']
    DETAIL_LEVEL_LOW = config['detail_level_low']
    DATASET_PATH = config['dataset_path']
    NUM_EXAMPLES = config['num_examples']
    NUM_SAMPLES = config['num_tests']
    DOMAIN_NAME = config['domain_name']
    OUTPUT_FOLDER = config['output_folder']
    VERBOSE = config['verbose']
    
    FULL_RESPONSES_OBJECTS_FILE = config["objects_full_responses_path"]
    FULL_RESPONSES_ATOMS_FILE = config["atoms_full_responses_path"]
    FULL_RESPONSES_GOALS_FILE = config["goals_full_responses_path"]
    
    # Reinitialize the API client with new credentials
    reinitialize_client()
    
    print(f"Configuration reloaded from: {config_path}")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Model: {os.getenv('MODEL_NAME', 'unknown')}")


# Main execution block
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VLM evaluation on ViLaIn/PDDLGYM datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file (default: ./config.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible few-shot example selection (default: 42)"
    )
    args = parser.parse_args()

    # Set the module-level seed used by get_random_examples / get_random_examples_prediction.
    # Each function creates a local random.Random((_GLOBAL_SEED, problem_num)) so the
    # examples for a given problem are always identical, regardless of run order.
    import sys as _sys
    _sys.modules[__name__].__dict__['_GLOBAL_SEED'] = args.seed

    # Reload config if a different config file was specified
    if args.config != "./config.yaml":
        reload_config(args.config)
    
    # Load dataset
    dataset = load_dataset(DATASET_PATH)

    # List of API keys (OpenAI)
    api_keys = [OPENAI_API_KEY]

    create_sequential_calls_from_dataset_for_domain(
        dataset=dataset,
        dataset_path=DATASET_PATH,
        domain_name=DOMAIN_NAME,
        num_examples=NUM_EXAMPLES,
        num_tests=NUM_SAMPLES,
        output_folder=OUTPUT_FOLDER,
        api_keys=api_keys,
    )
