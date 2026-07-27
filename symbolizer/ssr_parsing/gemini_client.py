"""
Gemini Vision API client for SSR pipeline.

Uses the new google-genai SDK (replaces deprecated google-generativeai).

Supported models:
  - gemini-3-pro-preview      (production)
  - gemini-3-flash-preview    (development / fast iteration)
  - gemini-3-flash-preview          (stable fallback)

Usage:
    Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file
    Set MODEL_TO_USE=gemini
    Set MODEL_NAME to the desired model (e.g. gemini-3-pro-preview)
"""

import os
import json
import base64
import logging
import time
import re
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, TypeVar, Set
from PIL import Image
from PIL import ImageFile
import io

from pydantic import BaseModel

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- New SDK: google-genai ----
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    genai_types = None
    logger.warning("google-genai not installed. Gemini support disabled. Install with: pip install google-genai")

T = TypeVar('T', bound=BaseModel)

# Default model (latest stable)
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


class GeminiClient:
    """
    Client for Google Gemini Vision API using the new google-genai SDK.

    Supports structured JSON output via Pydantic schemas, vision requests
    with inline images, and is a drop-in replacement for OpenAI/Mistral
    in the SSR pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        vertexai: Optional[bool] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.project = project or os.getenv("PROJECT")
        self.location = location or os.getenv("LOCATION")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
        raw_fallback = os.getenv("GEMINI_VERTEX_FALLBACK_LOCATIONS", "")
        self._vertex_fallback_locations = [x.strip() for x in raw_fallback.split(",") if x.strip()]
        if self.location and self.location not in self._vertex_fallback_locations:
            self._vertex_fallback_locations.insert(0, self.location)
        # Safety default: preview Gemini models often resolve via global while
        # regional endpoints can return model-not-found for the same model ID.
        if "global" not in self._vertex_fallback_locations:
            self._vertex_fallback_locations.append("global")
        if self.location in self._vertex_fallback_locations:
            self._vertex_location_index = self._vertex_fallback_locations.index(self.location)
        else:
            self._vertex_location_index = 0
        self._unsupported_vertex_locations: Set[str] = set()

        # Default: always use vertexai=True with API key — this enables
        # logprobs on all models and avoids per-model daily quota limits
        # that affect the plain genai.Client(api_key=...) path.
        env_vertex = os.getenv("GEMINI_USE_VERTEXAI", "true").lower() != "false"
        if vertexai is not None:
            self.vertexai = vertexai
        else:
            self.vertexai = env_vertex

        self.model_name = model_name
        preferred_region = self._preferred_region_for_model(self.model_name)
        force_preferred = os.getenv("GEMINI_FORCE_MODEL_PREFERRED_REGION", "true").strip().lower() == "true"
        if preferred_region:
            if preferred_region in self._vertex_fallback_locations:
                self._vertex_fallback_locations = [
                    preferred_region,
                    *[r for r in self._vertex_fallback_locations if r != preferred_region],
                ]
            else:
                self._vertex_fallback_locations.insert(0, preferred_region)
            if force_preferred:
                self.location = preferred_region
            if self.location in self._vertex_fallback_locations:
                self._vertex_location_index = self._vertex_fallback_locations.index(self.location)
            else:
                self._vertex_location_index = 0
        self._region_cache_path = Path(
            os.getenv("GEMINI_VERTEX_REGION_CACHE_PATH", "/tmp/gemini_vertex_region_cache.json")
        )
        self._context_cache_enabled = os.getenv("GEMINI_CONTEXT_CACHE_ENABLE", "false").strip().lower() == "true"
        self._context_cache_ttl_seconds = int(os.getenv("GEMINI_CONTEXT_CACHE_TTL_SECONDS", "86400") or 86400)
        self._context_cache_variants = max(1, int(os.getenv("GEMINI_CONTEXT_CACHE_VARIANTS", "1") or 1))
        self._context_cache_mode = os.getenv("GEMINI_CONTEXT_CACHE_MODE", "random").strip().lower()
        self._context_cache_index_path = Path(
            os.getenv("GEMINI_CONTEXT_CACHE_INDEX_PATH", "/tmp/gemini_context_cache_index.json")
        )
        cached_region = self._get_cached_model_region(self.model_name)
        if cached_region and cached_region in self._vertex_fallback_locations:
            self.location = cached_region
            self._vertex_location_index = self._vertex_fallback_locations.index(cached_region)
            logger.info(
                "GeminiClient using cached region '%s' for model '%s'",
                cached_region,
                self.model_name,
            )

        # Create the centralized Client object (new SDK pattern)
        if self.vertexai and self.project and self.location:
            # Service-account Vertex AI (no API key)
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
            logger.info(f"GeminiClient using Vertex AI project={self.project} location={self.location}")
            print(f"[gemini_client] mode=vertex_project_location project={self.project} location={self.location}")
        elif self.api_key and self.vertexai:
            # vertexai=True + api_key: routes through Vertex AI, enables
            # logprobs and avoids restrictive per-model quotas.
            self.client = genai.Client(vertexai=True, api_key=self.api_key)
            logger.info(f"GeminiClient using Vertex AI with API key")
            print("[gemini_client] mode=vertex_api_key")
        elif self.api_key:
            # Plain API key mode (logprobs may not work on all models)
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"GeminiClient using plain API key (logprobs may be limited)")
            print("[gemini_client] mode=api_key_direct")
        else:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY/GEMINI_API_KEY, "
                "or set PROJECT/LOCATION for service-account Vertex AI."
            )

        logger.info(f"GeminiClient initialized with model: {self.model_name}")

    def _load_context_cache_index(self) -> Dict[str, Any]:
        if not self._context_cache_index_path.exists():
            return {}
        try:
            data = json.loads(self._context_cache_index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to load context cache index %s: %s", self._context_cache_index_path, exc)
        return {}

    def _save_context_cache_index(self, data: Dict[str, Any]) -> None:
        try:
            self._context_cache_index_path.parent.mkdir(parents=True, exist_ok=True)
            self._context_cache_index_path.write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to save context cache index %s: %s", self._context_cache_index_path, exc)

    def _messages_to_parts(self, messages: List[Dict[str, Any]]) -> List[Any]:
        parts: List[Any] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(genai_types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(genai_types.Part.from_text(text=item["text"]))
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64data = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            parts.append(
                                genai_types.Part.from_bytes(
                                    data=base64.standard_b64decode(b64data),
                                    mime_type=mime,
                                )
                            )
        return parts

    def _select_cache_variant(self, prefix_messages: List[Dict[str, Any]]) -> int:
        if self._context_cache_variants <= 1:
            return 0
        fixed = os.getenv("GEMINI_CONTEXT_CACHE_VARIANT_FIXED", "").strip()
        if fixed.isdigit():
            return max(0, min(self._context_cache_variants - 1, int(fixed)))
        if self._context_cache_mode == "stable":
            key = json.dumps(prefix_messages, sort_keys=True, default=str).encode("utf-8")
            return int(hashlib.md5(key).hexdigest(), 16) % self._context_cache_variants
        return random.randint(0, self._context_cache_variants - 1)

    def _context_cache_key(self, prefix_messages: List[Dict[str, Any]], variant: int) -> str:
        payload = {
            "model": self.model_name,
            "project": self.project or "",
            "location": self.location or "",
            "variant": int(variant),
            "prefix_messages": prefix_messages,
        }
        dig = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return f"ctx::{dig}"

    def _get_or_create_context_cache_name(
        self,
        *,
        prefix_messages: List[Dict[str, Any]],
        variant: int,
    ) -> Optional[str]:
        if not self._context_cache_enabled:
            return None
        if len(prefix_messages) == 0:
            return None
        if not self.vertexai:
            return None

        key = self._context_cache_key(prefix_messages, variant)
        now_ts = int(time.time())
        idx = self._load_context_cache_index()
        rec = idx.get(key) if isinstance(idx, dict) else None
        if isinstance(rec, dict):
            name = str(rec.get("name", ""))
            exp = int(rec.get("expires_at", 0) or 0)
            if name and exp > now_ts:
                return name

        prefix_parts = self._messages_to_parts(prefix_messages)
        if not prefix_parts:
            return None
        display_name = f"phase2_ctx_v{variant}_{self.model_name[:32]}"
        ttl = f"{max(60, int(self._context_cache_ttl_seconds))}s"
        created = self.client.caches.create(
            model=self.model_name,
            config=genai_types.CreateCachedContentConfig(
                display_name=display_name,
                contents=prefix_parts,
                ttl=ttl,
            ),
        )
        name = str(getattr(created, "name", "") or "")
        if not name:
            return None
        idx[key] = {
            "name": name,
            "variant": int(variant),
            "model": self.model_name,
            "location": self.location or "",
            "created_at": now_ts,
            "expires_at": now_ts + int(self._context_cache_ttl_seconds),
        }
        self._save_context_cache_index(idx)
        return name

    @staticmethod
    def _preferred_region_for_model(model_name: str) -> Optional[str]:
        model = (model_name or "").strip()
        mapping = {
            "gemini-3-preview-flash": "global",
            "gemini-3-flash-preview": "global",
            "gemini-3.1-flash-lite-preview": "global",
            "gemini-2.5-flash-lite": "us-central1",
        }
        return mapping.get(model)

    @staticmethod
    def _allowed_regions_for_model(model_name: str) -> Optional[Set[str]]:
        model = (model_name or "").strip()
        mapping = {
            "gemini-3-preview-flash": {"global"},
            "gemini-3-flash-preview": {"global"},
            "gemini-3.1-flash-lite-preview": {"global"},
        }
        return mapping.get(model)

    def _load_region_cache(self) -> Dict[str, str]:
        if not self._region_cache_path.exists():
            return {}
        try:
            data = json.loads(self._region_cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Failed to load Gemini region cache %s: %s", self._region_cache_path, exc)
        return {}

    def _save_region_cache(self, data: Dict[str, str]) -> None:
        try:
            self._region_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._region_cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save Gemini region cache %s: %s", self._region_cache_path, exc)

    def _get_cached_model_region(self, model_name: str) -> Optional[str]:
        return self._load_region_cache().get(model_name)

    def _remember_model_region(self) -> None:
        if not (self.vertexai and self.project and self.location and self.model_name):
            return
        cache = self._load_region_cache()
        if cache.get(self.model_name) == self.location:
            return
        cache[self.model_name] = self.location
        self._save_region_cache(cache)
        logger.info(
            "Recorded Gemini preferred region model=%s region=%s cache=%s",
            self.model_name,
            self.location,
            self._region_cache_path,
        )

    def _switch_to_next_vertex_location(self) -> Optional[str]:
        """Rotate to next configured Vertex location when capacity throttles."""
        if not (self.vertexai and self.project and self._vertex_fallback_locations):
            return None
        if len(self._vertex_fallback_locations) <= 1:
            return None
        allowed = self._allowed_regions_for_model(self.model_name)
        total = len(self._vertex_fallback_locations)
        for _ in range(total):
            self._vertex_location_index = (self._vertex_location_index + 1) % total
            candidate = self._vertex_fallback_locations[self._vertex_location_index]
            if allowed is not None and candidate not in allowed:
                continue
            if candidate not in self._unsupported_vertex_locations:
                new_location = candidate
                break
        else:
            logger.warning(
                "All configured Vertex fallback regions are marked unsupported: %s",
                sorted(self._unsupported_vertex_locations),
            )
            return None
        self.location = new_location
        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )
        logger.warning(
            "GeminiClient switched Vertex location to '%s' due to capacity throttling",
            new_location,
        )
        print(f"[gemini_client] failover location={new_location}")
        return new_location

    def _mark_current_location_unsupported(self) -> None:
        if self.location:
            self._unsupported_vertex_locations.add(self.location)
            logger.warning(
                "Marking Vertex location '%s' unsupported for model '%s' after NOT_FOUND",
                self.location,
                self.model_name,
            )

    def encode_image(self, image_path: str) -> Dict[str, Any]:
        """Encode an image file for the Gemini API inline_data format."""
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                with Image.open(image_path) as img:
                    img.load()
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')

                    format_to_mime = {
                        'JPEG': 'image/jpeg',
                        'PNG': 'image/png',
                        'GIF': 'image/gif',
                        'WEBP': 'image/webp',
                    }
                    buffer = io.BytesIO()
                    save_format = img.format or 'JPEG'
                    img.save(buffer, format=save_format)
                    image_bytes = buffer.getvalue()
                    mime_type = format_to_mime.get(save_format, 'image/jpeg')
                break
            except OSError as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.25)
                    continue
                raise
        if last_exc is not None and 'image_bytes' not in locals():
            raise last_exc

        return {
            "mime_type": mime_type,
            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
        }

    def encode_image_from_base64(self, base64_data: str, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Prepare a base64-encoded image for inline_data."""
        return {"mime_type": mime_type, "data": base64_data}

    def send_vision_request(
        self,
        prompt: str,
        image_paths: List[str],
        max_tokens: int = 32768,
    ) -> str:
        """Send a vision request with images and text, return raw JSON string."""
        parts = []
        for image_path in image_paths:
            image_data = self.encode_image(image_path)
            parts.append(genai_types.Part.from_bytes(
                data=base64.standard_b64decode(image_data["data"]),
                mime_type=image_data["mime_type"],
            ))
        parts.append(genai_types.Part.from_text(text=prompt))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        return response.text

    def send_structured_request(
        self,
        prompt: str,
        image_paths: List[str],
        schema: Type[T],
        max_tokens: int = 32768,
    ) -> tuple:
        """Send a vision request and parse response into a Pydantic model.

        Returns:
            Tuple of (parsed Pydantic model or None, raw response dict).
        """
        parts = []
        for image_path in image_paths:
            image_data = self.encode_image(image_path)
            parts.append(genai_types.Part.from_bytes(
                data=base64.standard_b64decode(image_data["data"]),
                mime_type=image_data["mime_type"],
            ))
        parts.append(genai_types.Part.from_text(text=prompt))

        try:
            response = self._call_with_retry(parts, schema, max_tokens)

            # The new SDK can parse directly into the schema
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed, json.loads(response.text) if response.text else {}

            # Fallback: parse manually
            response_dict = json.loads(response.text)
            parsed = schema.model_validate(response_dict)
            return parsed, response_dict

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None, {}
        except Exception as e:
            logger.error(f"Error in structured request: {e}")
            return None, {}

    def _call_with_retry(
        self,
        parts,
        schema,
        max_tokens,
        enable_logprobs: bool = False,
        logprobs_count: int = 5,
        cached_content_name: Optional[str] = None,
        max_retries: int = 5,
    ):
        """Call generate_content with exponential backoff for 429 rate limits."""
        for attempt in range(max_retries):
            try:
                cfg_kwargs = dict(
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                )
                if enable_logprobs:
                    cfg_kwargs["response_logprobs"] = True
                    cfg_kwargs["logprobs"] = max(1, int(logprobs_count))
                if cached_content_name:
                    cfg_kwargs["cached_content"] = cached_content_name
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=genai_types.GenerateContentConfig(**cfg_kwargs),
                )
            except Exception as e:
                # Preserve planner timeout semantics; do not convert into None result.
                if e.__class__.__name__ == "SearchTimeout" or str(e).strip().lower() == "search timed out":
                    raise
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    lower_err = err_str.lower()
                    hard_quota = (
                        "exceeded your current quota" in lower_err
                        or "billing details" in lower_err
                        or "insufficient quota" in lower_err
                    )
                    if hard_quota:
                        raise RuntimeError(
                            f"Hard quota exhaustion for model '{self.model_name}'. "
                            f"Skipping retries to avoid long waits. "
                            f"Check billing/quota and retry later. "
                            f"error message: {err_str}"
                        ) from e
                    if "per_day" in err_str:
                        wait = 30
                    else:
                        wait = min(5 * (2 ** attempt), 30)
                    switched = self._switch_to_next_vertex_location()
                    logger.warning(
                        "Rate limited (attempt %s/%s), waiting %ss. switched_location=%s Full error: %s",
                        attempt + 1,
                        max_retries,
                        wait,
                        switched,
                        err_str,
                    )
                    time.sleep(wait)
                elif enable_logprobs and ("Logprobs is not enabled" in err_str or "logprobs" in err_str.lower()):
                    raise RuntimeError(
                        f"Model '{self.model_name}' did not return logprobs. "
                        "Use a model that supports response_logprobs (for example gemini-2.5-pro / gemini-3-flash-preview on Vertex), "
                        "or disable USE_LOGPROBS."
                    ) from e
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    self._mark_current_location_unsupported()
                    switched = self._switch_to_next_vertex_location()
                    if switched is not None:
                        logger.warning(
                            "Model not found/access denied in current location (attempt %s/%s), switched_location=%s. error=%s",
                            attempt + 1,
                            max_retries,
                            switched,
                            err_str,
                        )
                        time.sleep(1)
                        continue
                    raise
                else:
                    raise
        # Final attempt without catching
        cfg_kwargs = dict(
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        if enable_logprobs:
            cfg_kwargs["response_logprobs"] = True
            cfg_kwargs["logprobs"] = max(1, int(logprobs_count))
        if cached_content_name:
            cfg_kwargs["cached_content"] = cached_content_name
        return self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )

    def send_structured_request_from_messages(
        self,
        messages: List[Dict[str, Any]],
        schema: Type[T],
        max_tokens: int = 4096,
        enable_logprobs: bool = False,
        logprobs_count: int = 5,
    ) -> tuple:
        """Send a structured request from OpenAI-style messages.

        Extracts text and images from the messages list and calls the Gemini API.
        This is the method called by run_eval_vlm.py's get_structured_output_parsed().

        Returns:
            Tuple of (parsed Pydantic model or None, raw response dict, raw_response_obj).
        """
        prefix_messages = messages[:-1] if len(messages) > 1 else []
        dynamic_messages = messages[-1:] if messages else []
        variant = self._select_cache_variant(prefix_messages)
        cached_content_name = self._get_or_create_context_cache_name(
            prefix_messages=prefix_messages,
            variant=variant,
        )
        parts = self._messages_to_parts(dynamic_messages if cached_content_name else messages)

        last_json_error: Optional[Exception] = None
        parse_retries = max(1, int(os.getenv("GEMINI_JSON_PARSE_RETRIES", "3") or 3))
        for parse_attempt in range(parse_retries):
            try:
                response = self._call_with_retry(
                    parts,
                    schema,
                    max_tokens,
                    enable_logprobs=enable_logprobs,
                    logprobs_count=logprobs_count,
                    cached_content_name=cached_content_name,
                )

                # Extract diagnostics from response if available
                logprobs_info = self._extract_logprobs(response)
                usage_info = self._extract_usage(response)

                response_text = self._extract_json_text(response)
                if hasattr(response, "parsed") and response.parsed is not None:
                    raw_dict: Dict[str, Any] = {}
                    if response_text:
                        raw_dict = json.loads(response_text)
                    raw_dict["_gemini_logprobs"] = logprobs_info
                    raw_dict["_gemini_usage"] = usage_info
                    raw_dict["_gemini_context_cache"] = {
                        "enabled": bool(self._context_cache_enabled),
                        "variant": int(variant),
                        "cached_content_name": cached_content_name or "",
                        "prefix_message_count": len(prefix_messages),
                        "dynamic_message_count": len(dynamic_messages),
                    }
                    self._remember_model_region()
                    return response.parsed, raw_dict, response

                response_dict = json.loads(response_text or "{}")
                response_dict["_gemini_logprobs"] = logprobs_info
                response_dict["_gemini_usage"] = usage_info
                response_dict["_gemini_context_cache"] = {
                    "enabled": bool(self._context_cache_enabled),
                    "variant": int(variant),
                    "cached_content_name": cached_content_name or "",
                    "prefix_message_count": len(prefix_messages),
                    "dynamic_message_count": len(dynamic_messages),
                }
                parsed = schema.model_validate(response_dict)
                self._remember_model_region()
                return parsed, response_dict, response

            except json.JSONDecodeError as e:
                last_json_error = e
                logger.error(f"Failed to parse Gemini JSON response: {e}")
                raise RuntimeError(f"Gemini JSON parse failed: {e}") from e
            except Exception as e:
                # Preserve planner timeout semantics; do not convert into None result.
                if e.__class__.__name__ == "SearchTimeout" or str(e).strip().lower() == "search timed out":
                    raise
                logger.error(f"Error in Gemini structured request: {e}")
                raise RuntimeError(f"Gemini structured request failed: {e}") from e

        if last_json_error is not None:
            logger.error(f"Failed to parse Gemini JSON response: {last_json_error}")
            raise RuntimeError(
                f"Gemini JSON parse failed after {parse_retries} attempts: {last_json_error}"
            ) from last_json_error
        raise RuntimeError("Gemini structured request returned no parseable response.")

    def _extract_json_text(self, response: Any) -> str:
        """Best-effort JSON text extraction from Gemini response."""
        text = getattr(response, "text", None) or ""
        text = text.strip()
        if text:
            return text

        # Fallback: concatenate text parts manually from the first candidate.
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return ""
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            chunks: List[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
            merged = "".join(chunks).strip()
            if merged:
                return merged
        except Exception:
            pass

        # Last fallback: extract candidate text from dict if present.
        try:
            raw = response.to_dict() if hasattr(response, "to_dict") else {}
            body = json.dumps(raw)
            m = re.search(r"\\{.*\\}", body)
            if m:
                return m.group(0)
        except Exception:
            pass
        return ""

    def _extract_logprobs(self, response) -> Dict[str, Any]:
        """Extract logprobs information from a Gemini response.
        
        Returns a dict with avg_logprobs and per-token logprobs if available.
        """
        info = {"available": False, "avg_logprobs": None, "tokens": []}
        if not response or not response.candidates:
            return info
        
        cand = response.candidates[0]
        if cand.avg_logprobs is not None:
            info["avg_logprobs"] = cand.avg_logprobs
            info["available"] = True
        
        if cand.logprobs_result is not None:
            info["available"] = True
            if cand.logprobs_result.chosen_candidates:
                for c in cand.logprobs_result.chosen_candidates:
                    info["tokens"].append({
                        "token": c.token,
                        "log_probability": c.log_probability,
                        "token_id": getattr(c, "token_id", None),
                    })
            if cand.logprobs_result.top_candidates:
                info["top_candidates"] = []
                for tc in cand.logprobs_result.top_candidates:
                    top_list = []
                    for tcc in tc.candidates:
                        top_list.append({
                            "token": tcc.token,
                            "log_probability": tcc.log_probability,
                        })
                    info["top_candidates"].append(top_list)
        
        return info

    def _extract_usage(self, response) -> Dict[str, Any]:
        """Extract token usage information from a Gemini response."""
        info = {
            "available": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_content_tokens": 0,
        }
        if response is None:
            return info
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return info
        info["prompt_tokens"] = int(
            getattr(um, "prompt_token_count", None)
            or getattr(um, "promptTokenCount", 0)
            or 0
        )
        info["completion_tokens"] = int(
            getattr(um, "candidates_token_count", None)
            or getattr(um, "candidatesTokenCount", 0)
            or 0
        )
        info["total_tokens"] = int(
            getattr(um, "total_token_count", None)
            or getattr(um, "totalTokenCount", 0)
            or 0
        )
        info["cached_content_tokens"] = int(
            getattr(um, "cached_content_token_count", None)
            or getattr(um, "cachedContentTokenCount", 0)
            or 0
        )
        info["available"] = True
        return info

    def get_objects_with_probabilities(
        self, parsed_objects: Any, raw_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """REMOVED: Use logprobs for real probabilities instead of fake defaults."""
        raise RuntimeError(
            "get_objects_with_probabilities is disabled \u2014 use logprobs for real "
            "probabilities. Do not assign fake default probabilities."
        )

    def get_predicates_with_probabilities(
        self, parsed_predicates: Any, raw_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """REMOVED: Use logprobs for real probabilities instead of fake defaults."""
        raise RuntimeError(
            "get_predicates_with_probabilities is disabled \u2014 use logprobs for real "
            "probabilities. Do not assign fake default probabilities."
        )


def create_gemini_client() -> Optional[GeminiClient]:
    """Factory: create a GeminiClient from environment variables."""
    if not GEMINI_AVAILABLE:
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    model_name = os.getenv("MODEL_NAME", DEFAULT_GEMINI_MODEL)
    project = os.getenv("PROJECT")
    location = os.getenv("LOCATION")

    if not api_key and not (project and location):
        logger.warning("No Gemini credentials: set GOOGLE_API_KEY or PROJECT+LOCATION.")
        return None

    try:
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            project=project,
            location=location,
        )
    except Exception as e:
        logger.error(f"Failed to create GeminiClient: {e}")
        return None


def is_gemini_available() -> bool:
    """Check if Gemini SDK is installed and an API key is configured."""
    if not GEMINI_AVAILABLE:
        return False
    has_api_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY"))
    has_vertex = bool(os.getenv("PROJECT") and os.getenv("LOCATION"))
    return has_api_key or has_vertex
