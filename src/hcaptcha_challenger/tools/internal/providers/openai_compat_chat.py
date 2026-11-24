# -*- coding: utf-8 -*-
"""
OpenAI-compatible multimodal provider backed by the /v1/chat/completions API.

Unlike OpenAICompatProvider (Responses API), this provider works with any
endpoint exposing the classic OpenAI Chat Completions API with vision support
(image_url content parts), e.g. self-hosted LLM routers, vLLM, OpenRouter.
"""

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Type, TypeVar, cast

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)

_MIME_OVERRIDES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed or "image/png"


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


class OpenAIChatProvider:
    """
    OpenAI-compatible chat provider implementation (Chat Completions API).

    Works with any endpoint that implements the OpenAI Chat Completions API
    with vision support (image_url content parts).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        json_mode: bool = True,
        extra_headers: dict[str, str] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._json_mode = json_mode
        self._extra_headers = extra_headers or {}
        self._last_raw: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    @property
    def last_response(self) -> str | None:
        """Get the last raw text response for debugging/caching purposes."""
        return self._last_raw

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        mime = _guess_mime(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        stripped = (text or "").strip()
        if not stripped:
            return None
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        try:
            data = extract_first_json_block(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        # last resort: first {...} span in the text
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(stripped[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _normalize_for_schema(data: dict[str, Any], schema: Type[ResponseT]) -> dict[str, Any]:
        """Best-effort alias normalization for models from non-Gemini providers."""
        schema_name = getattr(schema, "__name__", "")
        if schema_name == "ChallengeRouterResult":
            challenge_type = (
                data.get("challenge_type")
                or data.get("type")
                or data.get("label")
                or data.get("answer")
                or data.get("classification")
                or data.get("result")
            )
            challenge_prompt = (
                data.get("challenge_prompt")
                or data.get("prompt")
                or data.get("question")
                or data.get("description")
                or data.get("text")
                or "unknown"
            )
            data = {
                "challenge_prompt": challenge_prompt,
                "challenge_type": challenge_type,
            }
            return data
        if schema_name == "ImageDragDropChallenge":
            paths = data.get("paths")
            if paths is None:
                moves = data.get("moves")
                if isinstance(moves, list) and moves:
                    normalized_moves = []
                    for move in moves:
                        if not isinstance(move, dict):
                            continue
                        source = move.get("from") or move.get("source") or move.get("start")
                        target = move.get("to") or move.get("target") or move.get("end")
                        if isinstance(source, (list, tuple)) and len(source) >= 2:
                            source = {"x": source[0], "y": source[1]}
                        if isinstance(target, (list, tuple)) and len(target) >= 2:
                            target = {"x": target[0], "y": target[1]}
                        if isinstance(source, dict) and isinstance(target, dict):
                            normalized_moves.append({"start_point": source, "end_point": target})
                    if normalized_moves:
                        paths = normalized_moves
            if paths is None:
                source = data.get("source") or data.get("start") or data.get("from")
                target = data.get("target") or data.get("end") or data.get("to")
                if isinstance(source, dict) and "source" in source:
                    source = source["source"]
                if isinstance(target, dict) and "target" in target:
                    target = target["target"]
                if isinstance(source, (list, tuple)) and len(source) >= 2:
                    source = {"x": source[0], "y": source[1]}
                if isinstance(target, (list, tuple)) and len(target) >= 2:
                    target = {"x": target[0], "y": target[1]}
                if isinstance(source, dict) and isinstance(target, dict):
                    paths = [{"start_point": source, "end_point": target}]
            if paths is not None:
                data = dict(data)
                data["paths"] = paths
                data.setdefault("challenge_prompt", "unknown")
        elif schema_name == "ImageAreaSelectChallenge":
            points = data.get("points")
            if points is None:
                raw_points = data.get("coordinates")
                if raw_points is None:
                    single = data.get("coordinate") or data.get("point") or data.get("answer")
                    raw_points = [single] if single else []
                normalized_points = []
                for item in raw_points:
                    coordinate = None
                    if isinstance(item, dict):
                        if "x" in item and "y" in item:
                            normalized_points.append({"x": item["x"], "y": item["y"]})
                            continue
                        coordinate = item.get("coordinate") or item.get("coord")
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        coordinate = item
                    elif isinstance(item, str):
                        coordinate = item
                    if isinstance(coordinate, str):
                        parts = [p for p in coordinate.replace(",", " ").split() if p]
                        if len(parts) >= 2:
                            normalized_points.append({"x": int(parts[0]), "y": int(parts[1])})
                    elif isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
                        normalized_points.append({"x": int(coordinate[0]), "y": int(coordinate[1])})
                points = normalized_points
            if points is not None:
                data = dict(data)
                data["points"] = points
                data.setdefault("challenge_prompt", "unknown")
        return data

    @classmethod
    def _validate_with_schema(cls, data: dict[str, Any], schema: Type[ResponseT]) -> ResponseT:
        normalized = cls._normalize_for_schema(data, schema)
        try:
            return schema.model_validate(normalized)
        except Exception:
            if normalized is not data:
                return schema.model_validate(data)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: list[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        valid_images = [Path(image) for image in images if image and Path(image).exists()]
        if not valid_images:
            raise ValueError("At least one valid image is required")

        content: list[dict[str, Any]] = []
        for image_path in valid_images:
            content.append(
                {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}}
            )
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})

        messages: list[dict[str, Any]] = []
        if description:
            messages.append({"role": "system", "content": description})
        messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {"model": self._model, "messages": messages}
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code >= 400 and "response_format" in payload:
                # Some OpenAI-compatible endpoints reject response_format;
                # retry once without it before giving up to tenacity.
                logger.warning(
                    f"Endpoint rejected response_format (HTTP {response.status_code}), "
                    f"retrying without it"
                )
                payload.pop("response_format", None)
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
            response.raise_for_status()
            raw_body = response.text
            # Some OpenAI-compatible endpoints return SSE-flavored bodies
            # (content-type text/event-stream) even for non-streaming
            # requests: a single JSON object with a trailing `data: [DONE]`
            # sentinel instead of a bare JSON document.
            if raw_body.lstrip().startswith("data:"):
                data_payloads = []
                for line in raw_body.splitlines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        data_payloads.append(json.loads(chunk))
                    except json.JSONDecodeError:
                        continue
                if data_payloads:
                    # Non-streaming SSE: single payload. Streaming SSE:
                    # merge the delta chunks into one synthetic body.
                    if len(data_payloads) == 1:
                        body = cast(dict[str, Any], data_payloads[0])
                    else:
                        merged_content_parts: list[str] = []
                        merged_reasoning_parts: list[str] = []
                        for payload_item in data_payloads:
                            for choice in payload_item.get("choices") or []:
                                delta = choice.get("delta") or {}
                                content_piece = delta.get("content")
                                if isinstance(content_piece, str):
                                    merged_content_parts.append(content_piece)
                                reasoning_piece = delta.get("reasoning_content")
                                if isinstance(reasoning_piece, str):
                                    merged_reasoning_parts.append(reasoning_piece)
                        body = cast(
                            dict[str, Any],
                            {
                                "choices": [
                                    {
                                        "message": {
                                            "content": "".join(merged_content_parts),
                                            "reasoning_content": "".join(merged_reasoning_parts),
                                        }
                                    }
                                ]
                            },
                        )
                else:
                    raise ValueError(f"Failed to parse SSE response: {raw_body[:500]}")
            elif raw_body.rstrip().endswith("data: [DONE]"):
                # Single SSE-flavored JSON object (no data: prefix):
                # strip the trailing `data: [DONE]` sentinel and parse directly.
                body = cast(dict[str, Any], json.loads(raw_body[: raw_body.rfind("data: [DONE]")]))
            else:
                body = cast(dict[str, Any], json.loads(raw_body))

        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"Failed to parse response: {body}")
        raw_content = (choices[0].get("message") or {}).get("content")
        self._last_raw = raw_content if isinstance(raw_content, str) else ""

        json_payload = self._extract_json(self._last_raw)
        if json_payload is not None:
            parsed = self._validate_with_schema(json_payload, response_schema)
            # Chat models sometimes answer with an empty result instead of
            # admitting they cannot see the images; treat that as a failure so
            # tenacity retries with a fresh challenge.
            dump = parsed.model_dump() if hasattr(parsed, "model_dump") else {}
            if dump.get("points") == [] or dump.get("paths") == [] or dump.get("coordinates") == []:
                raise ValueError(f"Empty answer from model, retrying: {self._last_raw[:120]}")
            return parsed

        raise ValueError(f"Failed to parse structured response: {self._last_raw}")

    def cache_response(self, path: Path) -> None:
        if not self._last_raw:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._last_raw, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to cache response: {exc}")
