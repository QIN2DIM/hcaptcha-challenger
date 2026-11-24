# -*- coding: utf-8 -*-
"""
OpenAI-compatible multimodal provider backed by the /v1/responses API.
"""
import base64
import copy
import json
import mimetypes
from pathlib import Path
from typing import Any, Type, TypeVar, cast

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(normalized)
    return normalized


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._response: dict[str, Any] | None = None

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str | None:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        return text
        return None

    @staticmethod
    def _extract_json_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        text = OpenAICompatProvider._extract_output_text(payload)
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "json_schema":
                    data = content.get("json")
                    if isinstance(data, dict):
                        return data
        return None

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

        input_content: list[dict[str, Any]] = []
        if description:
            input_content.append({"type": "input_text", "text": description})
        if user_prompt:
            input_content.append({"type": "input_text", "text": user_prompt})
        for image_path in valid_images:
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_to_data_url(image_path),
                }
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "input": [{"type": "message", "role": "user", "content": input_content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "schema": ensure_strict_json_schema(response_schema.model_json_schema()),
                    "strict": True,
                }
            },
        }

        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/responses",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            self._response = cast(dict[str, Any], response.json())

        json_payload = self._extract_json_payload(self._response)
        if json_payload is not None:
            return response_schema.model_validate(json_payload)

        raise ValueError(f"Failed to parse structured response: {self._response}")

    def cache_response(self, path: Path) -> None:
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._response, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to cache response: {exc}")
