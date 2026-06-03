# -*- coding: utf-8 -*-
"""
GroqProvider - Groq API implementation using Llama 3.2 Vision.

This provider uses httpx to call the Groq API, providing a high-quota 
alternative to Gemini for image-based content generation.
"""
import base64
import json
from pathlib import Path
from typing import List, Type, TypeVar, Any

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from hcaptcha_challenger.agent.logger import LoggerHelper

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class GroqProvider:
    """
    Groq-based chat provider implementation.
    """

    def __init__(self, api_key: str | List[str], model: str | List[str] = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        """
        Initialize the Groq provider.

        Args:
            api_key: Groq API key or list of keys.
            model: Model name or list of model names to use.
        """
        if isinstance(api_key, str):
            self._api_keys = [api_key]
        else:
            self._api_keys = api_key
        
        if isinstance(model, str):
            self._models = [model]
        else:
            self._models = model
        
        self._key_index = 0
        self._model_index = 0
        self._response_data: dict | None = None

    @property
    def model(self) -> str:
        """Get the current active model name."""
        return self._models[self._model_index]

    def rotate_key(self):
        """Rotate to the next API key in the list. If all keys used, rotate model."""
        if len(self._api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            LoggerHelper.log_info(f"Rotacionando chave API Groq. Novo índice: {self._key_index}", emoji='refresh')
            
            # If we wrapped around to the first key, rotate the model too
            if self._key_index == 0 and len(self._models) > 1:
                self.rotate_model()
        elif len(self._models) > 1:
            # Only one key, but multiple models - rotate model
            self.rotate_model()

    def rotate_model(self):
        """Rotate to the next model in the list."""
        if len(self._models) > 1:
            self._model_index = (self._model_index + 1) % len(self._models)
            LoggerHelper.log_info(f"Rotacionando modelo Groq. Novo modelo: {self.model}", emoji='refresh')

    @property
    def api_key(self) -> str:
        return self._api_keys[self._key_index]

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @retry(
        stop=stop_after_attempt(15),  # 3 cycles of (keys * models)
        wait=wait_fixed(5),
        before_sleep=lambda retry_state: LoggerHelper.log_provider_error(
            retry_state.attempt_number, 15, retry_state.outcome.exception()
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs using Groq API.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        content = []
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})
        
        for img_path in images:
            if img_path.exists():
                base64_image = self._encode_image(img_path)
                # Groq supports data URLs for images
                mime_type = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })

        messages = []
        if description:
            messages.append({"role": "system", "content": description})
        
        messages.append({"role": "user", "content": content})

        # Groq requires that if response_format is json_object, the prompt must contain "JSON"
        # We also inject the schema to ensure the model uses the correct keys
        schema_instruction = ""
        if response_schema and issubclass(response_schema, BaseModel):
            try:
                schema = response_schema.model_json_schema()
                schema_instruction = f" You must respond with a JSON object matching this schema: {json.dumps(schema)}"
            except Exception as e:
                logger.warning(f"Failed to generate JSON schema: {e}")

        if "JSON" not in (description or "") and "JSON" not in (user_prompt or ""):
            messages.append({"role": "user", "content": f"Please respond in JSON format.{schema_instruction}"})
        elif schema_instruction:
             messages.append({"role": "user", "content": f"Ensure the response follows this JSON schema: {schema_instruction}"})

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 429:
                LoggerHelper.log_warning(f"Quota da API Groq Esgotada (429) para o índice de chave {self._key_index}", emoji='skull')
                self.rotate_key()
                response.raise_for_status()
            
            if response.status_code >= 400:
                LoggerHelper.log_error(f"Erro na API Groq ({response.status_code}): {response.text}")
                if response.status_code == 400:
                    LoggerHelper.log_error("Verifique se o modelo suporta visão ou se o prompt é válido.")
                response.raise_for_status()
            self._response_data = response.json()

        # Extract content
        result_text = self._response_data["choices"][0]["message"]["content"]
        try:
            result_json = json.loads(result_text)
            
            # Normalize coordinates for ImageBinaryChallenge (box_2d format)
            # The model sometimes returns ["01"] instead of [0, 1]
            if "coordinates" in result_json and isinstance(result_json["coordinates"], list):
                for coord in result_json["coordinates"]:
                    if "box_2d" in coord and isinstance(coord["box_2d"], list):
                        normalized = []
                        for val in coord["box_2d"]:
                            if isinstance(val, str):
                                # Convert "01" to [0, 1] or "12" to [1, 2]
                                if len(val) == 2 and val.isdigit():
                                    normalized.extend([int(val[0]), int(val[1])])
                                else:
                                    normalized.append(int(val) if val.isdigit() else val)
                            else:
                                normalized.append(val)
                        coord["box_2d"] = normalized[:2] if len(normalized) >= 2 else normalized
            
            return response_schema(**result_json)
        except Exception as e:
            LoggerHelper.log_error(f"Falha ao analisar resposta da Groq como JSON: {result_text}")
            raise e

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response_data:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            LoggerHelper.log_warning(f"Falha ao salvar cache de resposta: {e}")
