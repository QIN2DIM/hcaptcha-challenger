import io
import json
import os
import re
from pathlib import Path
from typing import List
from typing import Union, Literal

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field

THINKING_PROMPT = """
Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as json.

Follow the following format to return a coordinates wrapped with a json code block:
```json
{
  "challenge_prompt": "please click on the largest animal",
  "coordinates": [
    {"box_2d": [0,0]},
    {"box_2d": [1,2]},
    {"box_2d": [2,1]}
  ]
}
```
"""


class BoundingBoxCoordinate(BaseModel):
    box_2d: List[int] = Field(
        description="It can only be in planar coordinate format, e.g. [0,2] for the 3rd element in the first row"
    )


class ImageBinaryChallenge(BaseModel):
    challenge_prompt: str
    coordinates: List[BoundingBoxCoordinate]

    def convert_box_to_boolean_matrix(self) -> List[bool]:
        """
        将坐标列表转换为一维布尔矩阵。

        将 3x3 矩阵中的坐标转换为一维布尔列表，其中：
        - [0,0] 对应索引 0
        - [0,1] 对应索引 1
        - ...
        - [2,2] 对应索引 8

        Returns:
            List[bool]: 长度为9的布尔列表，坐标位置为True，其他位置为False
        """
        # 初始化一个长度为9的布尔列表，全部为False
        result = [False] * 9

        # 遍历所有坐标
        for coord in self.coordinates:
            # 获取二维坐标
            row, col = coord.box_2d

            # 验证坐标是否在有效范围内
            if 0 <= row < 3 and 0 <= col < 3:
                # 将二维坐标转换为一维索引: index = row * 3 + col
                index = row * 3 + col
                # 将对应位置设置为True
                result[index] = True

        return result

    @property
    def log_message(self) -> str:
        _coordinates = [i.box_2d for i in self.coordinates]
        bundle = {"Challenge Prompt": self.challenge_prompt, "Coordinates": str(_coordinates)}
        return json.dumps(bundle, indent=2, ensure_ascii=False)


def extract_json_blocks(text: str) -> List[str]:
    """
    从文本中提取被 ```json 和 ``` 包围的 JSON 代码块内容。

    Args:
        text: 包含可能的 JSON 代码块的字符串

    Returns:
        提取出的 JSON 内容列表，不包含 ```json 和 ``` 标记

    Examples:
        >>> text = "这是一些文本 ```json\\n{\\n  \\"name\\": \\"John\\"\\n}\\n``` 后面还有内容"
        >>> extract_json_blocks(text)
        ['{\n  "name": "John"\n}']
    """
    try:
        # 使用正则表达式匹配 ```json 和 ``` 之间的内容
        pattern = r"```json\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)

        # 如果没有找到匹配项，记录警告并返回空列表
        if not matches:
            logger.warning("No JSON code blocks found in the provided text")
            return []

        return matches
    except Exception as e:
        logger.error(f"Error extracting JSON blocks: {str(e)}")
        return []


def extract_first_json_block(text: str) -> dict:
    """
    从文本中提取第一个被 ```json 和 ``` 包围的 JSON 代码块内容。

    Args:
        text: 包含可能的 JSON 代码块的字符串

    Returns:
        提取出的第一个 JSON 内容，如果没有找到则返回 None
    """
    if blocks := extract_json_blocks(text):
        return json.loads(blocks[0])


class GeminiImageClassifier:
    def __init__(self, gemini_api_key: str):
        self._api_key = gemini_api_key

    def invoke(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike, io.IOBase],
        model: Literal[
            "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash-thinking-exp-01-21"
        ] = "gemini-2.0-flash-thinking-exp-01-21",
    ) -> ImageBinaryChallenge:
        client = genai.Client(api_key=self._api_key)

        files = [client.files.upload(file=challenge_screenshot)]

        # NOT support JSON response schema
        if model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type)
                    ],
                )
            ]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=THINKING_PROMPT
                ),
            )
            return ImageBinaryChallenge(**extract_first_json_block(response.text))

        # Use JSON response schema
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                    types.Part.from_text(
                        text="""Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as json."""
                    ),
                ],
            )
        ]
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ImageBinaryChallenge,
            ),
        )

        return ImageBinaryChallenge(**response.parsed.model_dump())
