# -*- coding: utf-8 -*-
# Time       : 2023/9/14 18:46
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(slots=True)
class QADataGenerator:
    project_dir: Path = Path(__file__).parent.parent
    qa_data_path: Path = Path("qa_data.txt")
    prompts: List[str] = field(default_factory=list)

    def _unicode_dedup(self):
        result = []
        str_dict = {}

        for prompt in self.prompts:
            utf8_str = prompt.encode("utf-8")
            if utf8_str not in str_dict:
                str_dict[utf8_str] = prompt
                result.append(prompt)

        return result

    def _load_cache(self):
        if not self.project_dir.exists():
            return

        for root, _, files in os.walk(self.project_dir):
            for file in files:
                if not (file.endswith(".json") and file.startswith("image_label")):
                    continue
                json_path = Path(root, file)
                data = json.loads(json_path.read_text(encoding="utf8"))
                rq = data.get("requester_question", {})
                if "en" not in rq:
                    continue
                question = rq["en"]
                self.prompts.append(question)

    def _merge_qa_data(self):
        if self.qa_data_path.exists():
            self.prompts.extend(self.qa_data_path.read_text(encoding="utf8").split("\n"))
        prompts = self._unicode_dedup()
        self.qa_data_path.write_text("\n".join(prompts), encoding="utf8")

    def run(self):
        self._load_cache()
        self._merge_qa_data()


if __name__ == "__main__":
    gen = QADataGenerator()
    gen.run()
