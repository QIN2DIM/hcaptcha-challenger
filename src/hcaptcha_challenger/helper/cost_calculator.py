"""
https://ai.google.dev/gemini-api/docs/pricing
"""

import json
import pathlib
import re
from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import TypedDict, List, Dict, Union, Optional

from google.genai import types
from pydantic import BaseModel, Field

# Price per million tokens
UNIT_1_M = 0.000001
# Price per thousand tokens
UNIT_1_K = 0.001


class CostItem(TypedDict):
    model: str
    input_price: float
    output_price: float
    unit: float


# fmt:off
model_cost_list: List[CostItem] = [
    # == Paid Plan == #
    CostItem(model="gemini-2.5-pro-preview-03-25", input_price=1.25, output_price=10.0, unit=UNIT_1_M),
    CostItem(model="gemini-2.5-flash-preview-04-17", input_price=0.15, output_price=3.50, unit=UNIT_1_M),
    CostItem(model="gemini-2.0-flash", input_price=0.10, output_price=0.40, unit=UNIT_1_M),
    CostItem(model="gemini-2.0-flash-lite", input_price=0.075, output_price=0.30, unit=UNIT_1_M),
    # == Free Plan == #
    CostItem(model="gemini-2.5-pro-exp-03-25", input_price=0, output_price=0, unit=UNIT_1_M),
    CostItem(model="gemini-2.0-flash-thinking-exp-01-21", input_price=0, output_price=0, unit=UNIT_1_M),
]
# fmt:on

model_cost_mapping = {i['model']: i for i in model_cost_list}


class ModelUsageStats(BaseModel):
    """Statistical data model for model usage"""

    # fmt:off
    total_files: int = Field(default=0, description="Total number of model answer files")
    total_challenges: int = Field(default=0, description="Total number of unique challenges")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    average_cost_per_challenge: float = Field(default=0.0, description="Average cost per challenge in USD")
    median_cost_per_challenge: float = Field(default=0.0, description="Median cost per challenge in USD")
    start_time: Optional[datetime] = Field(default=None, description="First challenge timestamp")
    end_time: Optional[datetime] = Field(default=None, description="Last challenge timestamp")
    model_details: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Cost details by model")
    challenge_costs: List[float] = Field(default_factory=list, description="List of costs for each challenge")
    # fmt:on

    def save_to_json(self, file_path: Union[str, pathlib.Path]) -> None:
        """Save stats to a JSON file"""
        file_path = pathlib.Path(file_path)

        # Create a serializable dict
        data = self.model_dump()

        # Convert datetime objects to ISO format strings
        if data["start_time"]:
            data["start_time"] = data["start_time"].isoformat()
        if data["end_time"]:
            data["end_time"] = data["end_time"].isoformat()

        # Save to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Stats saved to {file_path}")


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename pattern"""
    # Typical pattern: challenge_1234567890123_xxx
    match = re.search(r'challenge_(\d{13})', str(filename))
    if match:
        timestamp_ms = int(match.group(1))
        return datetime.fromtimestamp(timestamp_ms / 1000)  # Convert from milliseconds to seconds
    return None


def calculate_model_cost(
    challenge_path: Union[str, pathlib.Path], detailed: bool = False
) -> Union[float, ModelUsageStats]:
    """
    Calculate the cost of model usage for all challenges in the specified path

    Args:
        challenge_path: Path to challenge data directory
        detailed: Whether to return detailed cost breakdown

    Returns:
        If detailed=False: Returns total cost as float
        If detailed=True: Returns ModelUsageStats object with detailed statistics
    """
    challenge_root = pathlib.Path(challenge_path)

    if not challenge_root.exists():
        raise FileNotFoundError(f"Specified path does not exist: {challenge_root}")

    # Initialize stats model
    stats = ModelUsageStats()

    # Track unique challenges by parent directory
    challenge_costs = defaultdict(float)
    challenge_files = defaultdict(list)

    # Process all model answer files
    for item_file in challenge_root.rglob("*_model_answer.json"):
        try:
            stats.total_files += 1

            # Extract timestamp for time range calculation
            timestamp = extract_timestamp_from_filename(item_file.parent.name)
            if timestamp:
                if stats.start_time is None or timestamp < stats.start_time:
                    stats.start_time = timestamp
                if stats.end_time is None or timestamp > stats.end_time:
                    stats.end_time = timestamp

            # Track this file under its parent challenge directory
            challenge_dir = str(item_file.parent)
            challenge_files[challenge_dir].append(item_file)

            record = types.GenerateContentResponse.model_validate_json(item_file.read_bytes())

            if record.model_version not in model_cost_mapping:
                continue

            cost_item = model_cost_mapping[record.model_version]

            # Calculate input cost
            input_tokens = record.usage_metadata.prompt_token_count
            input_cost = input_tokens * cost_item["input_price"] * cost_item["unit"]

            # Calculate output cost
            output_tokens = record.usage_metadata.candidates_token_count
            output_cost = output_tokens * cost_item["output_price"] * cost_item["unit"]

            # Calculate total cost for this item
            item_total_cost = input_cost + output_cost

            # Update global stats
            stats.total_input_tokens += input_tokens
            stats.total_output_tokens += output_tokens
            stats.total_cost += item_total_cost

            # Update challenge-specific cost
            challenge_costs[challenge_dir] += item_total_cost

            # Update model-specific stats if detailed reporting is requested
            if detailed:
                model_name = record.model_version
                if model_name not in stats.model_details:
                    stats.model_details[model_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_cost": 0,
                        "output_cost": 0,
                        "total_cost": 0,
                        "usage_count": 0,
                    }

                stats.model_details[model_name]["input_tokens"] += input_tokens
                stats.model_details[model_name]["output_tokens"] += output_tokens
                stats.model_details[model_name]["input_cost"] += input_cost
                stats.model_details[model_name]["output_cost"] += output_cost
                stats.model_details[model_name]["total_cost"] += item_total_cost
                stats.model_details[model_name]["usage_count"] += 1

        except Exception as e:
            print(f"Error processing file {item_file}: {e}")

    # Calculate challenge statistics
    stats.total_challenges = len(challenge_costs)
    stats.challenge_costs = list(challenge_costs.values())

    if stats.total_challenges > 0:
        stats.average_cost_per_challenge = stats.total_cost / stats.total_challenges
        stats.median_cost_per_challenge = (
            median(stats.challenge_costs) if stats.challenge_costs else 0
        )

    if detailed:
        # Add total summary
        stats.model_details["Total"] = {"total_cost": stats.total_cost}
        return stats

    return stats.total_cost


def export_stats(
    challenge_path: Union[str, pathlib.Path], output_file: Optional[Union[str, pathlib.Path]] = None
) -> ModelUsageStats:
    """
    Calculate and export detailed statistics for model usage

    Args:
        challenge_path: Path to challenge data directory
        output_file: Path to save JSON output (optional)

    Returns:
        ModelUsageStats object with complete statistics
    """
    stats = calculate_model_cost(challenge_path, detailed=True)

    if isinstance(stats, float):
        # This shouldn't happen as we specified detailed=True
        raise ValueError("Failed to generate detailed statistics")

    if output_file:
        stats.save_to_json(output_file)

    return stats
