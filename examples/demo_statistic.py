"""
This script analyzes the 'thoughts_token_count' from JSON files
within a specified directory structure. It performs the following steps:

1.  Data Collection: Recursively finds '*model_answer.json' files, extracts
    the challenge type and token count, and groups them by type.
2.  Analysis & Cleaning: For each challenge type, it uses the IQR (Interquartile Range)
    method to detect and remove outliers, provided there are enough data points.
    If not, it performs a basic analysis without outlier removal.
3.  Statistics Calculation: It calculates key statistics (mean, median, min, max)
    on the cleaned data.
4.  Budget Suggestion: It suggests a 'think_budget' based on the analysis. For
    data with outliers removed, this is the upper bound of the normal range.
    For smaller datasets, it's the maximum observed value.
5.  Reporting: It presents the final analysis in a clean, formatted table
    using the 'rich' library, with a fixed width for consistent output.
"""

import collections
import json
import math
import statistics
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# --- Configuration ---
# The minimum number of samples required to perform outlier detection.
MIN_SAMPLES_FOR_OUTLIER_DETECTION = 5
# The root directory to search for JSON files.
ROOT_DIRECTORY = Path("tmp")

# --- Initialization ---
# CRITICAL FIX: Force a console width to prevent bad rendering in terminals
# that misreport their size (e.g., some IDEs, CI/CD runners).
console = Console(width=140)
all_token_counts = collections.defaultdict(list)

# --- Step 1: Data Collection ---
console.print(
    Panel(
        "[bold cyan]Step 1: Collecting Token Counts[/bold cyan]\n"
        "Scanning for 'model_answer.json' files...",
        expand=False,
    )
)

files_to_process = list(ROOT_DIRECTORY.rglob("*model_answer.json"))

with Progress(
    "[progress.description]{task.description}",
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• [progress.completed]{task.completed} of {task.total} files",
    console=console,
) as progress:
    task = progress.add_task("[green]Processing files...", total=len(files_to_process))

    for file in files_to_process:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            challenge_type = file.parent.parent.parent.parent.name
            thoughts_token_count = data.get("usage_metadata", {}).get("thoughts_token_count")

            if thoughts_token_count is None:
                continue

            all_token_counts[challenge_type].append(thoughts_token_count)

        except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
            console.print(f"[yellow]Warning:[/yellow] Skipping file {file} due to an error: {e}")
            continue
        finally:
            progress.update(task, advance=1)

console.print(
    f"[green]✔[/green] Step 1 finished. Found data for {len(all_token_counts)} challenge types.\n"
)


# --- Step 2: Analysis, Cleaning, and Statistics ---
console.print(
    Panel(
        "[bold cyan]Step 2: Analyzing Data & Calculating Statistics[/bold cyan]\n"
        "Removing outliers and computing metrics for each type...",
        expand=False,
    )
)
final_analysis = {}

for challenge_type, counts_list in sorted(all_token_counts.items()):
    original_count = len(counts_list)

    if original_count < MIN_SAMPLES_FOR_OUTLIER_DETECTION:
        cleaned_counts = counts_list
        outliers_count = 0
        suggested_budget = max(cleaned_counts) if cleaned_counts else 0
        analysis_note = f"Basic stats (samples < {MIN_SAMPLES_FOR_OUTLIER_DETECTION})"
    else:
        q1 = np.percentile(counts_list, 25)
        q3 = np.percentile(counts_list, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned_counts = [x for x in counts_list if lower_bound <= x <= upper_bound]
        outliers_count = original_count - len(cleaned_counts)
        suggested_budget = math.ceil(upper_bound)
        analysis_note = f"IQR Method (Upper: {upper_bound:.1f})"

    if not cleaned_counts:
        stats = {'average': 0, 'median': 0, 'min': 0, 'max': 0}
    else:
        stats = {
            'average': statistics.mean(cleaned_counts),
            'median': statistics.median(cleaned_counts),
            'min': min(cleaned_counts),
            'max': max(cleaned_counts),
        }

    final_analysis[challenge_type] = {
        'analysis_note': analysis_note,
        'suggested_think_budget': suggested_budget,
        'original_sample_count': original_count,
        'outliers_removed': outliers_count,
        'outlier_percentage': (
            (outliers_count / original_count) * 100 if original_count > 0 else 0
        ),
        'cleaned_sample_count': len(cleaned_counts),
        'stats': stats,
    }

console.print(f"[green]✔[/green] Step 2 finished. Analysis complete.\n")


# --- Step 3: Formatted Output ---
console.print(
    Panel(
        "[bold cyan]Final Analysis Report[/bold cyan]", title="[bold]Summary[/bold]", expand=False
    )
)

table = Table(
    title="Agent 'Thoughts' Token Usage Analysis",
    show_header=True,
    header_style="bold magenta",
    width=138,  # Use slightly less than console width for padding
)

# Define table columns with specific widths for better layout
table.add_column("Challenge Type", style="cyan", no_wrap=True, min_width=25)
table.add_column("Suggested Budget", justify="right", style="bold green", min_width=10)
table.add_column("Total Samples", justify="right", min_width=8)
table.add_column("Outliers Removed", justify="right", min_width=18)
table.add_column("Cleaned Samples", justify="right", min_width=8)
table.add_column("Avg Tokens", justify="right", min_width=10)
table.add_column("Median Tokens", justify="right", min_width=10)
table.add_column("Min Tokens", justify="right", min_width=10)
table.add_column("Max Tokens", justify="right", min_width=10)
table.add_column("Analysis Notes", style="dim", min_width=20)

for challenge_type, analysis_data in final_analysis.items():
    stats = analysis_data['stats']
    table.add_row(
        challenge_type,
        f"{analysis_data['suggested_think_budget']:,}",
        f"{analysis_data['original_sample_count']}",
        f"{analysis_data['outliers_removed']} ({analysis_data['outlier_percentage']:.1f}%)",
        f"{analysis_data['cleaned_sample_count']}",
        f"{stats['average']:.1f}",
        f"{stats['median']:,}",
        f"{stats['min']:,}",
        f"{stats['max']:,}",
        analysis_data['analysis_note'],
    )

console.print(table)

console.print(
    Panel(
        "[bold]How to Interpret 'Suggested Budget':[/bold]\n"
        "• [bold]IQR Method[/bold]: The budget is the upper boundary for non-outlier data (Q3 + 1.5 * IQR), rounded up. It's designed to accommodate all typical cases while excluding extreme anomalies.\n"
        "• [bold]Basic Stats[/bold]: When the sample size is too small for outlier detection, the budget is simply the maximum token count observed.",
        title="Methodology Notes",
        border_style="dim",
        expand=False,
    )
)
