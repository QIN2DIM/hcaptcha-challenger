import json
from pathlib import Path
import statistics
import collections
import numpy as np  # 导入 numpy 用于计算百分位数
import math  # 导入 math 用于向上取整

# --- 开始修改后的代码 ---

MIN_SAMPLES_FOR_OUTLIER_DETECTION = 5  # 定义进行离群点分析所需的最少样本数

root = Path("tmp")
all_token_counts = collections.defaultdict(list)

# 1. 数据收集：收集每个类型的所有 token 数量
print("Step 1: Collecting token counts from all files...")
for file in root.rglob("*model_answer.json"):
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        challenge_type = file.parent.parent.parent.parent.name
        thoughts_token_count = data.get("usage_metadata", {}).get("thoughts_token_count")

        if not thoughts_token_count:
            continue

        all_token_counts[challenge_type].append(thoughts_token_count)

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        print(f"Warning: Skipping file {file} due to an error: {e}")
        continue
print(f"Step 1: Finished. Found data for {len(all_token_counts)} types.")
print("-" * 30)


# 2. 对每个类型进行分析、清洗和统计
print("Step 2: Analyzing data, removing outliers, and calculating stats...")
final_analysis = {}

for challenge_type, counts_list in all_token_counts.items():
    original_count = len(counts_list)

    # 如果样本太少，不进行离群点分析
    if original_count < MIN_SAMPLES_FOR_OUTLIER_DETECTION:
        print(
            f"  - Type '{challenge_type}': Too few samples ({original_count}) for outlier detection. Using basic stats."
        )
        cleaned_counts = counts_list
        outliers_count = 0
        # 预算建议使用原始数据的最大值
        suggested_budget = max(cleaned_counts) if cleaned_counts else 0
        analysis_note = f"Not enough data for outlier detection (min required: {MIN_SAMPLES_FOR_OUTLIER_DETECTION})."

    else:
        # 2a. 使用 IQR 方法识别离群点
        q1 = np.percentile(counts_list, 25)
        q3 = np.percentile(counts_list, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 2b. 过滤数据，得到清洗后的列表
        cleaned_counts = [x for x in counts_list if lower_bound <= x <= upper_bound]
        outliers_count = original_count - len(cleaned_counts)

        # 2c. 设定 think_budget
        # 使用上边界作为预算，并向上取整。这是一个能覆盖绝大多数正常情况的稳健值。
        suggested_budget = math.ceil(upper_bound)
        analysis_note = f"Outliers identified using IQR method (lower: {lower_bound:.2f}, upper: {upper_bound:.2f})."

    # 3. 基于清洗后的数据计算统计值
    if not cleaned_counts:
        # 如果所有数据点都被视为离群点或列表为空
        stats = {'average': 0, 'median': 0, 'min': 0, 'max': 0}
    else:
        stats = {
            'average': round(statistics.mean(cleaned_counts), 2),
            'median': statistics.median(cleaned_counts),
            'min': min(cleaned_counts),
            'max': max(cleaned_counts),
        }

    # 4. 存储最终分析结果
    final_analysis[challenge_type] = {
        'analysis_note': analysis_note,
        'suggested_think_budget': suggested_budget,
        'original_data_info': {'sample_count': original_count},
        'outlier_info': {
            'outliers_removed': outliers_count,
            'percentage': (
                f"{round((outliers_count / original_count) * 100, 2)}%"
                if original_count > 0
                else "0%"
            ),
        },
        'cleaned_data_stats': {
            'sample_count': len(cleaned_counts),
            'average_tokens': stats['average'],
            'median_tokens': stats['median'],
            'min_tokens': stats['min'],
            'max_tokens': stats['max'],
        },
    }

print("Step 2: Finished analysis.")
print("-" * 30)

# 5. 格式化输出最终结果
print("Final Analysis Report:")
print(json.dumps(final_analysis, indent=4))
