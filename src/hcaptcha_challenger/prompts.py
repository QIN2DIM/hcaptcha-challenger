from loguru import logger

from hcaptcha_challenger.models import ChallengeTypeEnum

IMAGE_DRAG_MULTI_COMPLETE_THE_PAIRS = """
**游戏背景:"**
左侧画布上有若干组成对出现的图案，它们之间由一根线段连接。但有一组搭配中，只有一根线段和一枚图案，它没有成对的伙伴。
**任务:**
step1. 识别左侧画布上有哪个组合的图案没有成对出现
step2. 在右侧待选区选择对应的图案方块
step3. 将方块拖拽至左侧画布的正确位置上，使得该组图案可以成对出现。
"""

IMAGE_DRAG_SINGLE_MOST_SIMILAR = """
**游戏背景:**
左侧画布上展示了多个不同的图案。右侧面板则单独显示一个可拖拽的图案元素。
**任务:**
step1. 仔细观察并记住右侧面板中可拖拽图案元素的**整体轮廓、曲线特征，以及任何内部细节（例如：锯齿、凹陷、孔洞、尖角的数量和形状等）。**
step2. 审视左侧画布上的所有图案。
step3. 在左侧画布上找出与右侧图案元素在**整体形状和内部细节上均最为匹配**的那个图案。**注意区分那些仅在宏观轮廓上部分相似，但在关键细节（如内部结构、边缘平滑度或尖角特征）上存在明显差异的干扰项。**
step4. 将右侧面板的图案元素拖拽并精确地覆盖到左侧画布上找到的最相似图案之上。
"""

IMAGE_LABEL_MULTI_SELECT_SAME_NUMBER_OF_HOLES = """
**游戏背景:**
画布上展示了若干个多边形（通常为5个），每个多边形内有若干个 hole 图案。
**任务:**
step1. 锁定画布上所有多边形的位置。
step2. 根据“数觉”判断每个多边形内的 hole 数量，hole 数量较少的多边形内的 hole 数量通常是一致的。
step3. 选择 hole 数量一致的 1 组多边形，也即，之多点击两个点。
"""


def match_user_prompt(job_type: ChallengeTypeEnum, challenge_prompt: str) -> str:
    try:
        match job_type:
            case ChallengeTypeEnum.IMAGE_DRAG_SINGLE:
                if "similar" in challenge_prompt:
                    return IMAGE_DRAG_SINGLE_MOST_SIMILAR.strip()
                return f"JobType: {job_type.value}"
            case ChallengeTypeEnum.IMAGE_DRAG_MULTI:
                if "pairs" in challenge_prompt:
                    return IMAGE_DRAG_MULTI_COMPLETE_THE_PAIRS.strip()
            case ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT:
                return "If you answer correctly, I will reward you with a tip of $20."
            case ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT:
                if "holes" in challenge_prompt and "same number" in challenge_prompt:
                    return IMAGE_LABEL_MULTI_SELECT_SAME_NUMBER_OF_HOLES.strip()
                return "When multiple clickable objects appear on Canvas, you need to carefully distinguish whether all objects are clickable."
            case _:
                return ""
    except Exception as e:
        logger.warning(f"Error matching user prompt: {str(e)}")

    return ""
