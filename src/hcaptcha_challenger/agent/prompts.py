from loguru import logger

from hcaptcha_challenger.tools.challenge_classifier import ChallengeTypeEnum

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
step1. 仔细观察并记住右侧面板中可拖拽图案元素的形状特征。
step2. 审视左侧画布上的所有图案。
step3. 在左侧画布上找出与右侧图案元素形状最为相似的那个图案。
step4. 将右侧面板的图案元素拖拽并精确地覆盖到左侧画布上找到的最相似图案之上。
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
            case _:
                return ""
    except Exception as e:
        logger.warning(f"Error matching user prompt: {str(e)}")

    return ""
