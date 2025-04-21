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


def match_user_prompt(job_type: ChallengeTypeEnum, challenge_prompt: str) -> str:
    try:
        match job_type:
            case ChallengeTypeEnum.IMAGE_DRAG_SINGLE:
                return f"JobType: {job_type.value}"
            case ChallengeTypeEnum.IMAGE_DRAG_MULTI:
                if "pairs" in challenge_prompt:
                    return IMAGE_DRAG_MULTI_COMPLETE_THE_PAIRS.strip()
            case _:
                return ""
    except Exception as e:
        logger.warning(f"Error matching user prompt: {str(e)}")

    return ""
