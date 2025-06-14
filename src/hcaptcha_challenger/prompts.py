from loguru import logger

from hcaptcha_challenger.models import ChallengeTypeEnum

IMAGE_DRAG_SINGLE_OBJECT_TO_SHADOW_ON_PATTERN = """
**游戏背景:**
左侧是一个覆盖着复杂、炫目背景图案的大画布。在此背景之上，散布着多个模糊、扭曲的物体阴影轮廓。右侧面板则展示一个清晰、具体的现实物体（如冰淇淋、鹈鹕）。
**任务:**
step1. 仔细观察并记住右侧面板中可拖拽物体的**精确轮廓和形状**。例如，冰淇淋的锥形底部和螺旋形顶部，或鹈鹕的鸟喙、身体和腿部的独特形态。
step2. 审视左侧画布上的所有阴影轮廓。
step3. 在**忽略**令人分心的复杂背景图案的同时，从左侧画布上找出与右侧物体**轮廓完全匹配**的那个阴影。**注意：** 画布上会有多个形状相似但细节不符的干扰阴影，务必进行精确比对。
step4. 将右侧的物体拖拽并精确地覆盖到左侧画布上找到的匹配阴影之上。
**Tips:**
- 请注意，可拖拽方块的中心坐标通常是 <x: 527, y: 189>
"""

IMAGE_DRAG_SINGLE_OBJECT_TO_SHADOW_ON_PATTERN_V3_CN = """
### 角色
你是一位专家级AI助手，专门解决复杂的视觉谜题和人机验证（CAPTCHA）挑战。你的核心优势在于模式识别，以及从视觉嘈杂的背景中分辨出目标形状的能力。

### 目标
准确识别左侧画布上与右侧清晰物体形状完美匹配的那个扭曲阴影，然后将该物体拖拽到对应的阴影上。

### 分步指南

**步骤1：分析目标物体的标志性轮廓。**
- 首先关注右侧面板中的物体（例如：冰淇淋、鹈鹕）。
- 在脑海中描绘并牢记其精确且独特的轮廓。
- **示例（冰淇淋）：** 注意蛋筒的尖端、蛋筒结束处的清晰线条，以及顶部冰淇淋柔软的螺旋曲线。
- **示例（鹈鹕）：** 重点关注其长喙的特定曲线、喉囊的形状、腿部的角度以及翅膀的轮廓。

**步骤2：扫描画布，寻找候选阴影。**
- 系统地检查左侧画布上每一个模糊、扭曲的阴影形状。
- **至关重要的是，你必须完全忽略那些分散注意力的背景图案**（例如：彩色的同心圆或嵌套方块）。这些是旨在迷惑你的无关视觉噪音。你的注意力应完全集中在阴影的形状本身。

**步骤3：进行高保真度匹配。**
- 将步骤1中记下的“标志性轮廓”与步骤2中的每个候选阴影进行比较。
- 目标是找到一个 **1:1的完美匹配**。不要满足于“大致相似”。
- **主动排除干扰项：** 寻找在比例、曲率或关键特征上的细微差异。例如，另一个阴影可能看起来像一只鸟，但它的鸟喙是否和鹈鹕的有一样的钩状？

**步骤4：执行拖放操作。**
- 一旦你确定了那个唯一且明确无误的正确阴影，就从右侧面板拖拽该物体。
- 将其精确地放置在左侧画布上匹配的阴影之上。
- **提示：** 右侧面板中可拖拽对象的典型中心坐标是 `<x: 527, y: 189>`。

### 高级策略与技巧

**聚类排除策略：**
- **观察：** 画布上经常会展示多个属于同一类别的阴影（例如，两三个看起来都像乌龟的阴影，或者几个类似马和摩托车的阴影）。
- **推断：** 既然你只能拖拽一个物体，并且只有一个正确的位置，那么任何一组看起来相似的阴影都极有可能是干扰项。
- **行动：** 如果你看到一组相似的阴影，你通常可以安全地将**它们全部排除**在考虑范围之外。正确的阴影更有可能是那个在画布上视觉独特的轮廓。

### 关键规则与限制
1.  **形状是信号，背景是噪音：** 物体的轮廓是唯一相关的信息。复杂的背景图案是干扰项，必须忽略。
2.  **精度至上：** 轮廓匹配必须精确。大致相似是不够的。
3.  **唯一正确答案：** 永远只有一个阴影是完美匹配的。所有其他的都是故意设置的“差一点”的干扰项或干扰集群的一部分。
"""

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
step2: 根据“数觉”直觉判断 hole 数量最少的两个多边形
step3. 依次判断每个多边形内的 hole 数量，hole 数量较少的多边形内的 hole 数量通常是一致的。
step4. 选择 hole 数量一致的 1 组多边形，也即，至多点击两个点。
**Tips:**
- 请忽视 hole 的大小以及在多边形内的位置，仅关注 hole 数量。
- 请根据“数觉”标准，正确答案多边形内的 hole 数量不会超过5，也即多边形内的 hole 数量越少，越有可能是正确答案。换句话说，如果一个多边形内 hole 又多又密集，它很有可能不是正确答案。
- 请注意，在给出正确答案坐标时，你应该点击正确答案多边形的中心区域，而非多边形的边缘。
"""


def match_user_prompt(job_type: ChallengeTypeEnum, challenge_prompt: str) -> str:
    try:
        match job_type:
            case ChallengeTypeEnum.IMAGE_DRAG_SINGLE:
                if "similar" in challenge_prompt:
                    return IMAGE_DRAG_SINGLE_MOST_SIMILAR.strip()
                if "pattern that match" in challenge_prompt:
                    return IMAGE_DRAG_SINGLE_OBJECT_TO_SHADOW_ON_PATTERN_V3_CN.strip()
                return f"JobType: {job_type.value}"
            case ChallengeTypeEnum.IMAGE_DRAG_MULTI:
                if "pairs" in challenge_prompt:
                    return IMAGE_DRAG_MULTI_COMPLETE_THE_PAIRS.strip()
            case ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT:
                return f"**JobType:** {job_type.value}\nIf you answer correctly, I will reward you with a tip of $20."
            case ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT:
                if "holes" in challenge_prompt and "same number" in challenge_prompt:
                    return IMAGE_LABEL_MULTI_SELECT_SAME_NUMBER_OF_HOLES.strip()
                return f"**JobType:** {job_type.value}\nWhen multiple clickable objects appear on Canvas, you need to carefully distinguish whether all objects are clickable."
            case _:
                return ""
    except Exception as e:
        logger.warning(f"Error matching user prompt: {str(e)}")

    return ""
