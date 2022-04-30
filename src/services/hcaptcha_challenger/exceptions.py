from typing import Optional, Sequence


class ArmorException(Exception):
    """Armor module basic exception"""

    def __init__(self, msg: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None):
        self.msg = msg
        self.stacktrace = stacktrace
        super().__init__()

    def __str__(self) -> str:
        exception_msg = f"Message: {self.msg}\n"
        if self.stacktrace:
            stacktrace = "\n".join(self.stacktrace)
            exception_msg += f"Stacktrace:\n{stacktrace}"
        return exception_msg


class ChallengeException(ArmorException):
    """hCAPTCHA Challenge basic exceptions"""


class ChallengeLangException(ChallengeException):
    """指定了不兼容的挑战语言"""


class ChallengePassed(ChallengeException):
    """挑战未弹出"""


class LoadImageTimeout(ChallengeException):
    """加载挑战图片超时"""


class ChallengeTimeout(ChallengeException):
    """人机挑战超时 CPU能力太弱无法在规定时间内完成挑战"""


class LabelNotFoundException(ChallengeException):
    """获取到空的图像标签名"""


class AssertTimeout(ChallengeTimeout):
    """断言超时"""
