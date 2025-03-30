import typing


class ArmorException(Exception):
    """Armor module basic exception"""

    def __init__(
        self,
        msg: typing.Optional[str] = None,
        stacktrace: typing.Optional[typing.Sequence[str]] = None,
    ):
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


class ChallengePassed(ChallengeException):
    """Challenge not popping up"""


class LoadImageTimeout(ChallengeException):
    """Loading challenge image timed out"""


class LabelNotFoundException(ChallengeException):
    """Get an empty image label name"""


class AuthException(ArmorException):
    """Thrown when there is a problem with authentication,
    such as encountering 2FA authentication inserted after hcaptcha"""


class AuthMFA(AuthException):
    """Authentication failed, 2FA is not supported"""


class LoginException(AuthException):
    """Authentication failed, account or password error"""


class AuthUnknownException(AuthException):
    def __init__(self, msg=None, stacktrace=None):
        super().__init__(msg, stacktrace)
        self.__doc__ = None

    def report(self, msg):
        self.__doc__ = msg
