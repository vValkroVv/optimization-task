class EndpointUnavailableError(BaseException):
    pass


class OutputTokenBudgetExceededError(BaseException):
    pass


class RLIExecProxyConnectionLostError(RuntimeError):
    pass


class RLILongRunningCommandLostError(RuntimeError):
    pass
