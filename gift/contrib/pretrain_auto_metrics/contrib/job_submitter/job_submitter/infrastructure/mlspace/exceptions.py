class MlSpaceError(RuntimeError):
    pass


class MlSpaceConfigError(MlSpaceError):
    pass


class MlSpaceApiError(MlSpaceError):
    pass


class MlSpaceJobNotFoundError(MlSpaceApiError):
    pass
