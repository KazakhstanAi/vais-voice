"""VAIS Voice research infrastructure."""


class GenerationNotImplementedError(NotImplementedError):
    pass


def generate() -> None:
    raise GenerationNotImplementedError("VAIS VoiceGen is not implemented in v0.1.")
