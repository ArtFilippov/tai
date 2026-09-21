"""Runtime validation at text-construction boundaries."""


def require_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Message text must be a string")
    return value


def require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value
