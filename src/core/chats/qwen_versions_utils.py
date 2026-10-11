"""Optional package versions included in detached Qwen diagnostics."""

from importlib.metadata import PackageNotFoundError, version


def versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "garak"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
