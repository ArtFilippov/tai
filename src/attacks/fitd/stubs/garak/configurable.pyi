"""Configuration hook inherited by the real probe and detector hierarchies."""

class Configurable:
    def _load_config(self, config_root: object = ...) -> None: ...
