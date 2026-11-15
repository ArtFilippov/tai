"""Independent native garak loading/rendering reference without probe execution."""

import json
import subprocess
import sys


def reference(name: str) -> dict[str, str]:
    """Run pinned metaclass loading and rendering with the harness disconnected.

    Args:
        name (str): Target model name supplied to native rendering.

    Returns:
        dict[str, str]: All named prompts in native declaration order.
    """
    script = """
import json
import sys
from types import SimpleNamespace
from importlib.resources import files
from unittest.mock import patch
from garak.probes import Probe, dan

result = {}
with patch.object(dan, 'data_path', files('garak').joinpath('data')), patch.object(Probe, '__init__', return_value=None), patch.object(Probe, 'probe', return_value=[]):
    for name, cls in vars(dan).items():
        if name == 'Ablation_Dan_11_0':
            break
        if isinstance(cls, dan.DANProbeMeta):
            probe = cls()
            probe.probe(SimpleNamespace(name=sys.argv[1]))
            [text] = probe.prompts
            result[name] = text
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, name],
        check=True, capture_output=True, text=True,
    )
    content: dict[str, str] = json.loads(result.stdout)
    return content


def collection(variant: str, name: str) -> list[str]:
    """Load and render a complete pinned native collection in an isolated process.

    Args:
        variant (str): Native garak class name.
        name (str): Configured Target identity.

    Returns:
        list[str]: Native prompts without invoking the probe harness.
    """
    script = """
import json
import sys
from types import SimpleNamespace
from importlib.resources import files
from unittest.mock import patch
from garak.probes import Probe, dan

with patch.object(dan, 'data_path', files('garak').joinpath('data')), patch.object(Probe, '__init__', return_value=None), patch.object(Probe, 'probe', side_effect=AssertionError('Harness executed')):
    probe = getattr(dan, sys.argv[1])()
    prompts = list(probe.prompts)
    try:
        probe.probe(SimpleNamespace(name=sys.argv[2]))
    except AssertionError:
        pass
    else:
        raise RuntimeError('Native formatting did not reach the disconnected harness')
    print(json.dumps(probe.prompts))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, variant, name],
        check=True, capture_output=True, text=True,
    )
    content: list[str] = json.loads(result.stdout)
    return content
