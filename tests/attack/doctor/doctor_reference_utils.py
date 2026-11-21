"""Isolated upstream reference without exposing or interpreting template content."""

import json
import subprocess
import sys


def reference(base: str, seed: int) -> list[str]:
    """Build reference candidates from pinned upstream AST and native encoding.

    Args:
        base (str): Exact request for substitution.
        seed (int): Explicit native encoding seed in the isolated process.

    Returns:
        list[str]: Five upstream-derived candidate prompts.
    """
    script = """
import ast, importlib.util, json, pathlib, random, re, sys
from garak.resources.encodings import leetspeak
tree = ast.parse(pathlib.Path(importlib.util.find_spec('garak.probes.doctor').origin).read_text())
classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
def assignment(name, attribute):
    return next(ast.literal_eval(node.value) for node in ast.walk(classes[name]) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == attribute for target in node.targets))
marker = '{{HARMFUL_BEHAVIOR}}'
base = sys.argv[1]
puppetry = re.sub(r'(?<=<query>)[^\\r\\n]*', lambda match: base, assignment('Puppetry', 'prompts')[0])
templates = assignment('Bypass', 'templates')
suffix = next(node.args[1].value for node in ast.walk(classes['BypassLeet']) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'replace' and isinstance(node.args[0], ast.Constant) and node.args[0].value == marker)
random.seed(int(sys.argv[2]))
print(json.dumps([puppetry] + [text.replace(marker, base) for text in templates] + [leetspeak(text.replace(marker, suffix).replace(marker, base)) for text in templates]))
"""
    process = subprocess.run(
        [
            sys.executable, "-c", script, base,
            str(seed)
        ],
        check=True, capture_output=True, text=True
    )
    return json.loads(process.stdout)
