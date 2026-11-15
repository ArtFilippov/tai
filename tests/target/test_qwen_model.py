"""Reject oversized real tokenized conversations without relying on native warnings."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .qwen_fixture_utils import model


class TestQwenModel(unittest.TestCase):
    """Keep original input and full generation limits inside native context capacity."""

    def test_oversized_input_is_rejected_instead_of_native_context_overrun(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=2, context=8
            )
            with pytest.raises(
                ValueError, match="context"
            ):
                target.generate(
                    [
                        {
                            "role": "user",
                            "content": "word " * 8
                        }
                    ]
                )

    def test_generation_allowance_is_not_silently_reduced_to_fit(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=4, context=8
            )
            with pytest.raises(
                ValueError, match="context"
            ):
                target.generate(
                    [
                        {
                            "role": "user",
                            "content": "word " * 4
                        }
                    ]
                )

    def test_prefill_preserves_the_generation_allowance_and_context_boundary(self) -> None:
        with TemporaryDirectory() as directory:
            target = model(
                Path(directory),
                tokens=4, context=8
            )
            with pytest.raises(
                ValueError, match="context"
            ):
                target.prefill(
                    [
                        {
                            "role": "user",
                            "content": "word " * 4
                        }
                    ],
                    "reply"
                )
