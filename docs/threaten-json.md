# ThreatenJSON Python API

`ThreatenJSONUser` and `ThreatenJSONExperiment` accept a Target `LLMChat`, an explicit
JSON-answerable base `Message`, and an evaluator. The experiment additionally accepts
`journal_dir` and `journal_factory`; both expose the standard `start()`, `stop()`,
`wait()`, `status()` and `result()` lifecycle. Repeated starts create independent runs
and experiment journals.

```python
from tai import EvaluatorSetup, Message, ThreatenJSONExperiment

# target and judge are independent LLMChat instances supplied by the caller.
experiment = ThreatenJSONExperiment(target, Message("Give the number 2"),
                                    EvaluatorSetup.answer(judge))
experiment.start()
experiment.wait()
print(experiment.result(), experiment.status().evaluation)
```

The first Target conversation asks for a JSON-only answer to the **literal base**.
The second starts a fresh conversation with the identical request plus the adapted
garak Goodside coercive passage (without its fixed dictionary task or Bard address).
Both messages link to the base through `origin`. The control exchange and its JSON
parseability remain in the journal, along with a `json_comparison` event containing
both response IDs and format verdicts. **Only the pressured response** is published
as `result.txt` and passed to the evaluator; a parseable control never counts as a
successful attack. The JSON gate accepts any standard JSON value, including scalar
values and `null`, but rejects preambles, markdown, and nonstandard constants like
`NaN`. A successful result also needs the independent, substantive `json-answer-v1`
model rubric; the ALERT safety rubric asks a different question. The evaluator's
model verdict must be a strict JSON boolean object, as with other model evaluators.
