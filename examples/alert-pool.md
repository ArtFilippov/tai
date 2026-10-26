# Offline ALERT base-request pool

`alert-manifest-640.json` is a checked-in pool of **original, unmodified**
`prompt` values: twenty records with the smallest source IDs in each of the
32 Figure 2 microcategories, in the order of `ALERT_taxonomy.md`. It contains
no model responses. The instruction/response wrapper is part of each original
prompt; no extraction, translation, or paraphrase is applied.

The source is `Babelscape/ALERT`, config `alert`, split `test`, file
`alert.jsonl` at revision `c76d0637f4573742982d5fe961a31bdb3d1e2c80`.
Its SHA256 is `0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4`.
The 14,763 source rows and the complete Figure 2 mapping are checked during
preparation. The manifest also records the SHA256 of `ALERT_taxonomy.md`,
`0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd`.
The committed manifest's SHA256 is
`3ee64c7a6b86c2b28174d8c699407e24f94a4e54ca9b8f9c5fc35bf460725f43`.

From the repository root, download the source explicitly (the only online step):

```bash
SOURCE=$(uv run --locked --group dev --extra qwen python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("Babelscape/ALERT", filename="alert.jsonl", repo_type="dataset", revision="c76d0637f4573742982d5fe961a31bdb3d1e2c80"))')
uv run --locked --group dev --extra qwen python -m main.alert_manifest_cli "$SOURCE" examples/alert-manifest-640.json
sha256sum examples/alert-manifest-640.json
```

Preparation uses no randomness: records are sorted by numeric source ID within
each canonical category, then the first twenty are selected. It validates all
source records, unique IDs, known labels, nonempty text, checksums, and complete
coverage *before* writing the manifest. It is safe to regenerate without a model.

The manifest is JSON with `source` (revision, file, SHA256, row count, dataset
config and split), `taxonomy` (document, Figure 2 version, SHA256 and all named macro groups and
their ordered microcategories), `selection` (algorithm, limit and text rule),
and `requests` (640 ordered objects: `id`, `prompt`, prompt `sha256`,
`source_category`, canonical `microcategory`, `macro`). The Figure 2 spelling
`crime_kidnapp` retains the original source spelling `crime_kidnap` in each
record. Unrecognized categories raise an error.

**Pool API:** `from base.base_request import BaseRequest` (a Protocol
with `text() -> str` and `metadata() -> Mapping[str, str | int]`),
`from base.alert_identity import AlertIdentity` (source ID and categories), and
`from base.alert_pool import AlertPool`. Construct
`AlertPool(Path("examples/alert-manifest-640.json"))`, then call `requests(N)`
for `1 <= N <= 20`; it returns `32 * N` `AlertIdentity` instances, category-major
in canonical Figure 2 order. `AlertPool.taxonomy()` returns a fresh full mapping
of macro keys to names and ordered microcategories. Each request's `text()` returns
the raw prompt; `metadata()` returns its source ID, source category,
canonical microcategory, macro key, and pinned source and taxonomy provenance.
`requests(N)` reads only the local manifest and validates **all 640** records
before returning any selection.

Run the offline pool from the repository root after `tai-qwen prepare`:

```bash
uv run --locked --group dev --extra qwen tai-alert /tmp/alert-example --gpu 0 --limit 1
```

`--limit` defaults to 20 and selects the first N entries **per category** (32*N
runs). The command accepts `--manifest PATH` for an explicit local manifest,
and `--turns`, `--tokens`, `--attacker qwen|gguf`, `--host`, `--port` as in
`tai-qwen run`. It verifies the entire manifest before contacting a model.
Give each invocation a new output directory; the command never overwrites an
existing one. Each run has its own JSONL journal with `start.base_metadata`;
`summary.json` indexes source IDs and relative journal paths and counts the
recorded final verdicts across all 32 microcategories and six macro groups.
`rate` is the evaluated-only success percentage (`null` with no evaluations).
