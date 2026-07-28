# Submission Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the submitted GUI-World case studies and evaluation scripts as maintained, documented, tested repository components without committing the benchmark JSONL data.

**Architecture:** Create a standalone `evaluation` Python package with pure prompt, parsing, frame, and JSONL modules around lazily loaded provider adapters. Preserve the supplementary benchmark result schema, add atomic resume support, and move normalized case-study assets to `case_studies/`.

**Tech Stack:** Python 3.9+, argparse, OpenCV, Pillow, OpenAI, Anthropic, Google Gen AI, DashScope, Together, pytest

---

**Implementation base:** `3b81f82` (the approved design commit). Final
repository diffs are measured from this exact SHA rather than a relative
`HEAD~N` range.

## File Map

**Assets and documentation**

- Create: `case_studies/{android,ios,multi,software,website,xr}/`
- Modify: `Readme.md`
- Create: `evaluation/README.md`
- Create: `evaluation/requirements.txt`

**Evaluation package**

- Create: `evaluation/__init__.py` — package version and public metadata.
- Create: `evaluation/parsing.py` — provider-output JSON extraction and answer normalization.
- Create: `evaluation/prompts.py` — scenarios, tasks, prompt construction.
- Create: `evaluation/frames.py` — linspace and annotation-driven video frame extraction.
- Create: `evaluation/jsonl.py` — validated JSONL reads and atomic record upserts.
- Create: `evaluation/providers/base.py` — provider protocol, configuration helpers, retry.
- Create: `evaluation/providers/openai.py` — Azure OpenAI inference and OpenAI judge.
- Create: `evaluation/providers/anthropic.py` — Claude inference.
- Create: `evaluation/providers/gemini.py` — Gemini inference.
- Create: `evaluation/providers/qwen.py` — Qwen inference.
- Create: `evaluation/providers/together.py` — Together judge.
- Create: `evaluation/providers/__init__.py` — provider aliases and factories.
- Create: `evaluation/benchmark.py` — benchmark CLI and inference workflow.
- Create: `evaluation/judge.py` — judge CLI and scoring workflow.

**Tests**

- Create: `evaluation/tests/conftest.py`
- Create: `evaluation/tests/test_case_studies.py`
- Create: `evaluation/tests/test_parsing.py`
- Create: `evaluation/tests/test_prompts.py`
- Create: `evaluation/tests/test_frames.py`
- Create: `evaluation/tests/test_jsonl.py`
- Create: `evaluation/tests/test_providers.py`
- Create: `evaluation/tests/test_benchmark.py`
- Create: `evaluation/tests/test_judge.py`

## Chunk 1: Assets and Pure Evaluation Core

### Task 1: Establish the evaluation environment

**Files:**

- Create: `evaluation/requirements.txt`
- Create: `evaluation/__init__.py`

- [ ] **Step 1: Add compatible dependency ranges**

```text
anthropic>=0.49,<1
dashscope>=1.22,<2
google-genai>=1.11,<2
numpy>=1.23,<3
openai>=1.68,<3
opencv-python-headless>=4.7,<5
Pillow>=9.5,<13
pytest>=7.4,<9
together>=1.5,<3
tqdm>=4.64,<5
```

- [ ] **Step 2: Add a minimal package initializer**

Create `evaluation/__init__.py` with `__version__ = "1.0.0"`. Every Python
module that uses `X | Y` annotations must start with
`from __future__ import annotations` so the package remains importable on
Python 3.9.

- [ ] **Step 3: Install and smoke-test the environment**

Run:

```bash
python -m pip install -r evaluation/requirements.txt
python -c "import cv2, PIL, pytest; import evaluation"
```

Expected: both commands exit 0.

- [ ] **Step 4: Commit**

```bash
git add evaluation/__init__.py evaluation/requirements.txt
git commit -m "build(evaluation): add evaluation dependencies"
```

### Task 2: Normalize and validate case-study assets

**Files:**

- Create: `evaluation/tests/test_case_studies.py`
- Create: `case_studies/android/*`
- Create: `case_studies/ios/*`
- Create: `case_studies/multi/*`
- Create: `case_studies/software/*`
- Create: `case_studies/website/*`
- Create: `case_studies/xr/*`

- [ ] **Step 1: Write the failing asset-integrity test**

```python
import json
from pathlib import Path

SCENARIOS = {"android", "ios", "multi", "software", "website", "xr"}


def test_all_case_studies_are_complete():
    root = Path(__file__).parents[2] / "case_studies"
    assert {path.name for path in root.iterdir() if path.is_dir()} == SCENARIOS
    for scenario in SCENARIOS:
        case_dir = root / scenario
        annotation = json.loads((case_dir / "annotation.json").read_text())
        assert annotation["video_path"] == "example.mp4"
        assert (case_dir / "example.mp4").is_file()
        for keyframe in annotation["keyframes"]:
            assert (case_dir / f"{keyframe['frame']}.png").is_file()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest evaluation/tests/test_case_studies.py -q`

Expected: FAIL because `case_studies/` does not exist.

- [ ] **Step 3: Copy and normalize the submitted assets**

Run the exact directory copies:

```bash
mkdir -p case_studies
cp -R "Submission/Case Study/android" case_studies/android
cp -R "Submission/Case Study/IOS" case_studies/ios
cp -R "Submission/Case Study/multi" case_studies/multi
cp -R "Submission/Case Study/software" case_studies/software
cp -R "Submission/Case Study/website" case_studies/website
cp -R "Submission/Case Study/XR" case_studies/xr
mv case_studies/multi/emample.mp4 case_studies/multi/example.mp4
```

Rewrite only the top-level `video_path` with `jq`, then compare every other
parsed value to its source:

```bash
for annotation in case_studies/*/annotation.json; do
  annotation_tmp=$(mktemp)
  jq '.video_path = "example.mp4"' "$annotation" > "$annotation_tmp"
  mv "$annotation_tmp" "$annotation"
done

for mapping in \
  "android:android" "IOS:ios" "multi:multi" \
  "software:software" "website:website" "XR:xr"; do
  source_name=${mapping%%:*}
  destination_name=${mapping##*:}
  diff \
    <(jq -S 'del(.video_path)' "Submission/Case Study/$source_name/annotation.json") \
    <(jq -S 'del(.video_path)' "case_studies/$destination_name/annotation.json")
done

test -z "$(rg -l '/media/sata2' case_studies)"
```

Expected: every `diff` is empty and the final `test` exits 0.

- [ ] **Step 4: Select the six verified videos**

Retain only the submitted multi and website videos. Download the four official
scenario videos individually, with redirects and bounded retries:

```bash
curl -L --fail --retry 5 \
  https://huggingface.co/datasets/ONE-Lab/GUI-World/resolve/main/android/58.mp4 \
  -o case_studies/android/example.mp4
curl -L --fail --retry 5 \
  https://huggingface.co/datasets/ONE-Lab/GUI-World/resolve/main/IOS/176.mp4 \
  -o case_studies/ios/example.mp4
curl -L --fail --retry 5 \
  https://huggingface.co/datasets/ONE-Lab/GUI-World/resolve/main/software/172.mp4 \
  -o case_studies/software/example.mp4
curl -L --fail --retry 5 \
  https://huggingface.co/datasets/ONE-Lab/GUI-World/resolve/main/XR/151.mp4 \
  -o case_studies/xr/example.mp4
```

Android and XR do not have usable submitted source videos. The submitted
software video is byte-identical to official `XR/151.mp4`, rather than the
software annotation. The submitted iOS video does not align with its annotated
PNG frames. Therefore `software/172.mp4` and `IOS/176.mp4` replace those two
submitted videos as well.

Extend the asset test to decode every annotated frame from all six example
videos. Require the decoded frame and retained PNG to have equal dimensions and
a grayscale normalized correlation of at least `0.95`. This automated
all-frame similarity check verifies the complete video selection, not just file
presence or dimensions. Visually inspect at least one frame from each of the
four official replacements as an additional check.

- [ ] **Step 5: Run the asset test**

Run: `python -m pytest evaluation/tests/test_case_studies.py -q`

Expected: `1 passed`.

- [ ] **Step 6: Commit the asset migration**

```bash
git add case_studies evaluation/tests/test_case_studies.py
git commit -m "data: integrate GUI-World case studies"
```

### Task 3: Parse and normalize provider output

**Files:**

- Create: `evaluation/parsing.py`
- Create: `evaluation/tests/test_parsing.py`

- [ ] **Step 1: Write parsing tests**

Cover:

```python
def test_parse_fenced_answer():
    assert normalize_answer('```json\n{"Answer":"B"}\n```') == {"Answer": "B"}


def test_plain_text_becomes_answer():
    assert normalize_answer("The user clicks Save.") == {
        "Answer": "The user clicks Save."
    }


def test_invalid_answer_type_is_rejected():
    with pytest.raises(ResponseParseError):
        normalize_answer('{"Answer": ["B"]}')
```

Also test case-insensitive `answer`, retention of `Description` and `Analysis`,
balanced JSON embedded in prose, empty output, and malformed fenced JSON.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest evaluation/tests/test_parsing.py -q`

Expected: FAIL because `evaluation.parsing` does not exist.

- [ ] **Step 3: Implement minimal parsing**

Implement `ResponseParseError`, `extract_json_object(text)`, and
`normalize_answer(value)`. Accept dictionaries or strings. A non-JSON string is
a plain answer; a JSON dictionary must contain a string `Answer` or `answer`.
Return only string-valued `Description`, `Analysis`, and `Answer` fields.

- [ ] **Step 4: Verify parsing**

Run: `python -m pytest evaluation/tests/test_parsing.py -q`

Expected: all parsing tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/__init__.py evaluation/parsing.py evaluation/tests/test_parsing.py
git commit -m "feat(evaluation): add response parsing"
```

### Task 4: Build benchmark tasks and prompts

**Files:**

- Create: `evaluation/prompts.py`
- Create: `evaluation/tests/conftest.py`
- Create: `evaluation/tests/test_prompts.py`

- [ ] **Step 1: Add a representative record fixture**

The fixture must contain all ten output tasks: `Sequential-QA`, `Prediction`,
`Reasoning`, `Description1`, `Description2`, `Caption`, `static QA`, `MCQA`,
and two `Conversation` rounds. Include dictionary options for MCQA and list
options for Reasoning.

- [ ] **Step 2: Write failing prompt tests**

Test:

- `canonical_scenario("IOS") == "ios"`;
- `canonical_scenario("mobile") == "android"`;
- unknown scenarios raise `ValueError`;
- dictionary and list options become ordered strings;
- `build_tasks(record)` returns the exact ten task keys;
- MCQA/Reasoning questions include all options;
- conversation round two includes round-one context;
- generated prompts request `Description`, `Analysis`, and `Answer`.

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_prompts.py -q`

Expected: FAIL because `evaluation.prompts` does not exist.

- [ ] **Step 4: Implement prompt construction**

Expose:

```python
TASK_NAMES = (...)
FREE_FORM_TASKS = frozenset(...)
MULTIPLE_CHOICE_TASKS = frozenset({"MCQA", "Reasoning"})

def canonical_scenario(value: str) -> str: ...
def normalize_options(value: dict | list) -> list[str]: ...
def build_tasks(record: dict) -> dict[str, dict[str, str]]: ...
def build_prompt(scenario: str, task_name: str, question: str) -> str: ...
```

Match the supplementary wording where it affects evaluation behavior, correcting
grammar only where it does not change intent.

- [ ] **Step 5: Verify prompts**

Run: `python -m pytest evaluation/tests/test_prompts.py -q`

Expected: all prompt tests pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/prompts.py evaluation/tests/conftest.py evaluation/tests/test_prompts.py
git commit -m "feat(evaluation): add benchmark prompt construction"
```

### Task 5: Extract valid video frames

**Files:**

- Create: `evaluation/frames.py`
- Create: `evaluation/tests/test_frames.py`

- [ ] **Step 1: Write failing frame tests**

Generate tiny videos in pytest temporary directories. Test:

- `canonical_strategy("Random") == "linspace"`;
- `canonical_strategy("Extracted") == "annotated"`;
- 10-frame linspace indices include `0` and `frame_count - 1`;
- one-frame and five-frame videos never use invalid indices;
- annotated indices preserve order, remove duplicates, reject negatives, and
  clamp indices beyond the last frame;
- `annotated` with an absent or empty `keyframes` array raises
  `FrameExtractionError` explaining that the scenario must use `linspace`;
- unreadable and zero-frame videos raise `FrameExtractionError`;
- returned images are RGB Pillow images and the capture is released.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_frames.py -q`

Expected: FAIL because `evaluation.frames` does not exist.

- [ ] **Step 3: Implement frame extraction**

Expose:

```python
class FrameExtractionError(RuntimeError): ...

def canonical_strategy(value: str) -> str: ...
def linspace_indices(frame_count: int, count: int = 10) -> list[int]: ...
def annotated_indices(keyframes: list[dict], frame_count: int) -> list[int]: ...
def extract_frames(video_path: Path, strategy: str, keyframes=()) -> list[Image.Image]: ...
```

Use `numpy.linspace`, `cv2.VideoCapture`, `try/finally`, BGR-to-RGB conversion,
and no persistent temporary images. Android benchmark records do not contain
`keyframes`, so their supported strategy is `linspace`; this constraint must
also appear in `evaluation/README.md`.

- [ ] **Step 4: Verify frame extraction**

Run: `python -m pytest evaluation/tests/test_frames.py -q`

Expected: all frame tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/frames.py evaluation/tests/test_frames.py
git commit -m "feat(evaluation): add video frame extraction"
```

## Chunk 2: Providers and Workflows

### Task 6: Add provider adapters with lazy SDK loading

**Files:**

- Create: `evaluation/providers/base.py`
- Create: `evaluation/providers/openai.py`
- Create: `evaluation/providers/anthropic.py`
- Create: `evaluation/providers/gemini.py`
- Create: `evaluation/providers/qwen.py`
- Create: `evaluation/providers/together.py`
- Create: `evaluation/providers/__init__.py`
- Create: `evaluation/tests/test_providers.py`

- [ ] **Step 1: Write provider contract and configuration tests**

Test each factory with an isolated environment:

- missing required variables raise `ProviderConfigurationError` naming every
  missing variable;
- canonical and legacy provider aliases resolve correctly;
- unsupported inference/judge combinations fail before importing an SDK;
- `--model-id` style overrides take priority over environment model values;
- an injected fake client receives ordered image-before-text content;
- mocked SDK response shapes normalize through `normalize_answer`;
- retries stop after the configured attempt count.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_providers.py -q`

Expected: FAIL because provider modules do not exist.

- [ ] **Step 3: Implement the provider base**

Define:

```python
class ProviderConfigurationError(RuntimeError): ...

class InferenceProvider(Protocol):
    def generate(self, prompt: str, frames: Sequence[Image.Image]) -> dict: ...

class JudgeProvider(Protocol):
    def complete(self, system_prompt: str, prompt: str, json_output: bool) -> str: ...

def require_env(names: Sequence[str]) -> dict[str, str]: ...
def retry_call(function, attempts=3, base_delay=1.0): ...
```

`retry_call` retries exceptions with exponential backoff and raises the final
exception.

- [ ] **Step 4: Implement official SDK adapters**

- Azure OpenAI: `AzureOpenAI(...).chat.completions.create(...)`, data-URL JPEG
  frames, deployment from the model override or environment.
- OpenAI judge: `OpenAI().chat.completions.create(...)`.
- Anthropic: base64 JPEG `image` blocks followed by prompt text.
- Gemini: `google.genai.Client`, inline Pillow/byte image inputs supported by
  `models.generate_content`.
- Qwen: DashScope multimodal messages with temporary `file://` images scoped to
  a `TemporaryDirectory`.
- Together judge: `Together().chat.completions.create(...)`.

Do not instantiate SDK clients at module import time.

- [ ] **Step 5: Verify provider adapters**

Run: `python -m pytest evaluation/tests/test_providers.py -q`

Expected: all provider tests pass without network access.

- [ ] **Step 6: Commit**

```bash
git add evaluation/providers evaluation/tests/test_providers.py
git commit -m "feat(evaluation): add model provider adapters"
```

### Task 7: Add atomic JSONL persistence

**Files:**

- Create: `evaluation/jsonl.py`
- Create: `evaluation/tests/test_jsonl.py`

- [ ] **Step 1: Write failing JSONL tests**

Test valid loading, blank-line handling, line-numbered malformed JSON errors,
duplicate-ID rejection, atomic upsert preserving input order, and unchanged
output after a simulated replacement failure. Test that `record_id(record)`
returns `str(record["id"])` when present and otherwise returns
`str(record["video_path"])`, matching the supplementary code for the five
non-Android scenarios.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_jsonl.py -q`

Expected: FAIL because `evaluation.jsonl` does not exist.

- [ ] **Step 3: Implement JSONL helpers**

Expose:

```python
class JsonlError(RuntimeError): ...
def read_jsonl(path: Path) -> list[dict]: ...
def record_id(record: dict) -> str: ...  # id, then video_path fallback
def atomic_write_jsonl(path: Path, records: Iterable[dict]) -> None: ...
def upsert_record(records: list[dict], value: dict) -> list[dict]: ...
```

Write a named temporary file in the destination directory, flush and `fsync`,
then use `os.replace`.

- [ ] **Step 4: Verify JSONL helpers**

Run: `python -m pytest evaluation/tests/test_jsonl.py -q`

Expected: all JSONL tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/jsonl.py evaluation/tests/test_jsonl.py
git commit -m "feat(evaluation): add atomic JSONL persistence"
```

### Task 8: Implement benchmark inference CLI

**Files:**

- Create: `evaluation/benchmark.py`
- Create: `evaluation/tests/test_benchmark.py`

- [ ] **Step 1: Write workflow tests with a fake provider**

Test:

- `--output auto` generates a deterministic filename;
- `--video-root` resolves relative benchmark paths;
- `start` and `end` bounds select the intended records;
- every successful record has ten task results;
- failures add `error` without `MLLM_result`;
- resume reuses successful tasks, retries failed tasks, and never duplicates IDs;
- a complete resume record makes no provider calls;
- missing video and invalid range errors are actionable;
- the legacy CLI aliases work.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_benchmark.py -q`

Expected: FAIL because `evaluation.benchmark` does not exist.

- [ ] **Step 3: Implement the inference workflow**

Keep orchestration injectable:

```python
def run_benchmark(
    input_path: Path,
    output_path: Path,
    provider: InferenceProvider,
    scenario: str,
    strategy: str,
    video_root: Path | None = None,
    start: int = 0,
    end: int | None = None,
    resume: bool = False,
) -> None: ...
```

Construct and upsert one record at a time. Remove an old task `error` only after
that task succeeds. Add `main(argv=None)` and support module execution with
`python -m evaluation.benchmark`.

- [ ] **Step 4: Verify benchmark workflow**

Run: `python -m pytest evaluation/tests/test_benchmark.py -q`

Expected: all benchmark tests pass.

- [ ] **Step 5: Run CLI help smoke test**

Run: `python -m evaluation.benchmark --help`

Expected: exit 0 with documented provider, scenario, strategy, range, resume,
and video-root options.

- [ ] **Step 6: Commit**

```bash
git add evaluation/benchmark.py evaluation/tests/test_benchmark.py
git commit -m "feat(evaluation): add benchmark inference CLI"
```

### Task 9: Implement LLM judge CLI

**Files:**

- Create: `evaluation/judge.py`
- Create: `evaluation/tests/test_judge.py`

- [ ] **Step 1: Write judge tests with a fake provider**

Test:

- free-form tasks require a JSON object with string `Evaluation` and integer
  `Score` from 1 through 5;
- MCQA and Reasoning normalize only unambiguous Yes/No output while storing the
  legacy raw string;
- tasks without `MLLM_result` are skipped;
- malformed output adds `judge_error`;
- resume retries only missing/failed judge results;
- concurrent work returns output in input order and all writes occur on the
  caller thread;
- `start`, `end`, `workers`, and automatic output paths behave correctly.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest evaluation/tests/test_judge.py -q`

Expected: FAIL because `evaluation.judge` does not exist.

- [ ] **Step 3: Implement judge prompts and workflow**

Expose:

```python
def judge_free_form(provider, question, gold, answer) -> dict: ...
def judge_multiple_choice(provider, question, gold, answer) -> str: ...
def run_judge(input_path, output_path, provider, *, start=0, end=None,
              workers=8, resume=False) -> None: ...
def main(argv=None) -> None: ...
```

Use the supplementary scoring rubric. Worker threads perform provider calls and
return completed records; only the main thread calls `atomic_write_jsonl`.

- [ ] **Step 4: Verify judge workflow**

Run: `python -m pytest evaluation/tests/test_judge.py -q`

Expected: all judge tests pass.

- [ ] **Step 5: Run CLI help smoke test**

Run: `python -m evaluation.judge --help`

Expected: exit 0 with provider, model, range, workers, and resume options.

- [ ] **Step 6: Commit**

```bash
git add evaluation/judge.py evaluation/tests/test_judge.py
git commit -m "feat(evaluation): add LLM judge CLI"
```

## Chunk 3: Dependencies, Documentation, and Cleanup

### Task 10: Document installation and end-to-end usage

**Files:**

- Create: `evaluation/README.md`
- Modify: `Readme.md`

- [ ] **Step 1: Write evaluation documentation**

Document:

- Python environment creation and installation;
- canonical `ONE-Lab/GUI-World` download commands and benchmark paths under
  `Annotation/benchmark/`;
- provider operation matrix and environment variables;
- linspace and annotated inference commands;
- judge commands;
- JSONL result examples;
- resume behavior;
- how to run tests;
- migration aliases and removed unsupported choices.

- [ ] **Step 2: Update the root README**

Remove the stale “release soon” line. Add concise Case Studies and Evaluation
sections linking to `case_studies/`, `evaluation/README.md`, and the Hugging Face
dataset. Add both sections to Contents.

- [ ] **Step 3: Validate README commands**

Run every offline command shown in the documentation (`--help`, test command,
and import smoke tests). Inspect relative links with `rg`.

- [ ] **Step 4: Commit**

```bash
git add Readme.md evaluation/README.md
git commit -m "docs: add GUI-World evaluation guide"
```

### Task 11: Remove staging material and run final verification

**Files:**

- Remove: `Submission/`

- [ ] **Step 1: Confirm every retained source asset has a destination**

Run a read-only semantic comparison that:

- maps source `IOS`/`XR` to destination `ios`/`xr`;
- compares SHA-256 for every PNG;
- removes only top-level `video_path` from both parsed annotation dictionaries
  and asserts the remaining structures are equal;
- compares the submitted and destination video hashes only for multi and
  website, accounting for the corrected `multi/example.mp4` name;
- checks the four official Android, iOS, software, and XR videos against fixed
  SHA-256 values captured from the verified migration;
- reruns the all-keyframe alignment test for every case study.

The command must exit nonzero on the first mismatch and print the mismatched
path. Rerun `python -m pytest evaluation/tests/test_case_studies.py -q` before
deletion.

Run this exact comparison:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

source_root = Path("Submission/Case Study")
destination_root = Path("case_studies")
mapping = {
    "android": "android",
    "IOS": "ios",
    "multi": "multi",
    "software": "software",
    "website": "website",
    "XR": "xr",
}
retained_source_videos = {
    "multi": "emample.mp4",
    "website": "example.mp4",
}
official_video_sha256 = {
    "android": (
        "f61e0276b918d67c8e600fdda46a69865d78efca100a27fc97d166eea4eaecc7"
    ),
    "ios": (
        "fe8f47964ec829de7a3813b9be7cd3a33644d8087851a33f6163b40276b7ec94"
    ),
    "software": (
        "1fe44ee2fdf57bc5876fdd89d93b8df2860da135bee9c55f0e45f9737a7753d1"
    ),
    "xr": (
        "ce48735a6bdb3d0df92cd9796cbad0330a48fd6451bd5610610c1e2d51b60ce8"
    ),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fail(message):
    raise SystemExit(message)


for source_name, destination_name in mapping.items():
    source = source_root / source_name
    destination = destination_root / destination_name

    source_pngs = {path.name: digest(path) for path in source.glob("*.png")}
    destination_pngs = {
        path.name: digest(path) for path in destination.glob("*.png")
    }
    if source_pngs != destination_pngs:
        fail(f"PNG mismatch: {source} -> {destination}")

    source_annotation = json.loads((source / "annotation.json").read_text())
    destination_annotation = json.loads(
        (destination / "annotation.json").read_text()
    )
    source_annotation.pop("video_path", None)
    destination_annotation.pop("video_path", None)
    if source_annotation != destination_annotation:
        fail(f"Annotation mismatch: {source} -> {destination}")

    if source_name in retained_source_videos:
        source_video = source / retained_source_videos[source_name]
        destination_video = destination / "example.mp4"
        if digest(source_video) != digest(destination_video):
            fail(f"Video mismatch: {source_video} -> {destination_video}")

for scenario, expected in official_video_sha256.items():
    video = destination_root / scenario / "example.mp4"
    actual = digest(video)
    if actual != expected:
        fail(
            f"Official video mismatch: {video}; "
            f"expected {expected}, got {actual}"
        )

print("Case-study semantic comparison passed")
PY
python -m pytest evaluation/tests/test_case_studies.py -q
```

Expected: `Case-study semantic comparison passed` and the asset test passes.

- [ ] **Step 2: Remove the staging directory**

Delete `Submission/` only after the comparison passes. This intentionally
removes the excluded benchmark JSONL and duplicate ZIP from the local worktree.
Report that they remain available from Hugging Face and the original submission
archive, not from git.

- [ ] **Step 3: Run static and unit verification**

Run:

```bash
python -m compileall -q evaluation
python -m pytest evaluation/tests -q
python -m evaluation.benchmark --help
python -m evaluation.judge --help
git diff --check
```

Expected: compilation succeeds, all tests pass, both CLIs exit 0, and
`git diff --check` prints nothing.

- [ ] **Step 4: Inspect repository state**

Run: `git status --short` and `git diff --stat 3b81f82..HEAD`.

Expected: only the pre-existing `.DS_Store` remains untracked; no
`Submission/`, benchmark JSONL, ZIP, secrets, caches, or generated result files
are present.

- [ ] **Step 5: Run secret and stale-reference checks**

Run:

```bash
rg -n "<Your API|api_key\\s*=|/media/sata2|Submission/|emample|mixtral|ChatUnivi" \
  evaluation case_studies Readme.md
```

Expected: no hard-coded secrets or stale paths; any intentional migration note
is clearly documented.

- [ ] **Step 6: Request code review**

Dispatch a code-reviewer against the approved design and all implementation
commits. Resolve correctness, security, and compatibility findings and repeat
verification.

- [ ] **Step 7: Commit any review fixes**

```bash
git add <specific-reviewed-files>
git commit -m "fix(evaluation): address integration review"
```

- [ ] **Step 8: Confirm final status**

Run: `git status --short --branch`

Expected: branch is ahead by the integration commits and only the user's
pre-existing `.DS_Store` is untracked.
