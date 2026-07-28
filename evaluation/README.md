# GUI-World Evaluation

This package runs GUI-World benchmark inference and LLM-based judging. It
supports only the provider adapters implemented in `evaluation/providers`:
Azure OpenAI, Anthropic, Gemini, and Qwen for benchmark inference; OpenAI and
Together for judging. Mixtral, VideoChat2, and ChatUniVi are not evaluation
backends in this package.

Run every command below from the repository root.

## Installation

Python 3.9 or newer is required. A separate virtual environment keeps these
dependencies isolated from the Python 3.9-pinned GUI-Vid environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evaluation/requirements.txt
```

The requirements include the supported provider SDKs, OpenCV, Pillow, NumPy,
and pytest. Provider SDKs are imported lazily, so importing the package or
using offline utilities does not require credentials for every provider.

## Download the dataset

The canonical Hugging Face dataset is
[ONE-Lab/GUI-World](https://huggingface.co/datasets/ONE-Lab/GUI-World).
The optional Hugging Face CLI can download it into a local directory:

```bash
python -m pip install huggingface_hub
hf download ONE-Lab/GUI-World \
  --repo-type dataset \
  --local-dir GUI-World-data
```

Benchmark annotations are located at:

```text
GUI-World-data/Annotation/benchmark/android.jsonl
GUI-World-data/Annotation/benchmark/IOS.jsonl
GUI-World-data/Annotation/benchmark/XR.jsonl
GUI-World-data/Annotation/benchmark/multi.jsonl
GUI-World-data/Annotation/benchmark/software.jsonl
GUI-World-data/Annotation/benchmark/website.jsonl
```

Pass the download directory itself as `--video-root GUI-World-data`. Relative
`video_path` values in the JSONL are resolved below that directory; there is no
separate keyframe or video root.

## Providers and configuration

The operation matrix is intentionally explicit:

| Provider | Canonical CLI name | Benchmark | Judge |
| --- | --- | ---: | ---: |
| Azure OpenAI | `azure-openai` | Yes | No |
| Anthropic Claude | `anthropic` | Yes | No |
| Google Gemini | `gemini` | Yes | No |
| Alibaba Qwen | `qwen` | Yes | No |
| OpenAI | `openai` | No | Yes |
| Together | `together` | No | Yes |

Set the variables for only the provider you select:

| Provider | Required variables | Optional variable |
| --- | --- | --- |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT` | — |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | — |
| Gemini | `GEMINI_API_KEY`, `GEMINI_MODEL` | — |
| Qwen | `DASHSCOPE_API_KEY`, `DASHSCOPE_MODEL` | `DASHSCOPE_BASE_URL` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` | — |
| Together | `TOGETHER_API_KEY`, `TOGETHER_MODEL` | — |

Every API key remains required. Passing `--model-id MODEL` replaces the
provider's model variable for that command, so the corresponding
`*_MODEL` variable—or `AZURE_OPENAI_DEPLOYMENT`—may be omitted. It does not
replace endpoint, API version, or key variables.

For example:

```bash
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="..."
```

No model identifier is hard-coded because available model names and
deployments vary by account.

## Run benchmark inference

Inspect all available options:

```bash
python -m evaluation.benchmark --help
```

### Linspace sampling on Android

`linspace` selects up to ten evenly spaced decoded frames, including the first
and last frames. Android benchmark records do not include annotated keyframes,
so use `linspace` for Android:

```bash
python -m evaluation.benchmark \
  --input GUI-World-data/Annotation/benchmark/android.jsonl \
  --video-root GUI-World-data \
  --provider azure-openai \
  --scenario android \
  --strategy linspace
```

### Annotated sampling on iOS

`annotated` uses the JSONL `keyframes[].frame` values in annotation order,
removes duplicates, and selects at most ten frames:

```bash
python -m evaluation.benchmark \
  --input GUI-World-data/Annotation/benchmark/IOS.jsonl \
  --video-root GUI-World-data \
  --provider anthropic \
  --scenario ios \
  --strategy annotated
```

Use `--attempts` and `--base-delay` to tune bounded provider retries. To process
only part of a file, add an inclusive zero-based `--start` and an exclusive
`--end`:

```bash
python -m evaluation.benchmark \
  --input GUI-World-data/Annotation/benchmark/IOS.jsonl \
  --video-root GUI-World-data \
  --provider anthropic \
  --scenario ios \
  --strategy annotated \
  --start 0 \
  --end 10
```

When `--output` is omitted or set to `auto`, the result is written beside the
input as:

```text
output_<provider>_<scenario>_<strategy>_<start>_<end>.jsonl
```

For example, the first command produces a name such as
`output_azure-openai_android_linspace_0_123.jsonl`, where `123` is the number
of input records. Pass `--output path/to/results.jsonl` to choose a different
location; its parent directory must already exist.

An existing output is never overwritten accidentally. Re-run the same command
with `--resume` to retain tasks whose question, gold answer, and non-empty model
answer still match, and retry missing, failed, or stale tasks:

```bash
mkdir -p results
python -m evaluation.benchmark \
  --input GUI-World-data/Annotation/benchmark/IOS.jsonl \
  --output results/ios-anthropic.jsonl \
  --video-root GUI-World-data \
  --provider anthropic \
  --scenario ios \
  --strategy annotated \
  --resume
```

## Judge benchmark results

The judge reads benchmark result JSONL rather than the source annotations:

```bash
python -m evaluation.judge --help
```

Example with OpenAI:

```bash
python -m evaluation.judge \
  --input results/ios-anthropic.jsonl \
  --provider openai \
  --workers 8
```

OpenAI and Together providers are called concurrently across records.
`--workers` must be positive. `--start` is inclusive, `--end` is exclusive,
and `--resume` reuses only judgments whose question, gold answer, model result,
and judge schema still match.

Automatic judge output is written beside its input as:

```text
llm_judge_<provider>_<input-stem>_<start>_<end>.jsonl
```

As with benchmark inference, pass `--output` for an explicit path. Without
`--resume`, an existing output causes an error.

## Canonical names and legacy aliases

Names are case-insensitive. New commands should use the canonical names:

| Kind | Canonical | Accepted legacy alias |
| --- | --- | --- |
| Inference provider | `azure-openai` | `GPT-4V` |
| Inference provider | `anthropic` | `Claude3-Opus` |
| Inference provider | `gemini` | `Gemini-Pro` |
| Inference provider | `qwen` | `Qwen-VL-Max` |
| Judge provider | `openai` | `gpt-4` |
| Judge provider | `together` | `llama3` |
| Scenario | `android` | `mobile` |
| Frame strategy | `linspace` | `Random` |
| Frame strategy | `annotated` | `Extracted` |

Uppercase scenario spellings such as `IOS` and `XR` are accepted. The old
option names `--model`, `--setting`, and `--keyframe` remain aliases for
`--provider`, `--scenario`, and `--strategy`.

## JSONL format and errors

Input and output use strict UTF-8 JSON Lines:

- blank lines are ignored;
- every non-blank line must be one JSON object;
- `NaN`, `Infinity`, arrays, scalars, and malformed JSON are rejected;
- every record must contain `id` or `video_path`;
- record identifiers, converted to strings, must be unique;
- errors name the file and line number where possible.

Writes use a temporary file, `fsync`, and atomic replacement. The input and
output paths must differ, and the output directory must already exist and
permit creation of an atomic temporary file.

A successful benchmark task has this shape:

```json
{
  "id": "record-id",
  "video_path": "IOS/176.mp4",
  "result": {
    "Sequential-QA": {
      "q": "What happened before the menu opened?",
      "a": "The user selected a document.",
      "MLLM_result": {
        "Description": "Optional visual summary.",
        "Analysis": "Optional reasoning.",
        "Answer": "The user selected a document."
      }
    }
  }
}
```

`MLLM_result.Answer` is always a string after successful normalization.
`Description` and `Analysis` are included only when the provider returns them
as strings. A failed inference task has no `MLLM_result`; it receives an
`error` string and is retried by `--resume`.

Free-form judge tasks add:

```json
{
  "LLM_Judge": {
    "Evaluation": "The response agrees with the gold answer.",
    "Score": 4
  }
}
```

`Evaluation` must be non-empty and `Score` must be an integer from 1 through 5.
For `MCQA` and `Reasoning`, `LLM_Judge` is the provider's raw, unambiguous
`"Yes"` or `"No"` string for compatibility. Invalid judge output leaves
`LLM_Judge` absent and adds `judge_error`.

Before any judge request, all selected records are validated. A selected record
with a judgeable model answer must have a top-level `result` object and
non-empty string `q` and `a` values for that task. Structural errors stop the
run before provider calls or output writes.

## Tests and offline checks

The test suite uses fake clients and local fixtures; it makes no paid API calls
and requires no provider credentials:

```bash
python -c "import evaluation; import evaluation.benchmark; import evaluation.judge"
python -m evaluation.benchmark --help
python -m evaluation.judge --help
python -m pytest evaluation/tests -q
```

The case-study test also decodes every annotated frame and verifies it against
the retained PNG:

```bash
python -m pytest evaluation/tests/test_case_studies.py -q
```

For the included examples and their normalized annotations, see
[Case Studies](../case_studies/). Return to the
[project README](../Readme.md) for dataset and paper links.
