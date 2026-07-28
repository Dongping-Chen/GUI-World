# Submission Integration Design

## Goal

Integrate the case-study assets and evaluation code from `Submission/` into the
main GUI-World repository as maintained, documented, and tested components.
Benchmark JSONL data remains hosted on Hugging Face and is not committed to this
repository.

## Scope

The integration includes:

- the six case studies and their annotations, videos, and extracted frames;
- benchmark inference for the providers already represented by working
  implementations in the supplementary code;
- LLM-based evaluation for the judge providers already represented by working
  implementations;
- dependency declarations, usage documentation, configuration validation, and
  offline tests.

The integration excludes:

- `Submission/Benchmark/`;
- the duplicate `Submission/Code.zip`;
- model backends that only appear as command-line choices but have no
  implementation (`VideoChat2`, `ChatUniVi`, and Mixtral);
- changes to the `GUI_Vid/` training and local-inference implementation.

## Repository Layout

Case-study assets move to a top-level `case_studies/` directory:

```text
case_studies/
├── android/
├── ios/
├── multi/
├── software/
├── website/
└── xr/
```

Scenario directory names are normalized to lowercase. Each directory retains
its `annotation.json`, example video, and extracted frame images. The misspelled
`multi/emample.mp4` filename becomes `multi/example.mp4`.

The staged Android, iOS, multi, and website videos are retained after
byte-for-byte comparison with their destinations. The staged software video is
actually identical to official `XR/151.mp4`, so it is replaced with official
`software/172.mp4`. The staged XR directory has no video, so official
`XR/151.mp4` is restored. The final Android and iOS videos correspond to
`android/58.mp4` and `IOS/176.mp4`. All submitted PNGs and annotation content
are preserved, and every annotated frame is decoded from the selected video
and checked for alignment in the asset test. Every copied annotation replaces
its stale machine-specific `video_path` with the relative value `example.mp4`.

Evaluation code becomes an importable top-level `evaluation/` package:

```text
evaluation/
├── __init__.py
├── benchmark.py
├── frames.py
├── judge.py
├── parsing.py
├── prompts.py
├── providers/
├── requirements.txt
├── README.md
└── tests/
```

The root `Readme.md` links to both components and explains that full benchmark
data is downloaded from Hugging Face.

## Evaluation Architecture

`benchmark.py` owns command-line parsing and the JSONL inference loop. It loads
benchmark records, builds the ten GUI-World tasks for each record, resolves
video paths relative to an optional `--video-root`, obtains frames, invokes a
selected provider, and writes records in the existing result schema.

`prompts.py` owns scenario aliases, task prompts, task extraction, and prompt
construction. Canonical scenarios are `android`, `ios`, `multi`, `software`,
`website`, and `xr`; legacy `mobile` and uppercase spellings remain accepted.

`frames.py` provides two explicit strategies:

- `linspace`: ten evenly spaced frames, matching the behavior previously called
  `Random`;
- `annotated`: frames selected by the integer `frame` fields in the benchmark
  record's `keyframes` array, in annotation order and capped at ten. Duplicate
  indices are removed while preserving the first occurrence.

The legacy `Random` and `Extracted` CLI spellings remain accepted as aliases for
`linspace` and `annotated`, respectively. Frames are decoded directly from the
video; evaluation does not depend on a `selectedframes/` side directory or a
second keyframe root.
The unsupported `Description` and `Caption` frame settings are removed.

`providers/` contains one adapter per service. The supported operation matrix is:

| Provider | Benchmark inference | LLM judge |
| --- | --- | --- |
| Azure OpenAI | Yes | No |
| OpenAI | No | Yes |
| Anthropic Claude | Yes | No |
| Google Gemini | Yes | No |
| Alibaba Qwen | Yes | No |
| Together | No | Yes |

Adapters share a small protocol but load their SDK lazily. Selecting one
provider validates only that provider's environment variables. API keys,
endpoints, deployments, and model identifiers are never embedded in source
code. Provider-specific model identifiers can be set through CLI options or
environment variables so that SDK and model lifecycle changes do not require
editing evaluation logic.

The benchmark CLI accepts canonical providers `azure-openai`, `anthropic`,
`gemini`, and `qwen`. It also accepts legacy aliases `GPT-4V`,
`Claude3-Opus`, `Gemini-Pro`, and `Qwen-VL-Max`. The judge CLI accepts `openai`
and `together`; legacy `gpt-4` maps to `openai` and legacy `llama3` maps to
`together`.

Configuration names are:

- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
  `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_DEPLOYMENT`;
- OpenAI: `OPENAI_API_KEY` and `OPENAI_MODEL`;
- Anthropic: `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`;
- Gemini: `GEMINI_API_KEY` and `GEMINI_MODEL`;
- Qwen: `DASHSCOPE_API_KEY`, `DASHSCOPE_MODEL`, and optional
  `DASHSCOPE_BASE_URL`;
- Together: `TOGETHER_API_KEY` and `TOGETHER_MODEL`.

An explicit `--model-id` overrides the corresponding model or deployment
environment variable.

`judge.py` reads benchmark result JSONL, scores free-form tasks and
multiple-choice tasks, and writes the same enriched record format used by the
supplementary code. OpenAI and Together are supported as judge providers.
Concurrent evaluation preserves input order and performs output writes on the
main thread, preventing interleaved JSONL records.

`parsing.py` centralizes extraction of JSON from plain or fenced model output.
Malformed output produces a clear error or a controlled fallback rather than
mutating strings heuristically.

## Data Flow

```text
Hugging Face JSONL + local videos
               |
               v
       record/task extraction
               |
               v
     linspace or annotated frames
               |
               v
         provider adapter
               |
               v
      benchmark result JSONL
               |
               v
          judge provider
               |
               v
       judged result JSONL
```

The result payload retains `id`, `video_path`, and the per-task `q`, `a`,
`MLLM_result`, and `LLM_Judge` fields so existing downstream analysis remains
compatible.

## Reliability and Error Handling

- CLI arguments use explicit choices and actionable validation errors instead
  of bare assertions.
- Missing input files, videos, annotated frames, dependencies, or credentials
  fail with a message naming the missing resource.
- API calls use bounded retry with configurable attempts and exponential
  backoff.
- Output files are protected from accidental duplicate appends. In resume mode,
  an existing record is skipped only when every expected task has a successful
  result. Incomplete records retain successful task results and retry only
  missing or failed tasks. Each record is upserted through a temporary file and
  atomic replacement, so output contains at most one line per input ID and an
  interruption cannot leave a partially written JSON line.
- Video capture is always released, empty/corrupt videos are rejected, frame
  indices are valid for short videos, and temporary files are not left beside
  user videos.
- Provider responses are normalized to an answer object before serialization.
- Individual record failures are reported with record and task context.

## Result Schema

Benchmark output retains the supplementary code's top-level structure:

```json
{
  "id": "record-id",
  "video_path": "final_videos/android/58.mp4",
  "result": {
    "Sequential-QA": {
      "q": "question",
      "a": "gold answer",
      "MLLM_result": {
        "Description": "optional visual summary",
        "Analysis": "optional reasoning",
        "Answer": "model answer"
      }
    },
    "MCQA": {
      "q": "question A. ... B. ...",
      "a": "B",
      "MLLM_result": {
        "Answer": "B"
      }
    }
  }
}
```

The parser always normalizes a successful inference response to a dictionary
with a string `Answer`; `Description` and `Analysis` are retained when supplied.
A failed task has no `MLLM_result` and instead contains an `error` string.
Resume retries exactly those tasks.

Judge output enriches successful tasks without changing `MLLM_result`:

```json
{
  "LLM_Judge": {
    "Evaluation": "comparison with the gold answer",
    "Score": 4
  }
}
```

For multiple-choice tasks, `LLM_Judge` remains the raw string `"Yes"` or `"No"`
for compatibility with the supplementary output schema. Malformed judge output
leaves `LLM_Judge` absent and adds `judge_error`; resume retries it.

## Dependencies

`evaluation/requirements.txt` contains only evaluation and test dependencies,
separate from the heavyweight, Python 3.9-pinned `GUI_Vid/requirements.txt`.
Provider SDKs follow their current official clients:

- `openai`;
- `anthropic`;
- `google-genai`;
- `dashscope`;
- `together`;
- plus Pillow, OpenCV, tqdm, and pytest.

Runtime imports stay lazy so users can execute offline utilities and tests
without configuring every provider.

## Testing

All automated tests run without network access or paid API calls.

Unit tests cover:

- option normalization and task construction for representative benchmark
  records;
- canonical and legacy scenario/frame-strategy aliases;
- fenced, plain, and malformed provider output parsing;
- evenly spaced sampling for normal, one-frame, short, and invalid videos;
- annotation-driven frame extraction, deduplication, and ordering;
- provider configuration validation and mocked SDK response normalization;
- output protection, resume behavior, and ordered JSONL writing;
- free-form and multiple-choice judge routing.

An asset-integrity test verifies that all six case-study directories contain a
parseable annotation, an example video, and every retained annotated PNG. It
decodes all annotated video frames, requires equal image dimensions, and checks
normalized grayscale correlation against the PNGs.

Static validation compiles the package and runs the complete pytest suite.

## Documentation

`evaluation/README.md` documents:

- installation in a separate environment;
- required environment variables by provider;
- downloading benchmark JSONL and videos from Hugging Face;
- inference examples for linspace and annotated frames;
- judge examples;
- result schema, resume behavior, and troubleshooting.

The root README replaces the stale benchmark-release notice with links to the
integrated evaluation module, case studies, and Hugging Face dataset.

## Migration

After copied assets and code pass integrity checks, the untracked `Submission/`
staging directory is removed. Only `case_studies/`, `evaluation/`, documentation,
and tests are committed. This prevents the benchmark JSONL and duplicate ZIP
from entering version control.
