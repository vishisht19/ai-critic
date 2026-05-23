# commit-critic

An AI-powered CLI that reviews your git commit history and helps you write better commit messages. It scores each commit 1-10, explains what's weak, and streams a suggested message from your staged diff.

Works with Anthropic (Claude), OpenAI (GPT), or a local Ollama model. No lock-in.

## Setup

```bash
pip install -r requirements.txt
```

Set at least one API key (or run Ollama locally) - see [Configuration](#configuration) below.

## Sample output

```
COMMIT CRITIC

COMMITS THAT NEED WORK

  Commit: 3f7a1bc "fix bug"
  Score:  2/10  (vague)
  Issue:  No context - which bug, in which component, what was the impact?
  Better: fix(auth): resolve null pointer in token refresh handler

WELL-WRITTEN COMMITS

  Commit: a9d2e41 "feat(cache): add Redis caching layer"
  Score:  9/10  (excellent)
  Why it's good: Conventional commit, specific scope, measurable impact stated

YOUR STATS
Average score:       4.8/10
Vague commits:       28 (56%)
WIP commits:         4 (8%)
One-word commits:    6 (12%)
Well-written (8+):   7 (14%)
```

## Configuration

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | One of three | - | Use Anthropic Claude |
| `OPENAI_API_KEY` | One of three | - | Use OpenAI |
| `LLM_PROVIDER` | No | auto-detect | Force provider: `anthropic`, `openai`, `ollama` |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `llama3.2` | Ollama model name |
| `LANGSMITH_API_KEY` | No | - | Enables LangSmith tracing |
| `LANGSMITH_TRACING` | No | `false` | Set to `true` to activate tracing |
| `LANGSMITH_PROJECT` | No | `commit-critic` | Groups traces in LangSmith dashboard |

Provider is auto-detected in priority order: `LLM_PROVIDER` env var, then `ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`, then a running Ollama instance.

## Score categories

| Category | Score range | Meaning |
|---|---|---|
| `wip` | 1-3 | Work-in-progress placeholder (`wip`, `WIP: doing stuff`) |
| `vague` | 1-5 | Has words but no useful context (`fixed bug`, `update`) |
| `good` | 6-7 | Clear scope and action (`fix(auth): handle null token`) |
| `excellent` | 8-10 | Conventional commit with context and impact |

Calibration anchors included in every prompt keep scores consistent across providers and runs (within ~1 point).

## Usage

**Analyze your last 50 commits:**
```bash
python commit_critic.py --analyze
```

**Analyze any public GitHub repo:**
```bash
python commit_critic.py --analyze --url="https://github.com/steel-dev/steel-browser"
```

**Analyze more commits and output JSON:**
```bash
python commit_critic.py --analyze --n=100 --output=json
```

**Run LLM-as-a-judge validation after scoring:**
```bash
python commit_critic.py --analyze --judge
```
A second LLM call audits a sample of the scores (1 highest, 1 lowest, up to 3 borderline). It flags cases where the primary score looks unfair and rates the quality of each critique 1-5. Useful for catching systematic bias in a model's scoring.

**Generate a commit message from staged changes:**
```bash
git add .
python commit_critic.py --write
```

**Run benchmark eval:**
```bash
python commit_critic.py --eval
```

**Install git hooks:**
```bash
python commit_critic.py --install-hooks
```

## Hook installation

```bash
python commit_critic.py --install-hooks
# Then just use git commit normally — suggestions appear automatically
```

Two hooks are installed into `.git/hooks/`:

- `prepare-commit-msg` - fires before your editor opens and pre-fills it with an AI-suggested message. You can accept, edit, or clear it.
- `post-commit` - fires after every successful commit and prints a one-line score: `[commit-critic] Score: 7/10 - ...`

A Claude Code `PostToolUse` hook is also written to `.claude/settings.json` so the post-commit score appears when you commit via the Claude Code Bash tool.

To remove everything:
```bash
python commit_critic.py --uninstall-hooks
```

## Ollama quick-start

```bash
ollama serve
ollama pull llama3.2
python commit_critic.py --analyze  # auto-detects Ollama
```

Override the model:
```bash
OLLAMA_MODEL=mistral python commit_critic.py --analyze
```

## Eval mode

`--eval` runs the scorer against 15 hardcoded benchmark commits with known quality labels and expected score ranges. It reports:

- **Within expected range** - how many commits landed in their expected score band
- **Mean absolute error** - average distance from the midpoint of the expected range
- **Category accuracy** - how often vague/wip/good/excellent was classified correctly

Results are appended to `.commit_critic_metrics.jsonl` so you can compare accuracy across providers and models over time.

Example output:
```
EVAL RESULTS  (provider: anthropic, model: claude-haiku-4-5-20251001)
Within expected range:  12/15 (80%)
Mean absolute error:    0.8 points
Category accuracy:      13/15 (87%)
```

## Score caching

Results are cached in `.commit_critic_cache.json` keyed by commit hash. Re-running `--analyze` on the same repo only sends uncached commits to the LLM, making repeated runs fast and cheap. The cache is safe to commit or delete.

## LangSmith tracing (optional)

```bash
export LANGSMITH_API_KEY=ls__...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=commit-critic
python commit_critic.py --analyze
```

Every `analyze_commits` and `suggest_commit` call appears in your LangSmith dashboard with prompt, response, latency, and token usage. Has zero effect when not configured.

## Local metrics

Every run appends one line to `.commit_critic_metrics.jsonl`:
```json
{"ts": "2026-05-20T10:30:00Z", "mode": "analyze", "provider": "anthropic", "model": "claude-haiku-4-5-20251001", "commits": 50, "avg_score": 4.2, "latency_ms": 2100}
```

Inspect with:
```bash
cat .commit_critic_metrics.jsonl | python -m json.tool
```

## Running tests

```bash
pytest test_commit_critic.py -v
```

The test suite covers: git log parsing, all four JSON fallback strategies, provider auto-detection, cache roundtrip, stats computation, commit partitioning, hook install/uninstall, and eval metrics. No LLM calls are made - all external dependencies are mocked.
