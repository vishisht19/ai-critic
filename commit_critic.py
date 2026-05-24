#!/usr/bin/env python3
"""AI-powered commit message critic and writer."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

try:
    import openai as _openai
except ImportError:
    _openai = None

try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

console = Console()


@dataclass
class ScoredCommit:
    """A commit with its LLM-assigned score and critique."""
    hash: str
    message: str
    author: str
    date: str
    score: int                          # 1-10
    issue: str
    suggestion: str
    category: str                       # vague | wip | good | excellent
    judge_score: Optional[int] = None   # set by EvalRunner if --judge is used
    judge_note: Optional[str] = None


# ---------------------------------------------------------------------------
# GitClient
# ---------------------------------------------------------------------------

class GitClient:
    """Thin subprocess wrapper around git operations."""

    def get_commits(self, repo_path: str, n: int = 50) -> list[dict]:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"-{n}", "--pretty=format:%H|%s|%an|%ai"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[red]Not a git repository: {repo_path}[/red]")
            sys.exit(1)
        commits = []
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        return commits

    def get_staged_diff(self) -> tuple[str, str]:
        diff_result = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True, text=True
        )
        stat_result = subprocess.run(
            ["git", "diff", "--staged", "--stat"],
            capture_output=True, text=True
        )
        return diff_result.stdout, stat_result.stdout

    def clone_remote(self, url: str, depth: int = 50) -> str:
        tmp = tempfile.mkdtemp()
        result = subprocess.run(
            ["git", "clone", f"--depth={depth}", "--single-branch", url, tmp],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            console.print(f"[red]Could not clone repository. Check the URL and your network.[/red]")
            sys.exit(1)
        return tmp

    def get_head_commit(self, repo_path: str) -> dict:
        commits = self.get_commits(repo_path, n=1)
        if not commits:
            console.print("[red]No commits found.[/red]")
            sys.exit(1)
        return commits[0]


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")

_ANALYZE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index":      {"type": "integer"},
            "score":      {"type": "integer", "minimum": 1, "maximum": 10},
            "issue":      {"type": "string"},
            "suggestion": {"type": "string"},
            "category":   {"type": "string", "enum": ["vague", "wip", "good", "excellent"]},
        },
        "required": ["index", "score", "issue", "suggestion", "category"],
    },
}

_JUDGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index":       {"type": "integer"},
            "judge_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "fair":        {"type": "boolean"},
            "note":        {"type": "string"},
        },
        "required": ["index", "judge_score", "fair", "note"],
    },
}


def _load_prompt(name: str, version: str | None = None) -> str:
    """Load prompts/{version}/{name}.md."""
    ver = version or _PROMPT_VERSION
    return (_PROMPTS_DIR / ver / f"{name}.md").read_text()


class LLMClient:
    """Provider-agnostic LLM adapter supporting Anthropic, OpenAI, and Ollama."""

    _CACHE_FILE = Path(".commit_critic_cache.json")
    _DEFAULT_TEMP = 0.4

    def __init__(self):
        self.provider, self.model = self._detect_provider()
        self.temperature = self._load_calibrated_temperature()

    def _load_calibrated_temperature(self) -> float:
        if self._CACHE_FILE.exists():
            try:
                return json.loads(self._CACHE_FILE.read_text()).get("_meta", {}).get("calibrated_temperature", self._DEFAULT_TEMP)
            except Exception:
                pass
        return self._DEFAULT_TEMP

    @staticmethod
    def _save_calibrated_temperature(temp: float) -> None:
        cache_file = LLMClient._CACHE_FILE
        cache: dict = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
            except Exception:
                pass
        cache.setdefault("_meta", {})["calibrated_temperature"] = temp
        try:
            cache_file.write_text(json.dumps(cache, indent=2))
        except Exception:
            pass

    def _detect_provider(self) -> tuple[str, str]:
        """Auto-detect provider: LLM_PROVIDER env > ANTHROPIC_API_KEY > OPENAI_API_KEY > Ollama."""
        explicit = os.environ.get("LLM_PROVIDER", "").lower()
        if explicit in ("anthropic", "openai", "ollama"):
            return self._init_explicit(explicit)

        if os.environ.get("ANTHROPIC_API_KEY"):
            return ("anthropic", "claude-haiku-4-5-20251001")

        if os.environ.get("OPENAI_API_KEY"):
            return ("openai", "gpt-5.4-mini-2026-03-17")

        # Check Ollama
        ollama_result = self._check_ollama()
        if ollama_result:
            return ollama_result

        console.print(
            "[red]No LLM provider found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or run Ollama locally.[/red]"
        )
        sys.exit(1)

    def _init_explicit(self, provider: str) -> tuple[str, str]:
        if provider == "anthropic":
            return ("anthropic", "claude-haiku-4-5-20251001")
        if provider == "openai":
            return ("openai", "gpt-5.4-mini-2026-03-17")
        if provider == "ollama":
            result = self._check_ollama()
            if result:
                return result
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            console.print(f"[red]Cannot reach Ollama at {host}. Is `ollama serve` running?[/red]")
            sys.exit(1)
        raise ValueError(f"Unknown provider: {provider}")

    def _check_ollama(self) -> Optional[tuple[str, str]]:
        import requests as _requests
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        try:
            resp = _requests.get(f"{host}/api/tags", timeout=2)
            if resp.status_code != 200:
                return None
            tags = resp.json()
            model_names = [m.get("name", "").split(":")[0] for m in tags.get("models", [])]
            if model.split(":")[0] not in model_names:
                console.print(f"[red]Model '{model}' not found. Run: ollama pull {model}[/red]")
                sys.exit(1)
            return ("ollama", model)
        except Exception:
            return None

    def _get_anthropic_client(self):
        if _anthropic is None:
            console.print("[red]anthropic package not installed. Run: pip install anthropic[/red]")
            sys.exit(1)
        return _anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _get_openai_client(self):
        if _openai is None:
            console.print("[red]openai package not installed. Run: pip install openai[/red]")
            sys.exit(1)
        return _openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _call_llm(self, system: str, user: str, json_schema: dict | None = None) -> str:
        if self.provider == "anthropic":
            client = self._get_anthropic_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        if self.provider == "openai":
            client = self._get_openai_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=4096,
            )
            return resp.choices[0].message.content

        if self.provider == "ollama":
            import requests as _requests
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            body: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": self.temperature},
            }
            if json_schema:
                body["format"] = json_schema
            resp = _requests.post(
                f"{host}/api/chat",
                json=body,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        raise ValueError(f"Unknown provider: {self.provider}")

    @_traceable(name="analyze_commits")
    def analyze_commits(self, commits: list[dict]) -> list[ScoredCommit]:
        if not commits:
            return []

        items = "\n".join(
            f'  <commit index="{i+1}"><message>{c["message"][:500]}</message></commit>'
            for i, c in enumerate(commits)
        )
        xml_block = f"<commits>\n{items}\n</commits>"

        system = _load_prompt("analyze")
        user = f"Review these {len(commits)} commit messages:\n\n{xml_block}"

        start = time.time()
        schema = _ANALYZE_SCHEMA if self.provider == "ollama" else None
        raw = self._call_llm(system, user, json_schema=schema)
        latency_ms = int((time.time() - start) * 1000)

        parsed = self._parse_json_response(raw, len(commits))

        results = []
        for i, commit in enumerate(commits):
            p = parsed[i] if i < len(parsed) else {}
            results.append(ScoredCommit(
                hash=commit["hash"],
                message=commit["message"],
                author=commit["author"],
                date=commit["date"],
                score=p.get("score", 5),
                issue=p.get("issue", ""),
                suggestion=p.get("suggestion", ""),
                category=p.get("category", "vague"),
            ))

        self._append_metrics("analyze", len(commits), latency_ms,
                             sum(r.score for r in results) / len(results) if results else 0)
        return results

    _VALID_CATEGORIES = {"vague", "wip", "good", "excellent"}

    @staticmethod
    def _sanitize_entry(entry: dict) -> dict:
        """Clamp numeric fields and strip oversized strings to prevent injection bleed-through."""
        score = entry.get("score", 5)
        if not isinstance(score, int) or not (1 <= score <= 10):
            score = 5

        category = entry.get("category", "vague")
        if category not in LLMClient._VALID_CATEGORIES:
            category = "vague"

        judge_score = entry.get("judge_score")
        if judge_score is not None:
            if not isinstance(judge_score, int) or not (1 <= judge_score <= 5):
                judge_score = None

        fair = entry.get("fair")
        if not isinstance(fair, bool):
            fair = None

        return {
            **entry,
            "score": score,
            "category": category,
            "issue": str(entry.get("issue", ""))[:300],
            "suggestion": str(entry.get("suggestion", ""))[:300],
            "note": str(entry.get("note", ""))[:300],
            "judge_score": judge_score,
            "fair": fair,
        }

    def _parse_json_response(self, raw: str, expected_count: int) -> list[dict]:
        """Parse LLM JSON output with four fallback strategies for malformed responses."""
        text = raw.strip()

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [self._sanitize_entry(e) for e in data]
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [self._sanitize_entry(e) for e in data]
        except json.JSONDecodeError:
            pass

        # Regex: find array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [self._sanitize_entry(e) for e in data]
            except json.JSONDecodeError:
                pass

        # Fallback: individual objects
        objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        results = []
        for obj in objects:
            try:
                results.append(self._sanitize_entry(json.loads(obj)))
            except json.JSONDecodeError:
                pass
        if results:
            return results

        # Final fallback
        return [{"score": 5, "issue": "Could not parse", "suggestion": "", "category": "vague"}
                for _ in range(expected_count)]

    @_traceable(name="suggest_commit")
    def suggest_commit(self, diff: str, stat_summary: str) -> Iterator[str]:
        if len(diff) > 4000:
            diff = diff[:4000]

        system = _load_prompt("write")
        user = f"Staged changes summary:\n{stat_summary}\n\nDiff:\n{diff}\n\nRespond ONLY with the commit message. No explanation."

        yield from self._stream_llm(system, user)

    def _stream_llm(self, system: str, user: str) -> Iterator[str]:
        if self.provider == "anthropic":
            client = self._get_anthropic_client()
            with client.messages.stream(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                yield from stream.text_stream

        elif self.provider == "openai":
            client = self._get_openai_client()
            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=512,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        elif self.provider == "ollama":
            import requests as _requests
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            resp = _requests.post(
                f"{host}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    @_traceable(name="judge_commits")
    def judge_commits(self, scored: list[ScoredCommit]) -> list[ScoredCommit]:
        if not scored:
            return scored

        # Sample: highest, lowest, up to 3 borderline (4-6)
        sorted_by_score = sorted(scored, key=lambda s: s.score)
        lowest = [sorted_by_score[0]]
        highest = [sorted_by_score[-1]]
        borderline = [s for s in scored if 4 <= s.score <= 6][:3]

        seen_hashes = set()
        sample = []
        for s in lowest + borderline + highest:
            if s.hash not in seen_hashes:
                seen_hashes.add(s.hash)
                sample.append(s)
        sample = sample[:5]

        items = []
        for i, s in enumerate(sample, 1):
            items.append(
                f'  <entry index="{i}">\n'
                f'    <message>{s.message[:500]}</message>\n'
                f'    <score>{s.score}</score>\n'
                f'    <category>{s.category}</category>\n'
                f'    <issue>{s.issue[:300]}</issue>\n'
                f'    <suggestion>{s.suggestion[:300]}</suggestion>\n'
                f'  </entry>'
            )
        user = "<entries>\n" + "\n".join(items) + "\n</entries>"

        system = _load_prompt("judge")

        schema = _JUDGE_SCHEMA if self.provider == "ollama" else None
        raw = self._call_llm(system, user, json_schema=schema)
        parsed = self._parse_json_response(raw, len(sample))

        hash_to_scored = {s.hash: s for s in scored}
        for i, entry in enumerate(parsed):
            if i < len(sample):
                s = sample[i]
                if s.hash in hash_to_scored:
                    hash_to_scored[s.hash].judge_score = entry.get("judge_score")
                    hash_to_scored[s.hash].judge_note = entry.get("note", "")

        return scored

    def _append_metrics(self, mode: str, commits: int, latency_ms: int, avg_score: float) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": mode,
            "provider": self.provider,
            "model": self.model,
            "commits": commits,
            "avg_score": round(avg_score, 2),
            "latency_ms": latency_ms,
        }
        try:
            with open(".commit_critic_metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CommitAnalyzer
# ---------------------------------------------------------------------------

class CommitAnalyzer:
    """Orchestrates --analyze: fetches commits, calls LLM, caches results, renders output."""

    def __init__(self, git: GitClient, llm: LLMClient):
        self.git = git
        self.llm = llm
        self.cache_file = Path(".commit_critic_cache.json")
        self.metrics_file = Path(".commit_critic_metrics.jsonl")

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: dict) -> None:
        try:
            self.cache_file.write_text(json.dumps(cache, indent=2))
        except Exception:
            pass

    def run(self, url: Optional[str], n: int, output: str, judge: bool, quiet: bool = False) -> None:
        repo_path = None
        tmp_dir = None

        try:
            if url:
                tmp_dir = self.git.clone_remote(url, depth=n)
                repo_path = tmp_dir
            else:
                repo_path = os.getcwd()

            commits = self.git.get_commits(repo_path, n)
            if not commits:
                if not quiet:
                    console.print(f"[yellow]No commits found in the last {n} commits.[/yellow]")
                return

            cache = self._load_cache()
            cached_results: list[ScoredCommit] = []
            uncached: list[dict] = []

            for c in commits:
                if c["hash"] in cache:
                    d = cache[c["hash"]]
                    cached_results.append(ScoredCommit(
                        hash=c["hash"], message=c["message"],
                        author=c["author"], date=c["date"],
                        score=d["score"], issue=d["issue"],
                        suggestion=d["suggestion"], category=d["category"],
                    ))
                else:
                    uncached.append(c)

            new_results: list[ScoredCommit] = []
            if uncached:
                if not quiet:
                    with console.status(f"[cyan]Analyzing {len(uncached)} commits...[/cyan]"):
                        new_results = self.llm.analyze_commits(uncached)
                else:
                    new_results = self.llm.analyze_commits(uncached)

                for s in new_results:
                    cache[s.hash] = {
                        "score": s.score, "issue": s.issue,
                        "suggestion": s.suggestion, "category": s.category,
                    }
                self._save_cache(cache)

            scored = cached_results + new_results
            # Preserve original order
            hash_order = {c["hash"]: i for i, c in enumerate(commits)}
            scored.sort(key=lambda s: hash_order.get(s.hash, 9999))

            if judge:
                if not quiet:
                    with console.status("[cyan]Running judge review...[/cyan]"):
                        scored = self.llm.judge_commits(scored)
                else:
                    scored = self.llm.judge_commits(scored)

            stats = {
                "total": len(scored),
                "average_score": round(sum(s.score for s in scored) / len(scored), 1),
                "vague_count": sum(1 for s in scored if s.category == "vague"),
                "wip_count": sum(1 for s in scored if s.category == "wip"),
                "one_word_count": sum(1 for s in scored if len(s.message.split()) == 1),
                "well_written_count": sum(1 for s in scored if s.score >= 8),
            }

            bad = sorted([s for s in scored if s.score <= 5], key=lambda s: s.score)
            good = sorted([s for s in scored if s.score >= 8], key=lambda s: s.score, reverse=True)

            if output == "json":
                repo_name = url.rstrip("/").split("/")[-1] if url else Path(os.getcwd()).name
                out = {
                    "repository": repo_name,
                    "commits_analyzed": stats["total"],
                    "average_score": stats["average_score"],
                    "stats": {
                        "vague": stats["vague_count"],
                        "one_word": stats["one_word_count"],
                        "well_written": stats["well_written_count"],
                    },
                    "commits": [
                        {
                            "hash": s.hash, "message": s.message,
                            "score": s.score, "issue": s.issue,
                            "suggestion": s.suggestion, "category": s.category,
                        }
                        for s in scored
                    ],
                }
                print(json.dumps(out, indent=2))
            elif quiet:
                # Single line for post-commit hook
                if scored:
                    s = scored[0]
                    issue_short = s.issue[:60] if s.issue else s.category
                    console.print(f"[commit-critic] Score: {s.score}/10 -- {issue_short}")
            else:
                judge_summary = scored if judge else None
                Display.render_analysis(bad, good, stats, judge_summary)

        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CommitWriter
# ---------------------------------------------------------------------------

class CommitWriter:
    """Orchestrates --write: reads staged diff, streams a suggestion, prompts for acceptance."""

    def __init__(self, git: GitClient, llm: LLMClient):
        self.git = git
        self.llm = llm

    @staticmethod
    def _parse_stat(stat_output: str) -> tuple[str, list[str]]:
        """Return (formatted summary line, list of changed file paths) from git diff --stat."""
        lines = [line for line in stat_output.strip().splitlines() if line.strip()]
        if not lines:
            return "", []
        summary_raw = lines[-1].strip()
        file_paths = []
        for line in lines[:-1]:
            if "|" in line:
                file_paths.append(line.split("|")[0].strip())
        m = re.match(
            r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?",
            summary_raw,
        )
        if m:
            n_files = m.group(1)
            inserts = m.group(2) or "0"
            deletes = m.group(3) or "0"
            summary = f"{n_files} files changed, +{inserts} -{deletes} lines"
        else:
            summary = summary_raw
        return summary, file_paths

    def run(self, hook_mode: bool = False, msg_file: Optional[str] = None) -> None:
        diff, stat_summary = self.git.get_staged_diff()

        if not diff:
            if hook_mode:
                return
            console.print("[yellow]No staged changes found. Run `git add` first.[/yellow]")
            return

        if not hook_mode and stat_summary:
            summary_line, changed_files = self._parse_stat(stat_summary)
            console.print(f"\nAnalyzing staged changes... ({summary_line})")
            if changed_files:
                console.print("\nChanges detected:")
                for f in changed_files:
                    console.print(f"  - {f}")
            console.print()

        full_message = ""

        if hook_mode:
            for chunk in self.llm.suggest_commit(diff, stat_summary):
                full_message += chunk
        else:
            with Live(console=console, refresh_per_second=20) as live:
                for chunk in self.llm.suggest_commit(diff, stat_summary):
                    full_message += chunk
                    live.update(full_message)

        full_message = full_message.strip()

        if hook_mode:
            if msg_file:
                try:
                    Path(msg_file).write_text(full_message + "\n")
                except Exception:
                    pass
            return

        Display.render_write_suggestion(full_message)

        console.print("\nPress Enter to accept, or type your own message:")
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return

        final_message = full_message if user_input == "" else user_input

        console.print()
        console.print(Panel(
            final_message,
            title="[green]Final Commit Message[/green]",
            border_style="green",
        ))
        console.print("\n[dim]Run:[/dim]")
        safe = final_message.split("\n")[0].replace('"', '\\"')
        console.print(f'  [bold]git commit -m "{safe}"[/bold]')


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

BENCHMARK_COMMITS = [
    {"message": "wip",                             "expected_min": 1, "expected_max": 2, "category": "wip"},
    {"message": "fix",                             "expected_min": 1, "expected_max": 2, "category": "vague"},
    {"message": "asdf",                            "expected_min": 1, "expected_max": 2, "category": "vague"},
    {"message": "commit",                          "expected_min": 1, "expected_max": 2, "category": "vague"},
    {"message": "fixed bug",                       "expected_min": 2, "expected_max": 4, "category": "vague"},
    {"message": "fixed login bug",                 "expected_min": 2, "expected_max": 4, "category": "vague"},
    {"message": "update dependencies",             "expected_min": 3, "expected_max": 5, "category": "vague"},
    {"message": "WIP: working on auth",            "expected_min": 2, "expected_max": 4, "category": "wip"},
    {"message": "refactor auth module",            "expected_min": 5, "expected_max": 7, "category": "good"},
    {"message": "add error handling to API",       "expected_min": 5, "expected_max": 7, "category": "good"},
    {"message": "fix(auth): handle null token on refresh",
                                                   "expected_min": 7, "expected_max": 9, "category": "good"},
    {"message": "refactor(db): extract query builder\n\n- Separate concerns from model layer\n- Easier to test in isolation",
                                                   "expected_min": 7, "expected_max": 9, "category": "excellent"},
    {"message": "feat(auth): add OAuth2 login flow\n\n- Implement Google OAuth provider\n- Add callback route and session handling\n- Store tokens encrypted at rest",
                                                   "expected_min": 8, "expected_max": 10, "category": "excellent"},
    {"message": "feat(cache): add Redis caching layer\n\n- Cache read endpoints with configurable TTL\n- Add cache invalidation on write\n- Reduces p99 latency by 200ms",
                                                   "expected_min": 9, "expected_max": 10, "category": "excellent"},
    {"message": "fix(payments): prevent double-charge on network retry\n\n- Add idempotency key to Stripe calls\n- Log all retry attempts\n- Fixes #482",
                                                   "expected_min": 9, "expected_max": 10, "category": "excellent"},
]


class EvalRunner:
    """Runs the scorer against BENCHMARK_COMMITS and reports within-range accuracy and MAE."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.metrics_file = Path(".commit_critic_metrics.jsonl")

    def run(self) -> None:
        benchmark_commits = [
            {
                "hash": f"eval{i:04d}",
                "message": b["message"],
                "author": "eval",
                "date": "2026-01-01",
            }
            for i, b in enumerate(BENCHMARK_COMMITS)
        ]

        with console.status("[cyan]Running eval against benchmark...[/cyan]"):
            scored = self.llm.analyze_commits(benchmark_commits)

        results = []
        for s, b in zip(scored, BENCHMARK_COMMITS):
            in_range = b["expected_min"] <= s.score <= b["expected_max"]
            midpoint = (b["expected_min"] + b["expected_max"]) / 2
            category_correct = s.category == b["category"]
            results.append({
                "message": b["message"],
                "expected_min": b["expected_min"],
                "expected_max": b["expected_max"],
                "expected_category": b["category"],
                "score": s.score,
                "category": s.category,
                "in_range": in_range,
                "category_correct": category_correct,
                "abs_error": abs(s.score - midpoint),
            })

        within_range = sum(1 for r in results if r["in_range"])
        mae = sum(r["abs_error"] for r in results) / len(results)
        category_accuracy = sum(1 for r in results if r["category_correct"])

        # Calibrate temperature: generous model -> lower temp, strict -> higher
        avg_bias = sum(r["score"] - (r["expected_min"] + r["expected_max"]) / 2 for r in results) / len(results)
        calibrated_temp = round(max(0.0, min(0.8, LLMClient._DEFAULT_TEMP - avg_bias * 0.15)), 2)
        if self.llm.provider == "ollama":
            LLMClient._save_calibrated_temperature(calibrated_temp)

        Display.render_eval_results(results, within_range, mae, category_accuracy,
                                    self.llm.provider, self.llm.model, calibrated_temp)

        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "eval",
            "provider": self.llm.provider,
            "model": self.llm.model,
            "commits": len(BENCHMARK_COMMITS),
            "within_range_pct": round(within_range / len(BENCHMARK_COMMITS) * 100, 1),
            "mae": round(mae, 2),
            "category_accuracy_pct": round(category_accuracy / len(BENCHMARK_COMMITS) * 100, 1),
        }
        try:
            with self.metrics_file.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

class Display:
    """All terminal rendering via rich. Stateless; every method is a @staticmethod."""

    @staticmethod
    def _score_color(score: int) -> str:
        if score <= 3:
            return "red"
        if score <= 5:
            return "yellow"
        if score <= 7:
            return "blue"
        return "green"

    @staticmethod
    def render_analysis(bad, good, stats, judge_summary=None) -> None:
        console.print()
        console.print(Rule("[bold]COMMIT CRITIC[/bold]"))

        if bad:
            console.print()
            console.print(Rule("[red bold]COMMITS THAT NEED WORK[/red bold]"))
            for s in bad:
                Display.render_commit_card(s, mode="bad")

        if good:
            console.print()
            console.print(Rule("[green bold]WELL-WRITTEN COMMITS[/green bold]"))
            for s in good:
                Display.render_commit_card(s, mode="good")

        if judge_summary:
            Display.render_judge_summary(judge_summary)

        console.print()
        console.print(Rule("[bold]YOUR STATS[/bold]"))
        total = stats["total"]
        avg = stats["average_score"]
        vague = stats["vague_count"]
        wip = stats["wip_count"]
        one_word = stats["one_word_count"]
        well_written = stats["well_written_count"]

        console.print(f"Average score:       [bold]{avg}/10[/bold]")
        pct = lambda n: f"{round(n/total*100)}%" if total else "0%"
        console.print(f"Vague commits:       {vague} ({pct(vague)})")
        console.print(f"WIP commits:         {wip} ({pct(wip)})")
        console.print(f"One-word commits:    {one_word} ({pct(one_word)})")
        console.print(f"Well-written (8+):   {well_written} ({pct(well_written)})")
        console.print()

    @staticmethod
    def render_commit_card(scored: ScoredCommit, mode: str = "bad") -> None:
        color = Display._score_color(scored.score)
        short_hash = scored.hash[:7]
        console.print()
        console.print(f"  Commit: [dim]{short_hash}[/dim] [italic]\"{scored.message[:80]}\"[/italic]")
        console.print(f"  Score:  [{color}]{scored.score}/10[/{color}]  [dim]({scored.category})[/dim]")
        if mode == "bad":
            if scored.issue:
                console.print(f"  Issue:  {scored.issue}")
            if scored.suggestion:
                console.print(f"  Better: [green]{scored.suggestion}[/green]")
            if scored.judge_score is not None:
                console.print(f"  Judge:  [magenta]{scored.judge_note}[/magenta]")
        else:
            why = scored.suggestion or scored.issue
            if why:
                console.print(f"  Why it's good: {why}")

    @staticmethod
    def render_write_suggestion(message: str) -> None:
        console.print("Suggested commit message:")
        console.print(Rule())
        console.print(message)
        console.print(Rule())

    @staticmethod
    def render_judge_summary(scored: list[ScoredCommit]) -> None:
        judged = [s for s in scored if s.judge_score is not None]
        if not judged:
            return
        console.print()
        console.print(Rule("[magenta bold]JUDGE REVIEW[/magenta bold]"))
        console.print(f"Reviewed {len(judged)} scored commits")
        for s in judged:
            note = s.judge_note or ""
            console.print(f"  [yellow]>[/yellow] \"{s.message[:60]}\" scored {s.score}/10 -- {note}")
        console.print()

    @staticmethod
    def render_eval_results(results, within_range, mae, category_accuracy,
                             provider, model, calibrated_temp: Optional[float] = None) -> None:
        total = len(results)
        console.print()
        console.print(Rule(f"[bold]EVAL RESULTS[/bold]  (provider: {provider}, model: {model})"))
        console.print(f"Within expected range:  [bold]{within_range}/{total}[/bold] ({round(within_range/total*100)}%)")
        console.print(f"Mean absolute error:    [bold]{round(mae, 2)}[/bold] points")
        console.print(f"Category accuracy:      [bold]{category_accuracy}/{total}[/bold] ({round(category_accuracy/total*100)}%)")

        if calibrated_temp is not None and provider == "ollama":
            console.print(f"Calibrated temperature: [bold]{calibrated_temp}[/bold] (saved for next run)")

        missed = [r for r in results if not r["in_range"]]
        if missed:
            console.print("\nMissed:")
            for r in missed:
                msg = r["message"].split("\n")[0][:60]
                console.print(f"  \"{msg}\" -> scored {r['score']}, expected {r['expected_min']}-{r['expected_max']}")
        console.print()

    @staticmethod
    def render_hooks_installed(installed: list[str]) -> None:
        console.print()
        console.print("[bold]Installing commit-critic hooks...[/bold]")
        console.print()
        for path in installed:
            console.print(f"  [green]✓[/green] {path}")
        console.print()
        console.print("[green]Done.[/green] Your next `git commit` will auto-suggest a message.")
        console.print("To remove: python commit_critic.py --uninstall-hooks")
        console.print()

    @staticmethod
    def print_progress(msg: str) -> None:
        console.print(f"[cyan]{msg}[/cyan]")

    @staticmethod
    def print_error(msg: str) -> None:
        console.print(f"[red]{msg}[/red]")


# ---------------------------------------------------------------------------
# Hook install / uninstall
# ---------------------------------------------------------------------------

_SENTINEL = "# commit-critic:"


def _install_hooks() -> None:
    if not Path(".git").is_dir():
        console.print("[red]Not a git repository. Run from the repo root.[/red]")
        sys.exit(1)

    python_exe = str(Path(sys.executable))
    script_path = str(Path(__file__).resolve())

    hooks_dir = Path(".git/hooks")
    hooks_dir.mkdir(exist_ok=True)

    prepare_msg_hook = hooks_dir / "prepare-commit-msg"
    prepare_msg_hook.write_text(
        f"""#!/bin/sh
{_SENTINEL} auto-suggest commit message
{_SENTINEL} installed by: python commit_critic.py --install-hooks
COMMIT_SOURCE="$2"
MSG_FILE="$1"
if [ -z "$COMMIT_SOURCE" ]; then
  {python_exe} {script_path} --write --hook --msg-file="$MSG_FILE" 2>/dev/null || true
fi
"""
    )
    os.chmod(prepare_msg_hook, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)

    post_commit_hook = hooks_dir / "post-commit"
    post_commit_hook.write_text(
        f"""#!/bin/sh
{_SENTINEL} score last commit
{_SENTINEL} installed by: python commit_critic.py --install-hooks
{python_exe} {script_path} --analyze --n=1 --quiet 2>/dev/null || true
"""
    )
    os.chmod(post_commit_hook, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)

    # Claude Code hook
    claude_settings_path = Path(".claude/settings.json")
    claude_settings_path.parent.mkdir(exist_ok=True)
    settings = {}
    if claude_settings_path.exists():
        try:
            settings = json.loads(claude_settings_path.read_text())
        except Exception:
            settings = {}

    hook_command = (
        f"echo \"$CLAUDE_TOOL_OUTPUT\" | grep -qE '(master|main|HEAD|branch)' && "
        f"{python_exe} {script_path} --analyze --n=1 --quiet 2>/dev/null || true"
    )
    new_hook_entry = {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": hook_command}],
    }
    hooks_section = settings.setdefault("hooks", {})
    post_tool_use = hooks_section.setdefault("PostToolUse", [])
    # Remove any existing commit-critic entry
    post_tool_use[:] = [h for h in post_tool_use if "commit_critic" not in str(h)]
    post_tool_use.append(new_hook_entry)
    claude_settings_path.write_text(json.dumps(settings, indent=2))

    installed = [
        str(prepare_msg_hook),
        str(post_commit_hook),
        str(claude_settings_path),
    ]
    Display.render_hooks_installed(installed)


def _uninstall_hooks() -> None:
    if not Path(".git").is_dir():
        console.print("[red]Not a git repository. Run from the repo root.[/red]")
        sys.exit(1)

    removed = []

    for hook_name in ("prepare-commit-msg", "post-commit"):
        hook_path = Path(".git/hooks") / hook_name
        if hook_path.exists():
            content = hook_path.read_text()
            if _SENTINEL in content:
                hook_path.unlink()
                removed.append(str(hook_path))
            else:
                console.print(f"[yellow]Skipping {hook_path} — not installed by commit-critic[/yellow]")

    claude_settings_path = Path(".claude/settings.json")
    if claude_settings_path.exists():
        try:
            settings = json.loads(claude_settings_path.read_text())
            hooks_section = settings.get("hooks", {})
            post_tool_use = hooks_section.get("PostToolUse", [])
            new_list = [h for h in post_tool_use if "commit_critic" not in str(h)]
            if len(new_list) != len(post_tool_use):
                hooks_section["PostToolUse"] = new_list
                if not new_list:
                    del hooks_section["PostToolUse"]
                if not hooks_section:
                    del settings["hooks"]
                claude_settings_path.write_text(json.dumps(settings, indent=2))
                removed.append(str(claude_settings_path))
        except Exception as e:
            console.print(f"[yellow]Could not update {claude_settings_path}: {e}[/yellow]")

    if removed:
        console.print("[green]Removed:[/green]")
        for path in removed:
            console.print(f"  [green]✓[/green] {path}")
    else:
        console.print("[yellow]Nothing to remove.[/yellow]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--analyze",         is_flag=True,  help="Analyze commit history")
@click.option("--write",           is_flag=True,  help="Interactive commit writer")
@click.option("--url",             default=None,  help="Remote repository URL (--analyze only)")
@click.option("--n",               default=50,    help="Number of commits to analyze", show_default=True)
@click.option("--output",          default="terminal", type=click.Choice(["terminal", "json"]), help="Output format")
@click.option("--judge",           is_flag=True,  help="Run LLM-as-a-judge validation after analysis")
@click.option("--eval",            "run_eval", is_flag=True, help="Run against built-in benchmark dataset")
@click.option("--install-hooks",   is_flag=True,  help="Install git + Claude Code hooks into current repo")
@click.option("--uninstall-hooks", is_flag=True,  help="Remove previously installed hooks")
@click.option("--hook",            is_flag=True,  hidden=True)
@click.option("--msg-file",        default=None,  hidden=True)
@click.option("--quiet",           is_flag=True,  hidden=True)
def main(analyze, write, url, n, output, judge, run_eval, install_hooks, uninstall_hooks, hook, msg_file, quiet):
    try:
        if install_hooks:
            _install_hooks()
        elif uninstall_hooks:
            _uninstall_hooks()
        elif run_eval:
            llm = LLMClient()
            EvalRunner(llm).run()
        elif analyze:
            git = GitClient()
            llm = LLMClient()
            CommitAnalyzer(git, llm).run(url=url, n=n, output=output, judge=judge, quiet=quiet)
        elif write:
            git = GitClient()
            llm = LLMClient()
            CommitWriter(git, llm).run(hook_mode=hook, msg_file=msg_file)
        else:
            click.echo(click.get_current_context().get_help())
    except Exception:
        if hook or quiet:
            sys.exit(0)
        raise
    finally:
        try:
            from langsmith import Client as _LSClient
            _LSClient().flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
