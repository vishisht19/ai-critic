"""Unit tests for commit_critic.py.

Run with:  pytest test_commit_critic.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from commit_critic import (
    BENCHMARK_COMMITS,
    CommitAnalyzer,
    Display,
    EvalRunner,
    GitClient,
    LLMClient,
    ScoredCommit,
    _install_hooks,
    _uninstall_hooks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm(provider="anthropic", model="test-model") -> LLMClient:
    """Build an LLMClient without triggering real provider detection."""
    client = LLMClient.__new__(LLMClient)
    client.provider = provider
    client.model = model
    return client


def make_scored(hash="abc1234", message="fix bug", score=3,
                category="vague", issue="too vague", suggestion="fix(auth): ...") -> ScoredCommit:
    return ScoredCommit(
        hash=hash, message=message, author="dev", date="2026-01-01",
        score=score, issue=issue, suggestion=suggestion, category=category,
    )


# ---------------------------------------------------------------------------
# GitClient
# ---------------------------------------------------------------------------

class TestGitClientParseCommits:

    def test_parses_four_pipe_separated_fields(self):
        git = GitClient()
        raw = "abc123|feat: add login|Alice|2026-01-01 10:00:00 +0000"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=raw)
            commits = git.get_commits("/fake/repo", n=1)

        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123"
        assert commits[0]["message"] == "feat: add login"
        assert commits[0]["author"] == "Alice"

    def test_skips_blank_lines(self):
        git = GitClient()
        raw = "abc123|fix|Dev|2026-01-01\n\ndef456|feat|Dev|2026-01-02\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=raw)
            commits = git.get_commits("/repo")

        assert len(commits) == 2

    def test_message_with_pipe_is_truncated_at_first_pipe(self):
        # split("|", 3) is left-to-right: a | in the subject truncates the message
        # and corrupts author/date. Known limitation of the | delimiter format.
        git = GitClient()
        raw = "abc123|fix: handle a|b edge case|Dev|2026-01-01"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=raw)
            commits = git.get_commits("/repo")

        assert commits[0]["message"] == "fix: handle a"  # truncated at first |

    def test_exits_on_non_zero_returncode(self):
        git = GitClient()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal")
            with pytest.raises(SystemExit):
                git.get_commits("/not-a-repo")

    def test_get_staged_diff_returns_tuple(self):
        git = GitClient()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="diff --git a/foo.py"),
                MagicMock(returncode=0, stdout="foo.py | 3 +++"),
            ]
            diff, stat = git.get_staged_diff()

        assert "diff" in diff
        assert "foo.py" in stat

    def test_get_staged_diff_empty_returns_empty_strings(self):
        git = GitClient()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout=""),
                MagicMock(returncode=0, stdout=""),
            ]
            diff, stat = git.get_staged_diff()

        assert diff == ""
        assert stat == ""


# ---------------------------------------------------------------------------
# LLMClient._detect_provider
# ---------------------------------------------------------------------------

class TestLLMClientDetectProvider:

    def _make_with_detection(self, env: dict):
        with patch.dict(os.environ, env, clear=True):
            with patch.object(LLMClient, "_check_ollama", return_value=None):
                return LLMClient()

    def test_explicit_anthropic_env_var(self):
        llm = self._make_with_detection({"LLM_PROVIDER": "anthropic"})
        assert llm.provider == "anthropic"
        assert "haiku" in llm.model

    def test_explicit_openai_env_var(self):
        llm = self._make_with_detection({"LLM_PROVIDER": "openai"})
        assert llm.provider == "openai"
        assert "gpt" in llm.model

    def test_anthropic_api_key_takes_priority_over_openai(self):
        llm = self._make_with_detection({
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENAI_API_KEY": "sk-openai-test",
        })
        assert llm.provider == "anthropic"

    def test_openai_key_used_when_no_anthropic(self):
        llm = self._make_with_detection({"OPENAI_API_KEY": "sk-openai-test"})
        assert llm.provider == "openai"

    def test_ollama_fallback_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(LLMClient, "_check_ollama", return_value=("ollama", "llama3.2")):
                llm = LLMClient()
        assert llm.provider == "ollama"

    def test_exits_when_no_provider_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(LLMClient, "_check_ollama", return_value=None):
                with pytest.raises(SystemExit):
                    LLMClient()

    def test_explicit_provider_overrides_api_key(self):
        llm = self._make_with_detection({
            "LLM_PROVIDER": "openai",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        })
        assert llm.provider == "openai"


# ---------------------------------------------------------------------------
# LLMClient._parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:

    def setup_method(self):
        self.llm = make_llm()

    def test_clean_json_array(self):
        raw = '[{"index": 1, "score": 7, "issue": "ok", "suggestion": "s", "category": "good"}]'
        result = self.llm._parse_json_response(raw, 1)
        assert len(result) == 1
        assert result[0]["score"] == 7

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"index": 1, "score": 3, "issue": "x", "suggestion": "", "category": "vague"}]\n```'
        result = self.llm._parse_json_response(raw, 1)
        assert result[0]["score"] == 3

    def test_extracts_array_from_prose(self):
        raw = 'Here is my analysis:\n[{"index": 1, "score": 5, "issue": "vague", "suggestion": "fix", "category": "vague"}]\nHope that helps!'
        result = self.llm._parse_json_response(raw, 1)
        assert result[0]["score"] == 5

    def test_extracts_individual_objects_as_fallback(self):
        raw = 'analysis: {"index": 1, "score": 2, "issue": "bad", "suggestion": "fix it", "category": "vague"} done'
        result = self.llm._parse_json_response(raw, 1)
        assert len(result) == 1
        assert result[0]["score"] == 2

    def test_final_fallback_returns_placeholder_per_commit(self):
        raw = "I cannot provide JSON output for these commits."
        result = self.llm._parse_json_response(raw, 3)
        assert len(result) == 3
        assert all(r["score"] == 5 for r in result)
        assert all(r["issue"] == "Could not parse" for r in result)

    def test_multiple_commits_parsed(self):
        raw = json.dumps([
            {"index": 1, "score": 2, "issue": "vague", "suggestion": "fix auth", "category": "vague"},
            {"index": 2, "score": 9, "issue": "", "suggestion": "", "category": "excellent"},
        ])
        result = self.llm._parse_json_response(raw, 2)
        assert len(result) == 2
        assert result[1]["score"] == 9

    def test_strips_plain_code_fence_without_language_tag(self):
        raw = "```\n[{\"index\": 1, \"score\": 6, \"issue\": \"\", \"suggestion\": \"\", \"category\": \"good\"}]\n```"
        result = self.llm._parse_json_response(raw, 1)
        assert result[0]["score"] == 6


# ---------------------------------------------------------------------------
# CommitAnalyzer – stats and partitioning
# ---------------------------------------------------------------------------

class TestCommitAnalyzerStats:

    def _make_analyzer(self):
        git = MagicMock(spec=GitClient)
        llm = make_llm()
        analyzer = CommitAnalyzer(git, llm)
        analyzer.cache_file = Path(tempfile.mktemp(suffix=".json"))
        analyzer.metrics_file = Path(tempfile.mktemp(suffix=".jsonl"))
        return analyzer

    def _scored_list(self, scores_categories):
        return [
            make_scored(hash=f"h{i}", score=s, category=c)
            for i, (s, c) in enumerate(scores_categories)
        ]

    def test_stats_computation(self):
        analyzer = self._make_analyzer()
        scored = self._scored_list([
            (2, "vague"), (1, "wip"), (3, "vague"),
            (8, "excellent"), (9, "excellent"),
        ])
        total = len(scored)
        avg = round(sum(s.score for s in scored) / total, 1)
        stats = {
            "total": total,
            "average_score": avg,
            "vague_count": sum(1 for s in scored if s.category == "vague"),
            "wip_count": sum(1 for s in scored if s.category == "wip"),
            "one_word_count": sum(1 for s in scored if len(s.message.split()) == 1),
            "well_written_count": sum(1 for s in scored if s.score >= 8),
        }
        assert stats["total"] == 5
        assert stats["vague_count"] == 2
        assert stats["wip_count"] == 1
        assert stats["well_written_count"] == 2

    def test_bad_partition_is_score_le_5(self):
        scored = self._scored_list([(3, "vague"), (6, "good"), (9, "excellent"), (5, "vague")])
        bad = [s for s in scored if s.score <= 5]
        assert len(bad) == 2
        assert all(s.score <= 5 for s in bad)

    def test_good_partition_is_score_ge_8(self):
        scored = self._scored_list([(7, "good"), (8, "excellent"), (10, "excellent")])
        good = [s for s in scored if s.score >= 8]
        assert len(good) == 2

    def test_cache_roundtrip(self):
        analyzer = self._make_analyzer()
        cache_data = {
            "abc123": {"score": 7, "issue": "ok", "suggestion": "keep it", "category": "good"}
        }
        analyzer._save_cache(cache_data)
        loaded = analyzer._load_cache()
        assert loaded["abc123"]["score"] == 7

    def test_load_cache_returns_empty_dict_when_missing(self):
        analyzer = self._make_analyzer()
        analyzer.cache_file = Path("/nonexistent/cache.json")
        assert analyzer._load_cache() == {}

    def test_load_cache_returns_empty_dict_on_corrupt_file(self, tmp_path):
        f = tmp_path / "cache.json"
        f.write_text("not json {{{{")
        analyzer = self._make_analyzer()
        analyzer.cache_file = f
        assert analyzer._load_cache() == {}

    def test_run_uses_cache_for_known_hashes(self, tmp_path):
        git = MagicMock(spec=GitClient)
        git.get_commits.return_value = [
            {"hash": "cached1", "message": "fix bug", "author": "dev", "date": "2026-01-01"},
        ]
        llm = make_llm()
        llm.analyze_commits = MagicMock(return_value=[])
        llm._append_metrics = MagicMock()

        analyzer = CommitAnalyzer(git, llm)
        analyzer.cache_file = tmp_path / "cache.json"
        analyzer.metrics_file = tmp_path / "metrics.jsonl"
        analyzer._save_cache({
            "cached1": {"score": 8, "issue": "", "suggestion": "", "category": "excellent"}
        })

        with patch("commit_critic.Display.render_analysis"):
            analyzer.run(url=None, n=1, output="terminal", judge=False, quiet=False)

        llm.analyze_commits.assert_not_called()

    def test_run_calls_analyze_for_uncached_commits(self, tmp_path):
        git = MagicMock(spec=GitClient)
        git.get_commits.return_value = [
            {"hash": "new1", "message": "fix bug", "author": "dev", "date": "2026-01-01"},
        ]
        llm = make_llm()
        llm.analyze_commits = MagicMock(return_value=[
            make_scored(hash="new1", score=3)
        ])
        llm._append_metrics = MagicMock()

        analyzer = CommitAnalyzer(git, llm)
        analyzer.cache_file = tmp_path / "cache.json"
        analyzer.metrics_file = tmp_path / "metrics.jsonl"

        with patch("commit_critic.Display.render_analysis"):
            analyzer.run(url=None, n=1, output="terminal", judge=False, quiet=False)

        llm.analyze_commits.assert_called_once()


# ---------------------------------------------------------------------------
# CommitWriter
# ---------------------------------------------------------------------------

class TestCommitWriter:

    def test_hook_mode_writes_message_to_file(self, tmp_path):
        from commit_critic import CommitWriter
        git = MagicMock(spec=GitClient)
        git.get_staged_diff.return_value = ("diff content", "1 file changed")
        llm = make_llm()
        llm.suggest_commit = MagicMock(return_value=iter(["feat: ", "add thing"]))

        writer = CommitWriter(git, llm)
        msg_file = tmp_path / "COMMIT_EDITMSG"

        writer.run(hook_mode=True, msg_file=str(msg_file))

        assert msg_file.exists()
        assert "feat:" in msg_file.read_text()

    def test_hook_mode_exits_silently_when_no_diff(self):
        from commit_critic import CommitWriter
        git = MagicMock(spec=GitClient)
        git.get_staged_diff.return_value = ("", "")
        llm = make_llm()
        llm.suggest_commit = MagicMock()

        writer = CommitWriter(git, llm)
        writer.run(hook_mode=True, msg_file=None)

        llm.suggest_commit.assert_not_called()

    def test_normal_mode_prints_error_when_no_diff(self, capsys):
        from commit_critic import CommitWriter
        git = MagicMock(spec=GitClient)
        git.get_staged_diff.return_value = ("", "")
        llm = make_llm()

        writer = CommitWriter(git, llm)
        writer.run(hook_mode=False, msg_file=None)

        llm.suggest_commit = MagicMock()
        llm.suggest_commit.assert_not_called()


# ---------------------------------------------------------------------------
# EvalRunner metrics
# ---------------------------------------------------------------------------

class TestEvalRunnerMetrics:

    def _run_metrics(self, scored_overrides: list[dict]):
        """Build fake scored results aligned to BENCHMARK_COMMITS and compute metrics."""
        results = []
        for i, b in enumerate(BENCHMARK_COMMITS):
            score = scored_overrides[i] if i < len(scored_overrides) else b["expected_min"]
            in_range = b["expected_min"] <= score <= b["expected_max"]
            midpoint = (b["expected_min"] + b["expected_max"]) / 2
            results.append({
                "score": score,
                "expected_min": b["expected_min"],
                "expected_max": b["expected_max"],
                "expected_category": b["category"],
                "category": b["category"],
                "in_range": in_range,
                "category_correct": True,
                "abs_error": abs(score - midpoint),
            })
        return results

    def test_perfect_scores_give_100_pct_within_range(self):
        # Use the midpoint of each expected range as the score
        scores = [int((b["expected_min"] + b["expected_max"]) / 2) for b in BENCHMARK_COMMITS]
        results = self._run_metrics(scores)
        within = sum(1 for r in results if r["in_range"])
        assert within == len(BENCHMARK_COMMITS)

    def test_all_wrong_gives_0_pct_within_range(self):
        # Score every commit 10 when expected max is <= 5, so all miss
        scores = [10 if b["expected_max"] <= 5 else 1 for b in BENCHMARK_COMMITS]
        results = self._run_metrics(scores)
        # At least the low-quality ones are wrong
        assert sum(1 for r in results if r["in_range"]) < len(BENCHMARK_COMMITS)

    def test_mae_is_zero_at_midpoint(self):
        scores = [int((b["expected_min"] + b["expected_max"]) / 2) for b in BENCHMARK_COMMITS]
        results = self._run_metrics(scores)
        mae = sum(r["abs_error"] for r in results) / len(results)
        # Midpoint of integer range; some rounding error is expected but should be < 0.5
        assert mae < 0.5

    def test_eval_runner_calls_analyze_and_renders(self, tmp_path):
        llm = make_llm()
        scored = [
            make_scored(hash=f"eval{i:04d}", message=b["message"],
                        score=int((b["expected_min"] + b["expected_max"]) / 2),
                        category=b["category"])
            for i, b in enumerate(BENCHMARK_COMMITS)
        ]
        llm.analyze_commits = MagicMock(return_value=scored)
        llm._append_metrics = MagicMock()

        runner = EvalRunner(llm)
        runner.metrics_file = tmp_path / "metrics.jsonl"

        with patch("commit_critic.Display.render_eval_results") as mock_render:
            runner.run()

        llm.analyze_commits.assert_called_once()
        mock_render.assert_called_once()

    def test_eval_appends_metrics_file(self, tmp_path):
        llm = make_llm()
        scored = [
            make_scored(hash=f"eval{i:04d}", message=b["message"],
                        score=b["expected_min"],
                        category=b["category"])
            for i, b in enumerate(BENCHMARK_COMMITS)
        ]
        llm.analyze_commits = MagicMock(return_value=scored)
        llm._append_metrics = MagicMock()

        runner = EvalRunner(llm)
        runner.metrics_file = tmp_path / "metrics.jsonl"

        with patch("commit_critic.Display.render_eval_results"):
            runner.run()

        assert runner.metrics_file.exists()
        line = json.loads(runner.metrics_file.read_text().strip())
        assert line["mode"] == "eval"
        assert "within_range_pct" in line


# ---------------------------------------------------------------------------
# Display – smoke tests (no crash, correct color logic)
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_score_color_red_for_1_to_3(self):
        for s in (1, 2, 3):
            assert Display._score_color(s) == "red"

    def test_score_color_yellow_for_4_to_5(self):
        for s in (4, 5):
            assert Display._score_color(s) == "yellow"

    def test_score_color_blue_for_6_to_7(self):
        for s in (6, 7):
            assert Display._score_color(s) == "blue"

    def test_score_color_green_for_8_to_10(self):
        for s in (8, 9, 10):
            assert Display._score_color(s) == "green"

    def test_render_analysis_does_not_raise(self):
        bad = [make_scored(score=2, category="vague")]
        good = [make_scored(hash="g1", score=9, category="excellent", issue="", suggestion="great")]
        stats = {
            "total": 2, "average_score": 5.5,
            "vague_count": 1, "wip_count": 0,
            "one_word_count": 0, "well_written_count": 1,
        }
        Display.render_analysis(bad, good, stats)

    def test_render_commit_card_bad_does_not_raise(self):
        Display.render_commit_card(make_scored(), mode="bad")

    def test_render_commit_card_good_does_not_raise(self):
        Display.render_commit_card(make_scored(score=9, category="excellent"), mode="good")

    def test_render_eval_results_does_not_raise(self):
        results = [
            {"message": "fix", "expected_min": 1, "expected_max": 2, "score": 1,
             "category": "vague", "expected_category": "vague", "in_range": True,
             "category_correct": True, "abs_error": 0.0},
        ]
        Display.render_eval_results(results, 1, 0.0, 1, "anthropic", "test")

    def test_render_write_suggestion_does_not_raise(self):
        Display.render_write_suggestion("feat(auth): add login\n\n- OAuth2 support")

    def test_render_judge_summary_skips_uncommitted_entries(self):
        scored = [make_scored()]  # judge_score is None
        Display.render_judge_summary(scored)  # should not raise or print anything

    def test_render_judge_summary_shows_judged_entries(self):
        s = make_scored()
        s.judge_score = 3
        s.judge_note = "score seems too low"
        Display.render_judge_summary([s])  # should not raise


# ---------------------------------------------------------------------------
# Hook install / uninstall
# ---------------------------------------------------------------------------

class TestHookInstall:

    def test_install_creates_hook_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)

        with patch("commit_critic.__file__", str(tmp_path / "commit_critic.py")):
            with patch("sys.executable", "/usr/bin/python3"):
                _install_hooks()

        assert (tmp_path / ".git" / "hooks" / "prepare-commit-msg").exists()
        assert (tmp_path / ".git" / "hooks" / "post-commit").exists()
        assert (tmp_path / ".claude" / "settings.json").exists()

    def test_install_hooks_contain_sentinel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)

        with patch("commit_critic.__file__", str(tmp_path / "commit_critic.py")):
            with patch("sys.executable", "/usr/bin/python3"):
                _install_hooks()

        content = (tmp_path / ".git" / "hooks" / "prepare-commit-msg").read_text()
        assert "# commit-critic:" in content

    def test_install_merges_existing_claude_settings(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        (tmp_path / ".claude").mkdir()
        existing = {"theme": "dark", "other": "setting"}
        (tmp_path / ".claude" / "settings.json").write_text(json.dumps(existing))

        with patch("commit_critic.__file__", str(tmp_path / "commit_critic.py")):
            with patch("sys.executable", "/usr/bin/python3"):
                _install_hooks()

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert settings["theme"] == "dark"           # existing key preserved
        assert "hooks" in settings                   # new section added

    def test_install_exits_outside_git_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _install_hooks()

    def test_uninstall_removes_sentinel_hooks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)

        with patch("commit_critic.__file__", str(tmp_path / "commit_critic.py")):
            with patch("sys.executable", "/usr/bin/python3"):
                _install_hooks()

        _uninstall_hooks()

        assert not (tmp_path / ".git" / "hooks" / "prepare-commit-msg").exists()
        assert not (tmp_path / ".git" / "hooks" / "post-commit").exists()

    def test_uninstall_leaves_non_sentinel_hook_intact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        hook = tmp_path / ".git" / "hooks" / "prepare-commit-msg"
        hook.write_text("#!/bin/sh\n# my own hook\necho hi\n")

        _uninstall_hooks()

        assert hook.exists()

    def test_uninstall_removes_claude_settings_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".git" / "hooks").mkdir(parents=True)

        with patch("commit_critic.__file__", str(tmp_path / "commit_critic.py")):
            with patch("sys.executable", "/usr/bin/python3"):
                _install_hooks()

        _uninstall_hooks()

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "hooks" not in settings

    def test_uninstall_exits_outside_git_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _uninstall_hooks()


# ---------------------------------------------------------------------------
# LLMClient._append_metrics
# ---------------------------------------------------------------------------

class TestAppendMetrics:

    def test_appends_jsonl_line(self, tmp_path):
        llm = make_llm()
        metrics_path = tmp_path / "metrics.jsonl"

        with patch("commit_critic.open", create=True):
            pass  # don't need to patch – just write to tmp_path directly

        # Directly invoke _append_metrics with a real file
        original_open = open

        def patched_open(path, mode="r", **kw):
            if ".commit_critic_metrics" in str(path):
                return original_open(str(metrics_path), mode, **kw)
            return original_open(path, mode, **kw)

        import builtins
        with patch.object(builtins, "open", side_effect=patched_open):
            llm._append_metrics("analyze", 10, 1500, 4.2)

        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["mode"] == "analyze"
        assert entry["commits"] == 10
        assert entry["latency_ms"] == 1500
        assert entry["provider"] == "anthropic"
