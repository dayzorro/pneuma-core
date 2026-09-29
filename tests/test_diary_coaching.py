"""Tests for DiaryCoaching (Issue #89): ユーザー日記連携コーチング."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pneuma_core.llm.adapter import LLMResponse
from pneuma_core.models.message import MessageInput


# --- helpers ---


def _make_coaching(diary_dir: Path):
    """DiaryCoaching インスタンスを作成する."""
    from pneuma_core.runtime.diary_coaching import DiaryCoaching

    return DiaryCoaching(diary_dir=diary_dir)


def _write_diary(diary_dir: Path, date_str: str, content: str) -> Path:
    """テスト用日記ファイルを作成する."""
    diary_dir.mkdir(parents=True, exist_ok=True)
    path = diary_dir / f"{date_str}.md"
    path.write_text(content, encoding="utf-8")
    return path


# === should_trigger テスト ===


class TestShouldTrigger:
    """トリガーキーワード判定のテスト."""

    def test_nikki_kaita_triggers(self, tmp_path: Path) -> None:
        """「日記書いた」でトリガーされる."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("写了日记") is True

    def test_nikki_yonde_triggers(self, tmp_path: Path) -> None:
        """「日記読んで」でトリガーされる."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("看下日记") is True

    def test_nikki_mite_triggers(self, tmp_path: Path) -> None:
        """「日記見て」でトリガーされる."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("看看日记") is True

    def test_nikki_kaitayo_triggers(self, tmp_path: Path) -> None:
        """「日記書いたよ」でトリガーされる."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("日记写好了") is True

    def test_keyword_in_sentence_triggers(self, tmp_path: Path) -> None:
        """文中にキーワードが含まれていてもトリガーされる."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("今天也写了日记！") is True

    def test_no_keyword_does_not_trigger(self, tmp_path: Path) -> None:
        """キーワードがなければトリガーされない."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("你好") is False

    def test_empty_input_does_not_trigger(self, tmp_path: Path) -> None:
        """空文字列はトリガーされない."""
        coaching = _make_coaching(tmp_path / "diary")
        assert coaching.should_trigger("") is False

    def test_partial_keyword_does_not_trigger(self, tmp_path: Path) -> None:
        """キーワードの一部だけではトリガーされない."""
        coaching = _make_coaching(tmp_path / "diary")
        # 「日記」だけではトリガーされない（「日記書いた」等が必要）
        assert coaching.should_trigger("日记") is False


# === get_diary_content テスト ===


class TestGetDiaryContent:
    """日記内容取得のテスト."""

    def test_returns_specified_date_diary(self, tmp_path: Path) -> None:
        """指定日の日記がある場合にその内容を返す."""
        diary_dir = tmp_path / "diary"
        _write_diary(diary_dir, "2026-02-28", "今日は楽しい一日だった。")

        coaching = _make_coaching(diary_dir)
        content = coaching.get_diary_content(date_str="2026-02-28")
        assert content == "今日は楽しい一日だった。"

    def test_returns_none_for_missing_date(self, tmp_path: Path) -> None:
        """指定日の日記がない場合に None を返す."""
        diary_dir = tmp_path / "diary"
        diary_dir.mkdir(parents=True)

        coaching = _make_coaching(diary_dir)
        content = coaching.get_diary_content(date_str="2026-02-28")
        assert content is None

    def test_returns_today_diary_when_no_date_specified(
        self, tmp_path: Path
    ) -> None:
        """date_str が None のとき、今日の日記を返す."""
        diary_dir = tmp_path / "diary"
        today = datetime.now().strftime("%Y-%m-%d")
        _write_diary(diary_dir, today, "今日の日記の内容です。")

        coaching = _make_coaching(diary_dir)
        content = coaching.get_diary_content()
        assert content == "今日の日記の内容です。"

    def test_returns_latest_diary_when_today_missing(
        self, tmp_path: Path
    ) -> None:
        """今日の日記がない場合、最新の日記を返す."""
        diary_dir = tmp_path / "diary"
        _write_diary(diary_dir, "2026-02-26", "2日前の日記。")
        _write_diary(diary_dir, "2026-02-27", "昨日の日記。")

        coaching = _make_coaching(diary_dir)

        # 今日の日記がないことを保証するため、モックで日付を変える
        with patch("pneuma_core.runtime.diary_coaching.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 28, 12, 0, 0)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )
            content = coaching.get_diary_content()

        assert content == "昨日の日記。"

    def test_returns_none_when_no_diary_files(self, tmp_path: Path) -> None:
        """日記ファイルがひとつもない場合に None を返す."""
        diary_dir = tmp_path / "diary"
        diary_dir.mkdir(parents=True)

        coaching = _make_coaching(diary_dir)

        with patch("pneuma_core.runtime.diary_coaching.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 28, 12, 0, 0)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )
            content = coaching.get_diary_content()

        assert content is None

    def test_returns_none_when_diary_dir_not_exists(
        self, tmp_path: Path
    ) -> None:
        """diary_dir が存在しない場合に None を返す."""
        diary_dir = tmp_path / "nonexistent_diary"

        coaching = _make_coaching(diary_dir)
        content = coaching.get_diary_content(date_str="2026-02-28")
        assert content is None


# === build_coaching_context テスト ===


class TestBuildCoachingContext:
    """コーチングコンテキスト構築のテスト."""

    def test_returns_context_when_triggered_and_diary_exists(
        self, tmp_path: Path
    ) -> None:
        """トリガー + 日記ありでコーチングコンテキストを返す."""
        diary_dir = tmp_path / "diary"
        today = datetime.now().strftime("%Y-%m-%d")
        _write_diary(diary_dir, today, "今日は新しいことに挑戦した。")

        coaching = _make_coaching(diary_dir)
        context = coaching.build_coaching_context("写了日记")

        assert context is not None
        assert "今日は新しいことに挑戦した。" in context
        assert "[用户的日记]" in context
        assert "[教练指南]" in context

    def test_returns_none_when_not_triggered(self, tmp_path: Path) -> None:
        """トリガーなしで None を返す."""
        diary_dir = tmp_path / "diary"
        today = datetime.now().strftime("%Y-%m-%d")
        _write_diary(diary_dir, today, "今日の日記。")

        coaching = _make_coaching(diary_dir)
        context = coaching.build_coaching_context("你好")
        assert context is None

    def test_returns_none_when_no_diary(self, tmp_path: Path) -> None:
        """トリガーあり + 日記なしで None を返す."""
        diary_dir = tmp_path / "diary"
        diary_dir.mkdir(parents=True)

        coaching = _make_coaching(diary_dir)

        with patch("pneuma_core.runtime.diary_coaching.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 28, 12, 0, 0)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )
            context = coaching.build_coaching_context("看下日记")

        assert context is None

    def test_context_contains_coaching_guidelines(
        self, tmp_path: Path
    ) -> None:
        """コンテキストにコーチングガイドラインが含まれる."""
        diary_dir = tmp_path / "diary"
        today = datetime.now().strftime("%Y-%m-%d")
        _write_diary(diary_dir, today, "テスト日記")

        coaching = _make_coaching(diary_dir)
        context = coaching.build_coaching_context("看看日记")

        assert context is not None
        assert "提问" in context
        assert "新的视角" in context
        assert "压迫感" in context
        assert "保持角色的口吻" in context


# === RuntimeEngine 組み込みテスト ===


class TestDiaryCoachingEngineIntegration:
    """DiaryCoaching standalone test (engine integration removed in #145).

    DiaryCoaching is now used via middleware, not direct engine injection.
    """

    def test_diary_coaching_is_independent_class(self) -> None:
        """DiaryCoaching はスタンドアロンで使える."""
        from pneuma_core.runtime.diary_coaching import DiaryCoaching
        assert DiaryCoaching is not None
