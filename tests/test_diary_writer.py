"""Tests for DiaryWriter (Issue #88)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pneuma_core.llm.adapter import LLMRequest, LLMResponse
from pneuma_core.runtime.diary_writer import DiaryWriter, JST


# --- helpers ---

def _make_llm() -> AsyncMock:
    """LLMAdapter のモックを作成する."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=LLMResponse(
        content="今日はいい日だった。ユーザーさんと楽しくおしゃべりできた。",
        model="test-model",
        usage={},
    ))
    return llm


def _make_diary_writer(
    llm: AsyncMock | None = None,
    logs_dir: Path | None = None,
    diary_dir: Path | None = None,
    character_name: str = "ミラ",
    character_profile: str = "明るくて元気な AI キャラクター",
    model: str = "claude-opus-4-6",
) -> DiaryWriter:
    return DiaryWriter(
        llm=llm or _make_llm(),
        logs_dir=logs_dir or Path("/tmp/test_logs"),
        diary_dir=diary_dir or Path("/tmp/test_diary"),
        character_name=character_name,
        character_profile=character_profile,
        model=model,
    )


# --- should_trigger tests ---

class TestShouldTrigger:
    """トリガーキーワード判定のテスト."""

    def test_oyasumi_triggers(self) -> None:
        """「おやすみ」でトリガーされる."""
        writer = _make_diary_writer()
        assert writer.should_trigger("晚安") is True

    def test_ohayo_triggers(self) -> None:
        """「おはよう」でトリガーされる."""
        writer = _make_diary_writer()
        assert writer.should_trigger("早安") is True

    def test_oyasuminasai_triggers(self) -> None:
        """「おやすみなさい」でトリガーされる."""
        writer = _make_diary_writer()
        assert writer.should_trigger("晚安，我去睡啦") is True

    def test_ohayou_gozaimasu_triggers(self) -> None:
        """「おはようございます」でトリガーされる."""
        writer = _make_diary_writer()
        assert writer.should_trigger("早上好呀") is True

    def test_keyword_in_sentence_triggers(self) -> None:
        """文中にキーワードが含まれていてもトリガーされる."""
        writer = _make_diary_writer()
        assert writer.should_trigger("今天有点累，晚安〜") is True

    def test_no_keyword_does_not_trigger(self) -> None:
        """キーワードがなければトリガーされない."""
        writer = _make_diary_writer()
        assert writer.should_trigger("你好") is False

    def test_empty_input_does_not_trigger(self) -> None:
        """空文字列はトリガーされない."""
        writer = _make_diary_writer()
        assert writer.should_trigger("") is False

    def test_similar_but_different_word_does_not_trigger(self) -> None:
        """似ているが異なる語はトリガーされない."""
        writer = _make_diary_writer()
        assert writer.should_trigger("晚安啊") is True  # 包含「晚安」所以为 True
        assert writer.should_trigger("晚上好") is False  # 不包含「晚安」


# --- _get_effective_date tests (date boundary fix) ---

class TestGetEffectiveDate:
    """深夜帯（0:00-3:59）は前日として扱う日付境界テスト."""

    def test_before_4am_returns_previous_day(self) -> None:
        """3:59 AM JST → 前日の日付を返す."""
        writer = _make_diary_writer()
        # 2026-03-01 03:59 JST
        fake_now = datetime(2026, 3, 1, 3, 59, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = writer._get_effective_date()
        assert result == "2026-02-28"

    def test_at_4am_returns_current_day(self) -> None:
        """4:00 AM JST → 当日の日付を返す."""
        writer = _make_diary_writer()
        # 2026-03-01 04:00 JST
        fake_now = datetime(2026, 3, 1, 4, 0, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = writer._get_effective_date()
        assert result == "2026-03-01"

    def test_at_11pm_returns_current_day(self) -> None:
        """23:00 JST → 当日の日付を返す."""
        writer = _make_diary_writer()
        # 2026-02-28 23:00 JST
        fake_now = datetime(2026, 2, 28, 23, 0, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = writer._get_effective_date()
        assert result == "2026-02-28"

    def test_at_midnight_returns_previous_day(self) -> None:
        """0:00 AM JST → 前日の日付を返す."""
        writer = _make_diary_writer()
        # 2026-03-01 00:00 JST
        fake_now = datetime(2026, 3, 1, 0, 0, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = writer._get_effective_date()
        assert result == "2026-02-28"

    def test_at_030am_returns_previous_day(self) -> None:
        """0:30 AM JST（おやすみの典型的な時間）→ 前日の日付を返す."""
        writer = _make_diary_writer()
        # 2026-03-01 00:30 JST
        fake_now = datetime(2026, 3, 1, 0, 30, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = writer._get_effective_date()
        assert result == "2026-02-28"


class TestMaybeGenerateWithEffectiveDate:
    """maybe_generate が effective date を使うことのテスト."""

    @pytest.fixture()
    def logs_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "logs"
        d.mkdir()
        return d

    @pytest.fixture()
    def diary_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "diary"
        d.mkdir()
        return d

    @pytest.mark.asyncio()
    async def test_oyasumi_after_midnight_uses_previous_day(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """深夜0:30に「おやすみ」→ 前日の会話ログを読み、前日の日記として保存."""
        # 2/28 の会話ログを用意
        log_file = logs_dir / "2026-02-28.md"
        log_file.write_text("User: 今日は楽しかったね\nミラ: うん！", encoding="utf-8")

        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        # 3/1 00:30 JST に「おやすみ」
        fake_now = datetime(2026, 3, 1, 0, 30, 0, tzinfo=JST)
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            await writer.maybe_generate("晚安")

        # 2/28 の日記として保存されること
        diary_file = diary_dir / "2026-02-28.md"
        assert diary_file.exists()
        # 3/1 の日記は存在しないこと
        assert not (diary_dir / "2026-03-01.md").exists()

        # LLM に渡されたプロンプトに 2/28 のログが含まれること
        call_args = llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert "今日は楽しかったね" in request.messages[0]["content"]


# --- maybe_generate tests ---

class TestMaybeGenerate:
    """maybe_generate のテスト（1日1回制限含む）."""

    @pytest.fixture()
    def logs_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "logs"
        d.mkdir()
        return d

    @pytest.fixture()
    def diary_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "diary"
        d.mkdir()
        return d

    @pytest.mark.asyncio()
    async def test_generates_when_triggered(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """トリガーキーワードがあれば日記を生成する."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer.maybe_generate("晚安，我睡了")

        # LLM が呼ばれたことを確認
        assert llm.generate.call_count == 1

    @pytest.mark.asyncio()
    async def test_does_not_generate_without_trigger(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """トリガーキーワードがなければ生成しない."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer.maybe_generate("你好")

        assert llm.generate.call_count == 0

    @pytest.mark.asyncio()
    async def test_once_per_day_limit(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """同じ日に2回目は生成しない（1日1回制限）."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer.maybe_generate("晚安")
        await writer.maybe_generate("早安")

        # 1回目だけ生成される
        assert llm.generate.call_count == 1

    @pytest.mark.asyncio()
    async def test_different_days_allow_generation(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """日付が異なれば再度生成できる."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        # 1日目
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 28, 23, 0, 0)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            await writer.maybe_generate("晚安")

        # 2日目
        with patch("pneuma_core.runtime.diary_writer.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 1, 7, 0, 0)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            await writer.maybe_generate("早安")

        assert llm.generate.call_count == 2


# --- _generate_diary tests ---

class TestGenerateDiary:
    """_generate_diary のファイル出力テスト."""

    @pytest.fixture()
    def logs_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "logs"
        d.mkdir()
        return d

    @pytest.fixture()
    def diary_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "diary"
        # diary_dir は _generate_diary 内で作成されるので、ここでは作らない
        return d

    @pytest.mark.asyncio()
    async def test_saves_diary_file(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """日記ファイルが正しく保存される."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer._generate_diary("2026-02-28")

        diary_file = diary_dir / "2026-02-28.md"
        assert diary_file.exists()
        content = diary_file.read_text(encoding="utf-8")
        assert "今日はいい日だった" in content

    @pytest.mark.asyncio()
    async def test_creates_diary_dir_if_not_exists(
        self, logs_dir: Path, tmp_path: Path
    ) -> None:
        """diary_dir が存在しない場合に自動作成する."""
        diary_dir = tmp_path / "nested" / "diary"
        assert not diary_dir.exists()

        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer._generate_diary("2026-02-28")

        assert diary_dir.exists()
        assert (diary_dir / "2026-02-28.md").exists()

    @pytest.mark.asyncio()
    async def test_reads_log_file_when_exists(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """会話ログファイルが存在する場合に読み込んでプロンプトに含める."""
        # ログファイルを作成
        log_file = logs_dir / "2026-02-28.md"
        log_file.write_text("User: こんにちは\nミラ: やっほー！", encoding="utf-8")

        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer._generate_diary("2026-02-28")

        # LLM に渡されたリクエストにログ内容が含まれていることを確認
        call_args = llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert "こんにちは" in request.messages[0]["content"]
        assert "やっほー" in request.messages[0]["content"]

    @pytest.mark.asyncio()
    async def test_handles_missing_log_file(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """会話ログファイルが存在しない場合でもエラーにならない."""
        llm = _make_llm()
        writer = _make_diary_writer(llm=llm, logs_dir=logs_dir, diary_dir=diary_dir)

        await writer._generate_diary("2026-02-28")

        # LLM に渡されたリクエストに「今天没有对话」が含まれる
        call_args = llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert "今天没有对话" in request.messages[0]["content"]

    @pytest.mark.asyncio()
    async def test_system_prompt_contains_character_info(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """システムプロンプトにキャラクター名とプロファイルが含まれる."""
        llm = _make_llm()
        writer = _make_diary_writer(
            llm=llm,
            logs_dir=logs_dir,
            diary_dir=diary_dir,
            character_name="ミラ",
            character_profile="明るくて元気な AI キャラクター",
        )

        await writer._generate_diary("2026-02-28")

        call_args = llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert "ミラ" in request.system_prompt
        assert "明るくて元気な AI キャラクター" in request.system_prompt

    @pytest.mark.asyncio()
    async def test_uses_configured_model(
        self, logs_dir: Path, diary_dir: Path
    ) -> None:
        """設定されたモデルが LLMRequest に渡される."""
        llm = _make_llm()
        writer = _make_diary_writer(
            llm=llm,
            logs_dir=logs_dir,
            diary_dir=diary_dir,
            model="claude-sonnet-4-20250514",
        )

        await writer._generate_diary("2026-02-28")

        call_args = llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert request.model == "claude-sonnet-4-20250514"


# --- RuntimeEngine integration tests ---
# Note: DiaryWriter engine integration removed in #145 (middleware refactor).
# DiaryWriter is now used via middleware, not direct engine injection.
# Engine integration tests moved to tests/test_middleware.py.


class TestDiaryWriterEngineIntegration:
    """DiaryWriter standalone test (engine integration removed in #145).

    DiaryWriter is now used via middleware, not direct engine injection.
    """

    def test_diary_writer_is_independent_class(self) -> None:
        """DiaryWriter はスタンドアロンで使える."""
        from pneuma_core.runtime.diary_writer import DiaryWriter
        assert DiaryWriter is not None
