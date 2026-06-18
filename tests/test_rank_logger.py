"""Tests for RankLogger and log_ranks rank prefix behavior."""

from __future__ import annotations

import logging
from unittest.mock import patch

from vlm2emb.utils import logging as logging_utils
from vlm2emb.utils.logging import RankLogger, log_ranks


class TestRankLoggerNonDistributed:
    """RankLogger in non-distributed environment prints normally."""

    def test_info_logs_message(self, caplog):
        logger = RankLogger("test.info")
        with caplog.at_level(logging.INFO, logger="test.info"):
            logger.info("hello %s", "world")
        assert "hello world" in caplog.text

    def test_debug_logs_message(self, caplog):
        logger = RankLogger("test.debug")
        with caplog.at_level(logging.DEBUG, logger="test.debug"):
            logger.debug("debug detail")
        assert "debug detail" in caplog.text

    def test_warning_logs_message(self, caplog):
        logger = RankLogger("test.warning")
        with caplog.at_level(logging.WARNING, logger="test.warning"):
            logger.warning("something wrong")
        assert "something wrong" in caplog.text

    def test_no_rank_prefix_when_non_distributed(self, caplog):
        logger = RankLogger("test.noprefix")
        with caplog.at_level(logging.INFO, logger="test.noprefix"):
            logger.info("no prefix expected")
        assert "[rank" not in caplog.text


class TestRankLoggerDistributed:
    """RankLogger with mocked distributed environment."""

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(0, 4))
    def test_info_only_rank0(self, mock_rank, caplog):
        logger = RankLogger("test.dist.info")
        with caplog.at_level(logging.INFO, logger="test.dist.info"):
            logger.info("rank0 only")
        assert "[rank0]" in caplog.text
        assert "rank0 only" in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(2, 4))
    def test_info_suppressed_on_non_rank0(self, mock_rank, caplog):
        logger = RankLogger("test.dist.info2")
        with caplog.at_level(logging.INFO, logger="test.dist.info2"):
            logger.info("should not appear")
        assert "should not appear" not in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(3, 8))
    def test_warning_suppressed_on_non_rank0_by_default(self, mock_rank, caplog):
        logger = RankLogger("test.dist.warn")
        with caplog.at_level(logging.WARNING, logger="test.dist.warn"):
            logger.warning("problem detected")
        assert "problem detected" not in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(3, 8))
    def test_warning_all_ranks_with_explicit_override(self, mock_rank, caplog):
        logger = RankLogger("test.dist.warn.all")
        with caplog.at_level(logging.WARNING, logger="test.dist.warn.all"):
            logger.warning("problem detected", ranks=[-1])
        assert "[rank3]" in caplog.text
        assert "problem detected" in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(0, 4))
    def test_warning_rank0_with_prefix(self, mock_rank, caplog):
        logger = RankLogger("test.dist.warn0")
        with caplog.at_level(logging.WARNING, logger="test.dist.warn0"):
            logger.warning("warn from 0")
        assert "[rank0]" in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(1, 4))
    def test_debug_suppressed_on_non_rank0(self, mock_rank, caplog):
        logger = RankLogger("test.dist.debug")
        with caplog.at_level(logging.DEBUG, logger="test.dist.debug"):
            logger.debug("debug info")
        assert "debug info" not in caplog.text

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(0, 4))
    def test_custom_ranks_override(self, mock_rank, caplog):
        logger = RankLogger("test.dist.custom")
        with caplog.at_level(logging.INFO, logger="test.dist.custom"):
            logger.info("all ranks msg", ranks=[-1])
        assert "[rank0]" in caplog.text
        assert "all ranks msg" in caplog.text


class TestLogRanksPrefix:
    """Test that log_ranks adds [rank{N}] prefix in distributed mode."""

    @patch("vlm2emb.utils.logging._get_rank_info", return_value=(2, 4))
    def test_prefix_added_for_all_ranks(self, mock_rank, caplog):
        std_logger = logging.getLogger("test.logranks.prefix")
        with caplog.at_level(logging.WARNING, logger="test.logranks.prefix"):
            log_ranks(std_logger, "test msg", level=logging.WARNING, ranks=[-1])
        assert "[rank2] test msg" in caplog.text

    def test_no_prefix_non_distributed(self, caplog):
        std_logger = logging.getLogger("test.logranks.noprefix")
        with caplog.at_level(logging.INFO, logger="test.logranks.noprefix"):
            log_ranks(std_logger, "plain msg")
        assert "[rank" not in caplog.text
        assert "plain msg" in caplog.text


class TestEnvRankInfo:
    """Launcher env vars are available before Accelerator initializes."""

    def test_get_rank_info_reads_torchrun_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "8")

        assert logging_utils._get_env_rank_info() == (2, 8)

    def test_get_rank_info_ignores_single_process_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")

        assert logging_utils._get_env_rank_info() is None
