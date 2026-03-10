"""
Tests for MemoryDriftDiagnosticianAgent.
"""
import pytest
from unittest.mock import MagicMock
import numpy as np

from agentic_reliability_framework.runtime.agents.diagnostician_memory import MemoryDriftDiagnosticianAgent
from agentic_reliability_framework.core.models.event import ReliabilityEvent


@pytest.mark.asyncio
class TestMemoryDriftDiagnosticianAgent:
    async def test_initialization(self):
        agent = MemoryDriftDiagnosticianAgent(history_window=50, zscore_threshold=1.5)
        assert agent.history_window == 50
        assert agent.zscore_threshold == 1.5
        assert agent._retrieval_scores_history == []

    async def test_analyze_without_retrieval_scores(self):
        event = ReliabilityEvent(
            component="test-component",
            latency_p99=100.0,
            error_rate=0.01
        )
        agent = MemoryDriftDiagnosticianAgent()
        result = await agent.analyze(event)
        assert result['confidence'] == 0.0
        assert result['specialization'] == 'memory_drift'
        assert result['findings'] == {}

    async def test_analyze_with_retrieval_scores_insufficient_history(self):
        event = MagicMock(spec=ReliabilityEvent)
        event.retrieval_scores = [0.8, 0.7, 0.9]
        agent = MemoryDriftDiagnosticianAgent(history_window=10)
        result = await agent.analyze(event)
        assert result['findings']['drift_detected'] is False
        assert result['findings']['current_avg'] == pytest.approx(0.8)
        assert result['findings']['historical_avg'] is None
        assert result['findings']['z_score'] is None
        assert result['confidence'] == 0.0

    async def test_analyze_drift_detection(self):
        agent = MemoryDriftDiagnosticianAgent(history_window=10, zscore_threshold=2.0)
        for _ in range(15):
            event = MagicMock(spec=ReliabilityEvent)
            event.retrieval_scores = [0.5, 0.6, 0.55]
            await agent.analyze(event)
        event = MagicMock(spec=ReliabilityEvent)
        event.retrieval_scores = [0.9, 0.95, 0.92]
        result = await agent.analyze(event)
        assert result['findings']['drift_detected'] is True
        assert result['findings']['z_score'] > 2.0
        assert result['confidence'] > 0
        assert len(result['recommendations']) > 0

    async def test_analyze_no_drift(self):
        agent = MemoryDriftDiagnosticianAgent(history_window=10, zscore_threshold=2.0)
        for _ in range(20):
            event = MagicMock(spec=ReliabilityEvent)
            event.retrieval_scores = [0.5, 0.6, 0.55]
            await agent.analyze(event)
        event = MagicMock(spec=ReliabilityEvent)
        event.retrieval_scores = [0.5, 0.6, 0.55]
        result = await agent.analyze(event)
        assert result['findings']['drift_detected'] is False
        assert abs(result['findings']['z_score']) < 2.0
        assert result['recommendations'] == []

    async def test_history_window_limit(self):
        agent = MemoryDriftDiagnosticianAgent(history_window=5)
        for i in range(10):
            event = MagicMock(spec=ReliabilityEvent)
            event.retrieval_scores = [float(i)]
            await agent.analyze(event)
        assert len(agent._retrieval_scores_history) == 5
        expected = list(range(5, 10))
        actual = agent._retrieval_scores_history
        for e, a in zip(expected, actual):
            assert e == pytest.approx(a)

    async def test_error_handling(self):
        agent = MemoryDriftDiagnosticianAgent()
        event = MagicMock(spec=ReliabilityEvent)
        event.retrieval_scores = "not a list"
        result = await agent.analyze(event)
        assert result['confidence'] == 0.0
        assert result['findings'] == {}
        assert result['recommendations'] == []

    async def test_context_window_override(self):
        agent = MemoryDriftDiagnosticianAgent(history_window=100, zscore_threshold=2.0)
        for _ in range(50):
            event = MagicMock(spec=ReliabilityEvent)
            event.retrieval_scores = [0.5]
            await agent.analyze(event)
        event = MagicMock(spec=ReliabilityEvent)
        event.retrieval_scores = [0.9]
        result = await agent.analyze(event, context_window=10)
        assert 'findings' in result
        assert 'drift_detected' in result['findings']
