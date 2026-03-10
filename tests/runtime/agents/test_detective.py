"""
Comprehensive tests for AnomalyDetectionAgent.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from agentic_reliability_framework.runtime.agents.detective import AnomalyDetectionAgent
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.nlp.nli import NLIDetector
from agentic_reliability_framework.core.config.constants import (
    LATENCY_WARNING, LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_HIGH, ERROR_RATE_CRITICAL,
    CPU_WARNING, CPU_CRITICAL, MEMORY_WARNING, MEMORY_CRITICAL
)


@pytest.fixture
def mock_nli():
    """Mock NLI detector."""
    nli = MagicMock(spec=NLIDetector)
    nli.check.return_value = 0.8  # high entailment
    return nli


@pytest.fixture
def infrastructure_event():
    """Basic infrastructure event (no AI fields)."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=200.0,
        error_rate=0.05,
        cpu_util=0.6,
        memory_util=0.7
    )


@pytest.fixture
def agent_without_nli():
    """Agent without NLI detector."""
    return AnomalyDetectionAgent(nli_detector=None)


@pytest.fixture
def agent_with_nli(mock_nli):
    """Agent with mocked NLI detector."""
    return AnomalyDetectionAgent(nli_detector=mock_nli)


class TestAnomalyDetectionAgent:
    """Test suite for AnomalyDetectionAgent."""

    # ---------- Initialization ----------
    def test_init_without_nli(self):
        agent = AnomalyDetectionAgent()
        assert agent.nli is None
        assert agent.specialization.value == "anomaly_detection"

    def test_init_with_nli(self, mock_nli):
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        assert agent.nli is mock_nli
        assert agent.specialization.value == "anomaly_detection"

    # ---------- _calculate_anomaly_score ----------
    @pytest.mark.parametrize("latency,error_rate,cpu,memory,expected", [
        (100.0, 0.01, 0.3, 0.4, 0.0),
        (LATENCY_WARNING + 50, 0.01, 0.3, 0.4, 0.4 * ((LATENCY_WARNING+50 - LATENCY_WARNING)/500)),
        (100.0, ERROR_RATE_WARNING + 0.05, 0.3, 0.4, 0.3 * min(1.0, (ERROR_RATE_WARNING+0.05)/0.3)),
        (100.0, 0.01, CPU_WARNING + 0.1, MEMORY_WARNING + 0.1, 0.15 * min(1.0, 0.1/0.2) * 2),
        (LATENCY_WARNING + 100, ERROR_RATE_WARNING + 0.1, CPU_WARNING + 0.2, MEMORY_WARNING + 0.2,
         0.4*((LATENCY_WARNING+100 - LATENCY_WARNING)/500) + 0.3*min(1.0, (ERROR_RATE_WARNING+0.1)/0.3) + 0.15*min(1.0, 0.2/0.2)*2),
    ])
    def test_calculate_anomaly_score(self, latency, error_rate, cpu, memory, expected):
        event = ReliabilityEvent(
            component="test",
            latency_p99=latency,
            error_rate=error_rate,
            cpu_util=cpu,
            memory_util=memory
        )
        agent = AnomalyDetectionAgent()
        score = agent._calculate_anomaly_score(event)
        assert score == pytest.approx(min(1.0, expected), abs=0.05)

    # ---------- _classify_severity ----------
    @pytest.mark.parametrize("score,expected", [
        (0.9, "CRITICAL"),
        (0.7, "HIGH"),
        (0.5, "MEDIUM"),
        (0.3, "LOW"),
    ])
    def test_classify_severity(self, score, expected):
        agent = AnomalyDetectionAgent()
        assert agent._classify_severity(score) == expected

    # ---------- _identify_affected_metrics ----------
    def test_identify_affected_metrics(self):
        agent = AnomalyDetectionAgent()
        event = ReliabilityEvent(
            component="test",
            latency_p99=LATENCY_EXTREME + 10,
            error_rate=ERROR_RATE_CRITICAL + 0.02,
            cpu_util=CPU_CRITICAL + 0.1,
            memory_util=MEMORY_CRITICAL + 0.1
        )
        affected = agent._identify_affected_metrics(event)
        assert len(affected) >= 4
        for entry in affected:
            assert "metric" in entry
            assert "value" in entry
            assert "severity" in entry
            assert "threshold" in entry

    # ---------- _generate_detection_recommendations ----------
    def test_generate_recommendations(self):
        agent = AnomalyDetectionAgent()
        event = ReliabilityEvent(
            component="test",
            latency_p99=LATENCY_WARNING + 10,
            error_rate=ERROR_RATE_WARNING + 0.01,
            cpu_util=CPU_WARNING + 0.1,
            memory_util=MEMORY_WARNING + 0.1
        )
        score = agent._calculate_anomaly_score(event)
        recs = agent._generate_detection_recommendations(event, score)
        assert len(recs) > 0
        # The recommendation for error rate starts with "Errors increasing"
        assert any("Errors increasing" in r for r in recs)

    # ---------- _analyze_infrastructure ----------
    def test_analyze_infrastructure(self, infrastructure_event):
        agent = AnomalyDetectionAgent()
        result = agent._analyze_infrastructure(infrastructure_event)
        assert result['specialization'] == 'anomaly_detection'
        assert 0 <= result['confidence'] <= 1
        assert 'anomaly_score' in result['findings']
        assert result['findings']['type'] == 'infrastructure'
        assert isinstance(result['recommendations'], list)

    # ---------- _analyze_hallucination ----------
    def test_analyze_hallucination_without_nli(self):
        agent = AnomalyDetectionAgent()  # no NLI
        event = MagicMock()
        event.prompt = "test prompt"
        event.response = "test response"
        event.confidence = 0.9
        result = agent._analyze_hallucination(event)
        assert result['findings']['note'] == 'NLI detector not available'
        assert result['confidence'] == 0.0

    def test_analyze_hallucination_with_nli_no_flags(self, mock_nli):
        mock_nli.check.return_value = 0.9
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        event = MagicMock()
        event.prompt = "prompt"
        event.response = "response"
        event.confidence = 0.9
        result = agent._analyze_hallucination(event)
        assert result['findings']['is_hallucination'] is False
        assert result['findings']['flags'] == []
        assert result['confidence'] == 0

    def test_analyze_hallucination_low_confidence(self, mock_nli):
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        event = MagicMock()
        event.prompt = "prompt"
        event.response = "response"
        event.confidence = 0.5
        result = agent._analyze_hallucination(event)
        assert result['findings']['is_hallucination'] is True
        assert 'low_confidence' in result['findings']['flags']
        assert result['confidence'] > 0

    def test_analyze_hallucination_low_entailment(self, mock_nli):
        mock_nli.check.return_value = 0.3
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        event = MagicMock()
        event.prompt = "prompt"
        event.response = "response"
        event.confidence = 0.9
        result = agent._analyze_hallucination(event)
        assert result['findings']['is_hallucination'] is True
        assert 'low_entailment' in result['findings']['flags']
        assert result['confidence'] > 0

    def test_analyze_hallucination_exception(self, mock_nli):
        mock_nli.check.side_effect = Exception("NLI error")
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        event = MagicMock()
        event.prompt = "prompt"
        event.response = "response"
        event.confidence = 0.9
        result = agent._analyze_hallucination(event)
        assert result['confidence'] == 0.0
        assert 'error' in result['findings']

    # ---------- analyze (main entry point) ----------
    @pytest.mark.asyncio
    async def test_analyze_infrastructure_only(self, infrastructure_event, agent_without_nli):
        result = await agent_without_nli.analyze(infrastructure_event)
        assert result['specialization'] == 'anomaly_detection'
        assert 'anomaly_score' in result['findings']
        assert result['findings']['type'] == 'infrastructure'

    @pytest.mark.asyncio
    async def test_analyze_ai_event(self, agent_with_nli, mock_nli):
        mock_nli.check.return_value = 0.8
        event = MagicMock()
        event.prompt = "prompt"
        event.response = "response"
        event.confidence = 0.9
        event.latency_p99 = LATENCY_WARNING + 50
        event.error_rate = ERROR_RATE_WARNING + 0.02
        event.cpu_util = CPU_WARNING + 0.1
        event.memory_util = MEMORY_WARNING + 0.1
        event.component = "ai-service"
        result = await agent_with_nli.analyze(event)
        assert result['specialization'] == 'anomaly_detection'
        assert 'infrastructure' in result['findings']
        assert 'hallucination' in result['findings']
        assert result['findings']['infrastructure']['type'] == 'infrastructure'
        assert result['findings']['hallucination']['type'] == 'hallucination'
        assert len(result['recommendations']) > 0

    @pytest.mark.asyncio
    async def test_analyze_exception_handling(self, infrastructure_event, agent_without_nli):
        with patch.object(agent_without_nli, '_analyze_infrastructure', side_effect=Exception("Test error")):
            result = await agent_without_nli.analyze(infrastructure_event)
            assert result['confidence'] == 0.0
            assert result['findings'] == {}
            assert 'Analysis error' in result['recommendations'][0]

    # ---------- Edge cases ----------
    def test_ai_event_without_prompt(self, mock_nli):
        """AI event missing prompt/response should still be processed."""
        agent = AnomalyDetectionAgent(nli_detector=mock_nli)
        event = MagicMock()
        event.confidence = 0.8
        # No prompt/response
        event.latency_p99 = 100.0
        event.error_rate = 0.01
        event.cpu_util = 0.3
        event.memory_util = 0.4
        event.component = "ai-service"
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(agent.analyze(event))
        finally:
            loop.close()
        assert 'infrastructure' in result['findings']
        assert 'hallucination' in result['findings']
        assert result['findings']['hallucination']['type'] == 'hallucination'
