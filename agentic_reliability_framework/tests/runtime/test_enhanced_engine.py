import asyncio
import numpy as np
import pytest
from agentic_reliability_framework.runtime.engine import EnhancedReliabilityEngine
from agentic_reliability_framework.core.governance.intents import EventSeverity


class DummyOrchestrator:
    async def orchestrate_analysis(self, event):
        return {
            'incident_summary': {'anomaly_confidence': 0.9},
            'agent_metadata': {'participating_agents': ['a1', 'a2']}
        }


class DummyAnomalyDetector:
    def __init__(self, val):
        self.val = val

    def detect_anomaly(self, event):
        return self.val


class DummyHMC:
    def __init__(self, ready=False):
        self.is_ready = ready

    def posterior_predictive(self, comp, data):
        return np.array([0.1, 0.2, 0.3])


class DummyClaude:
    def generate_completion(self, prompt, system_prompt):
        return "synthesized summary"


@pytest.mark.asyncio
async def test_process_event_valid_and_anomaly():
    engine = EnhancedReliabilityEngine(
        orchestrator=DummyOrchestrator(),
        anomaly_detector=DummyAnomalyDetector(True),
        hmc_learner=DummyHMC(ready=True),
        claude_adapter=DummyClaude(),
    )
    result = await engine.process_event_enhanced("comp1", 200, 0.05)
    assert result['status'] == "ANOMALY"
    assert 'multi_agent_analysis' in result
    assert 'claude_synthesis' in result
    assert result['severity'] == EventSeverity.HIGH.value or result['severity'] == EventSeverity.CRITICAL.value


@pytest.mark.asyncio
async def test_process_event_invalid_component():
    engine = EnhancedReliabilityEngine(
        orchestrator=DummyOrchestrator(),
        anomaly_detector=DummyAnomalyDetector(False),
        hmc_learner=DummyHMC(ready=False),
        claude_adapter=DummyClaude(),
    )
    res = await engine.process_event_enhanced("", 100, 0.01)
    assert res['status'] == 'INVALID'
    assert 'error' in res
