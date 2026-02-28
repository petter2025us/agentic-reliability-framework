"""
Base agent classes for the multi-agent system.
"""

from enum import Enum
from typing import Dict, Any
from agentic_reliability_framework.core.models.event import ReliabilityEvent


class AgentSpecialization(Enum):
    DETECTIVE = "anomaly_detection"
    DIAGNOSTICIAN = "root_cause_analysis"
    PREDICTIVE = "predictive_analytics"


class BaseAgent:
    def __init__(self, specialization: AgentSpecialization):
        self.specialization = specialization
        self.performance_metrics = {
            'processed_events': 0,
            'successful_analyses': 0,
            'average_confidence': 0.0
        }

    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        raise NotImplementedError
