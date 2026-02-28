"""
Enhanced Reliability Engine – main entry point for processing reliability events.
"""

import asyncio
import logging
import datetime
from typing import Optional, Dict, Any

from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity, HealingAction
from agentic_reliability_framework.core.governance.policy_engine import PolicyEngine
from agentic_reliability_framework.runtime.analytics.anomaly import AdvancedAnomalyDetector
from agentic_reliability_framework.runtime.analytics.predictive import BusinessImpactCalculator
from agentic_reliability_framework.runtime.orchestration.manager import OrchestrationManager
from agentic_reliability_framework.runtime.hmc.hmc_learner import HMCRiskLearner
from agentic_reliability_framework.core.adapters.claude import ClaudeAdapter
from agentic_reliability_framework.core.config.constants import (
    MAX_EVENTS_STORED, AGENT_TIMEOUT_SECONDS
)

logger = logging.getLogger(__name__)


class ThreadSafeEventStore:
    """Simple thread-safe event store for recent events."""
    def __init__(self, max_size: int = MAX_EVENTS_STORED):
        from collections import deque
        self._events = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, event: ReliabilityEvent):
        with self._lock:
            self._events.append(event)

    def get_recent(self, n: int = 15) -> List[ReliabilityEvent]:
        with self._lock:
            return list(self._events)[-n:] if self._events else []


class EnhancedReliabilityEngine:
    def __init__(self, orchestrator: Optional[OrchestrationManager] = None,
                 policy_engine: Optional[PolicyEngine] = None,
                 event_store: Optional[ThreadSafeEventStore] = None,
                 anomaly_detector: Optional[AdvancedAnomalyDetector] = None,
                 business_calculator: Optional[BusinessImpactCalculator] = None,
                 hmc_learner: Optional[HMCRiskLearner] = None,
                 claude_adapter: Optional[ClaudeAdapter] = None):
        self.orchestrator = orchestrator or OrchestrationManager()
        self.policy_engine = policy_engine or PolicyEngine()
        self.event_store = event_store or ThreadSafeEventStore()
        self.anomaly_detector = anomaly_detector or AdvancedAnomalyDetector()
        self.business_calculator = business_calculator or BusinessImpactCalculator()
        self.hmc_learner = hmc_learner or HMCRiskLearner()
        self.claude_adapter = claude_adapter or ClaudeAdapter()
        self.performance_metrics = {
            'total_incidents_processed': 0,
            'multi_agent_analyses': 0,
            'anomalies_detected': 0
        }
        self._lock = threading.RLock()
        logger.info("Initialized EnhancedReliabilityEngine")

    async def process_event_enhanced(self, component: str, latency: float, error_rate: float,
                                      throughput: float = 1000, cpu_util: Optional[float] = None,
                                      memory_util: Optional[float] = None) -> Dict[str, Any]:
        logger.info(f"Processing event for {component}: latency={latency}ms, error_rate={error_rate*100:.1f}%")
        from agentic_reliability_framework.core.models.event import validate_component_id
        is_valid, error_msg = validate_component_id(component)
        if not is_valid:
            return {'error': error_msg, 'status': 'INVALID'}

        try:
            event = ReliabilityEvent(
                component=component,
                latency_p99=latency,
                error_rate=error_rate,
                throughput=throughput,
                cpu_util=cpu_util,
                memory_util=memory_util
            )
        except Exception as e:
            logger.error(f"Event creation error: {e}")
            return {'error': f'Invalid event data: {str(e)}', 'status': 'INVALID'}

        # Multi-agent analysis
        agent_analysis = await self.orchestrator.orchestrate_analysis(event)

        # Anomaly detection
        is_anomaly = self.anomaly_detector.detect_anomaly(event)

        # Determine severity based on agent confidence
        agent_confidence = agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0.0) if agent_analysis else 0.0
        if is_anomaly and agent_confidence > 0.8:
            severity = EventSeverity.CRITICAL
        elif is_anomaly and agent_confidence > 0.6:
            severity = EventSeverity.HIGH
        elif is_anomaly and agent_confidence > 0.4:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW
        event = event.model_copy(update={'severity': severity})

        # Evaluate healing policies
        healing_actions = self.policy_engine.evaluate_policies(event)

        # Calculate business impact
        business_impact = self.business_calculator.calculate_impact(event) if is_anomaly else None

        # HMC analysis (if available)
        hmc_analysis = None
        if self.hmc_learner.is_ready:
            try:
                risk_samples = self.hmc_learner.posterior_predictive(component, event.model_dump())
                hmc_analysis = {
                    'mean_risk': float(np.mean(risk_samples)),
                    'std_risk': float(np.std(risk_samples)),
                    'samples': risk_samples.tolist()[:5]
                }
            except Exception as e:
                logger.error(f"HMC analysis error: {e}")

        # Build result
        result = {
            "timestamp": event.timestamp.isoformat(),
            "component": component,
            "latency_p99": latency,
            "error_rate": error_rate,
            "throughput": throughput,
            "status": "ANOMALY" if is_anomaly else "NORMAL",
            "multi_agent_analysis": agent_analysis,
            "healing_actions": [a.value for a in healing_actions],
            "business_impact": business_impact,
            "severity": event.severity.value,
            "hmc_analysis": hmc_analysis,
            "processing_metadata": {
                "agents_used": agent_analysis.get('agent_metadata', {}).get('participating_agents', []),
                "analysis_confidence": agent_confidence
            }
        }

        self.event_store.add(event)
        with self._lock:
            self.performance_metrics['total_incidents_processed'] += 1
            self.performance_metrics['multi_agent_analyses'] += 1
            if is_anomaly:
                self.performance_metrics['anomalies_detected'] += 1

        # Enhance with Claude (optional)
        try:
            result = await self.enhance_with_claude(event, result)
        except Exception as e:
            logger.error(f"Claude enhancement failed: {e}")

        return result

    async def enhance_with_claude(self, event: ReliabilityEvent, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        context_parts = []
        context_parts.append("INCIDENT SUMMARY:")
        context_parts.append(f"Component: {event.component}")
        context_parts.append(f"Timestamp: {event.timestamp.isoformat()}")
        context_parts.append(f"Severity: {event.severity.value}")
        context_parts.append("")
        context_parts.append("METRICS:")
        context_parts.append(f"• Latency P99: {event.latency_p99}ms")
        context_parts.append(f"• Error Rate: {event.error_rate:.1%}")
        context_parts.append(f"• Throughput: {event.throughput} req/s")
        if event.cpu_util:
            context_parts.append(f"• CPU: {event.cpu_util:.1%}")
        if event.memory_util:
            context_parts.append(f"• Memory: {event.memory_util:.1%}")
        context_parts.append("")
        if agent_results.get('multi_agent_analysis'):
            context_parts.append("AGENT ANALYSIS:")
            context_parts.append(json.dumps(agent_results['multi_agent_analysis'], indent=2))
        context = "\n".join(context_parts)

        prompt = f"""{context}
TASK: Provide an executive summary synthesizing all agent analyses.
Include:
1. Concise incident description
2. Most likely root cause
3. Single best recovery action
4. Estimated impact and recovery time
Be specific and actionable."""

        system_prompt = """You are a senior Site Reliability Engineer synthesizing 
multiple AI agent analyses into clear, actionable guidance for incident response. 
Focus on clarity, accuracy, and decisive recommendations."""

        claude_synthesis = self.claude_adapter.generate_completion(prompt, system_prompt)
        agent_results['claude_synthesis'] = {
            'summary': claude_synthesis,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source': 'claude-opus-4'
        }
        return agent_results
