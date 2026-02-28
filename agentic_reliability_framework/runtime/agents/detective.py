"""
Anomaly Detection Agent – specializes in detecting anomalies and pattern recognition.
"""

import logging
from typing import Dict, Any, List
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    LATENCY_WARNING, LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_HIGH, ERROR_RATE_CRITICAL,
    CPU_WARNING, CPU_CRITICAL, MEMORY_WARNING, MEMORY_CRITICAL
)
from .base import BaseAgent, AgentSpecialization

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentSpecialization.DETECTIVE)
        logger.info("Initialized AnomalyDetectionAgent")

    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        try:
            anomaly_score = self._calculate_anomaly_score(event)
            return {
                'specialization': self.specialization.value,
                'confidence': anomaly_score,
                'findings': {
                    'anomaly_score': anomaly_score,
                    'severity_tier': self._classify_severity(anomaly_score),
                    'primary_metrics_affected': self._identify_affected_metrics(event)
                },
                'recommendations': self._generate_detection_recommendations(event, anomaly_score)
            }
        except Exception as e:
            logger.error(f"AnomalyDetectionAgent error: {e}", exc_info=True)
            return {'specialization': self.specialization.value, 'confidence': 0.0, 'findings': {}, 'recommendations': [f"Analysis error: {str(e)}"]}

    def _calculate_anomaly_score(self, event: ReliabilityEvent) -> float:
        scores = []
        if event.latency_p99 > LATENCY_WARNING:
            latency_score = min(1.0, (event.latency_p99 - LATENCY_WARNING) / 500)
            scores.append(0.4 * latency_score)
        if event.error_rate > ERROR_RATE_WARNING:
            error_score = min(1.0, event.error_rate / 0.3)
            scores.append(0.3 * error_score)
        resource_score = 0
        if event.cpu_util and event.cpu_util > CPU_WARNING:
            resource_score += 0.15 * min(1.0, (event.cpu_util - CPU_WARNING) / 0.2)
        if event.memory_util and event.memory_util > MEMORY_WARNING:
            resource_score += 0.15 * min(1.0, (event.memory_util - MEMORY_WARNING) / 0.2)
        scores.append(resource_score)
        return min(1.0, sum(scores))

    def _classify_severity(self, score: float) -> str:
        if score > 0.8: return "CRITICAL"
        if score > 0.6: return "HIGH"
        if score > 0.4: return "MEDIUM"
        return "LOW"

    def _identify_affected_metrics(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        affected = []
        if event.latency_p99 > LATENCY_EXTREME:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "CRITICAL", "threshold": LATENCY_WARNING})
        elif event.latency_p99 > LATENCY_CRITICAL:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "HIGH", "threshold": LATENCY_WARNING})
        elif event.latency_p99 > LATENCY_WARNING:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "MEDIUM", "threshold": LATENCY_WARNING})
        if event.error_rate > ERROR_RATE_CRITICAL:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "CRITICAL", "threshold": ERROR_RATE_WARNING})
        elif event.error_rate > ERROR_RATE_HIGH:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "HIGH", "threshold": ERROR_RATE_WARNING})
        elif event.error_rate > ERROR_RATE_WARNING:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "MEDIUM", "threshold": ERROR_RATE_WARNING})
        if event.cpu_util and event.cpu_util > CPU_CRITICAL:
            affected.append({"metric": "cpu", "value": event.cpu_util, "severity": "CRITICAL", "threshold": CPU_WARNING})
        elif event.cpu_util and event.cpu_util > CPU_WARNING:
            affected.append({"metric": "cpu", "value": event.cpu_util, "severity": "HIGH", "threshold": CPU_WARNING})
        if event.memory_util and event.memory_util > MEMORY_CRITICAL:
            affected.append({"metric": "memory", "value": event.memory_util, "severity": "CRITICAL", "threshold": MEMORY_WARNING})
        elif event.memory_util and event.memory_util > MEMORY_WARNING:
            affected.append({"metric": "memory", "value": event.memory_util, "severity": "HIGH", "threshold": MEMORY_WARNING})
        return affected

    def _generate_detection_recommendations(self, event: ReliabilityEvent, anomaly_score: float) -> List[str]:
        recommendations = []
        for metric in self._identify_affected_metrics(event):
            m = metric["metric"]
            sev = metric["severity"]
            val = metric["value"]
            thr = metric["threshold"]
            if m == "latency":
                if sev == "CRITICAL":
                    recommendations.append(f"🚨 CRITICAL: Latency {val:.0f}ms (>{thr}ms) - Check database & external dependencies")
                elif sev == "HIGH":
                    recommendations.append(f"⚠️ HIGH: Latency {val:.0f}ms (>{thr}ms) - Investigate service performance")
                else:
                    recommendations.append(f"📈 Latency elevated: {val:.0f}ms (>{thr}ms) - Monitor trend")
            elif m == "error_rate":
                if sev == "CRITICAL":
                    recommendations.append(f"🚨 CRITICAL: Error rate {val*100:.1f}% (>{thr*100:.1f}%) - Check recent deployments")
                elif sev == "HIGH":
                    recommendations.append(f"⚠️ HIGH: Error rate {val*100:.1f}% (>{thr*100:.1f}%) - Review application logs")
                else:
                    recommendations.append(f"📈 Errors increasing: {val*100:.1f}% (>{thr*100:.1f}%)")
            elif m == "cpu":
                recommendations.append(f"🔥 CPU {sev}: {val*100:.1f}% utilization - Consider scaling")
            elif m == "memory":
                recommendations.append(f"💾 Memory {sev}: {val*100:.1f}% utilization - Check for memory leaks")
        if anomaly_score > 0.8:
            recommendations.append("🎯 IMMEDIATE ACTION REQUIRED: Multiple critical metrics affected")
        elif anomaly_score > 0.6:
            recommendations.append("🎯 INVESTIGATE: Significant performance degradation detected")
        elif anomaly_score > 0.4:
            recommendations.append("📊 MONITOR: Early warning signs detected")
        return recommendations[:4]
