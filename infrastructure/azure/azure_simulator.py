"""
Azure Infrastructure Simulator – main entry point for evaluating infrastructure intents.

Orchestrates cost estimation, policy evaluation (static and dynamic), and risk scoring
to produce a HealingIntent.
"""

from typing import List, Optional

from agentic_reliability_framework.infrastructure.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
)
from agentic_reliability_framework.infrastructure.policies import Policy, PolicyEvaluator
from agentic_reliability_framework.infrastructure.cost_estimator import CostEstimator
from agentic_reliability_framework.infrastructure.risk_engine import RiskEngine
from agentic_reliability_framework.infrastructure.healing_intent import HealingIntent, RecommendedAction


class AzureInfrastructureSimulator:
    """
    Simulates Azure infrastructure governance.

    This class is the core of the OSS advisory module. It evaluates an infrastructure
    intent against policies, estimates cost, calculates risk, and returns a HealingIntent
    with a recommended action.
    """

    def __init__(
        self,
        policies: List[Policy],
        pricing_file: Optional[str] = None,
        risk_weights: Optional[dict] = None,
    ):
        """
        Initialize the simulator.

        Args:
            policies: List of Policy objects to enforce.
            pricing_file: Optional path to a YAML file with custom pricing.
            risk_weights: Optional custom weights for risk calculation.
        """
        self.policy_evaluator = PolicyEvaluator(policies)
        self.policies = policies  # keep for dynamic checks
        self.cost_estimator = CostEstimator(pricing_file)
        self.risk_engine = RiskEngine(risk_weights)

    def _collect_violations(
        self,
        intent: InfrastructureIntent,
        cost: Optional[float],
    ) -> List[str]:
        """Collect both static and dynamic policy violations."""
        # Static violations from PolicyEvaluator
        violations = self.policy_evaluator.evaluate(intent)

        # Dynamic: cost threshold checks
        if cost is not None and isinstance(intent, ProvisionResourceIntent):
            for policy in self.policies:
                cost_violation = policy.check_cost(cost)
                if cost_violation:
                    violations.append(cost_violation)

        return violations

    def evaluate(self, intent: InfrastructureIntent) -> HealingIntent:
        """
        Evaluate an infrastructure intent and return a HealingIntent.

        The evaluation is deterministic and has no side effects.
        """
        # Step 1: Estimate cost (if applicable)
        cost = None
        if isinstance(intent, ProvisionResourceIntent):
            cost = self.cost_estimator.estimate_monthly_cost(intent)

        # Step 2: Collect all policy violations
        violations = self._collect_violations(intent, cost)

        # Step 3: Calculate risk score
        risk_score, explanation = self.risk_engine.calculate_risk(intent, cost, violations)

        # Step 4: Determine recommended action
        # Rule: if risk > 0.8 or any policy violations -> DENY
        #       if risk > 0.4 -> ESCALATE
        #       else -> APPROVE
        if risk_score > 0.8 or violations:
            recommended_action = RecommendedAction.DENY
        elif risk_score > 0.4:
            recommended_action = RecommendedAction.ESCALATE
        else:
            recommended_action = RecommendedAction.APPROVE

        # Step 5: Build justification
        justification_parts = [f"Risk score: {risk_score:.2f}."]
        if cost is not None:
            justification_parts.append(f"Estimated monthly cost: ${cost:.2f}.")
        if violations:
            justification_parts.append(f"Policy violations: {'; '.join(violations)}.")
        justification_parts.append(explanation)
        justification = " ".join(justification_parts)

        # Step 6: Create summary
        intent_summary = f"{intent.intent_type} requested by {intent.requester}"

        # Step 7: (Optional) collect raw details for debugging
        details = {
            "cost_estimate": cost,
            "violations": violations,
            "risk_score": risk_score,
            "intent_type": intent.intent_type,
        }

        return HealingIntent(
            intent_summary=intent_summary,
            cost_projection=cost,
            risk_score=risk_score,
            policy_violations=violations,
            recommended_action=recommended_action,
            justification=justification,
            confidence_score=0.9,
            evaluation_details=details,
        )
