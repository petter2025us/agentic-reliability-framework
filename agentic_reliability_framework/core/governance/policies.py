# agentic_reliability_framework/core/governance/policies.py
"""
Policy Algebra – Composable, typed policies for infrastructure governance.
Enhanced with probabilistic evaluation for uncertain inputs.

This module defines a composable policy system using a monoid-like structure.
Policies can be combined (AND, OR) and evaluated against intents. The algebra
enables building complex rules from simple primitives while maintaining
deterministic evaluation.

Additionally, it provides a probabilistic evaluator that accepts uncertain
inputs (distributions) and returns violation probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, TypeVar, Union
from collections import defaultdict

import numpy as np
from scipy import stats

from pydantic import BaseModel

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
    PermissionLevel,
    Environment,
)

# -----------------------------------------------------------------------------
# Type Aliases for Readability
# -----------------------------------------------------------------------------
Violation = str
EvalResult = List[Violation]

# -----------------------------------------------------------------------------
# Abstract Policy Class – using Composite pattern
# -----------------------------------------------------------------------------
class Policy(ABC):
    """Abstract base for all policies. Evaluates an intent and returns violations."""

    @abstractmethod
    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        """Return list of violations. Empty list means the policy is satisfied."""
        pass

    # -------------------------------------------------------------------------
    # Combinators (returning new composite policies)
    # -------------------------------------------------------------------------
    def __and__(self, other: Policy) -> Policy:
        """Logical AND: violations from both policies."""
        return AndPolicy(self, other)

    def __or__(self, other: Policy) -> Policy:
        """Logical OR: violations only if both policies produce violations."""
        return OrPolicy(self, other)

    def __invert__(self) -> Policy:
        """Logical NOT: violations if the original policy yields no violations."""
        return NotPolicy(self)

# -----------------------------------------------------------------------------
# Atomic Policies (Primitives)
# -----------------------------------------------------------------------------
class AtomicPolicy(Policy, ABC):
    """Base class for policies that don't contain other policies."""
    pass

@dataclass(frozen=True)
class RegionAllowedPolicy(AtomicPolicy):
    """Ensure the intent's region is in an allowed set."""
    allowed_regions: Set[str]

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        if isinstance(intent, ProvisionResourceIntent):
            if intent.region not in self.allowed_regions:
                return [f"Region '{intent.region}' not allowed. Allowed: {self.allowed_regions}"]
        return []

@dataclass(frozen=True)
class ResourceTypeRestrictedPolicy(AtomicPolicy):
    """Forbid certain resource types."""
    forbidden_types: Set[ResourceType]

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        if isinstance(intent, ProvisionResourceIntent):
            if intent.resource_type in self.forbidden_types:
                return [f"Resource type '{intent.resource_type.value}' is forbidden."]
        return []

@dataclass(frozen=True)
class MaxPermissionLevelPolicy(AtomicPolicy):
    """Limit the maximum permission level that can be granted."""
    max_level: PermissionLevel

    # Permission level ordering (read < write < admin)
    _LEVEL_ORDER = {
        PermissionLevel.READ: 1,
        PermissionLevel.WRITE: 2,
        PermissionLevel.ADMIN: 3,
    }

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        if isinstance(intent, GrantAccessIntent):
            if self._LEVEL_ORDER[intent.permission_level] > self._LEVEL_ORDER[self.max_level]:
                return [f"Permission level '{intent.permission_level.value}' exceeds max allowed '{self.max_level.value}'."]
        return []

@dataclass(frozen=True)
class CostThresholdPolicy(AtomicPolicy):
    """
    Enforce a maximum monthly cost.
    Note: This policy requires the cost estimate to be provided externally.
    """
    max_cost_usd: float

    def evaluate(self, intent: InfrastructureIntent, cost: Optional[float] = None) -> EvalResult:
        # This method is used by the deterministic evaluator with a cost value.
        # For probabilistic evaluation, we use the UncertainNumber version via the evaluator.
        return []

# -----------------------------------------------------------------------------
# Composite Policies (using Combinators)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AndPolicy(Policy):
    """Logical AND of two policies."""
    left: Policy
    right: Policy

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        return self.left.evaluate(intent) + self.right.evaluate(intent)

@dataclass(frozen=True)
class OrPolicy(Policy):
    """Logical OR – violations only if both sub-policies have violations."""
    left: Policy
    right: Policy

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        left_violations = self.left.evaluate(intent)
        right_violations = self.right.evaluate(intent)
        # If either has no violations, overall no violations (OR satisfied)
        if not left_violations or not right_violations:
            return []
        # Both have violations – combine them
        return left_violations + right_violations

@dataclass(frozen=True)
class NotPolicy(Policy):
    """Logical NOT – violations if inner policy has no violations."""
    inner: Policy

    def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
        inner_violations = self.inner.evaluate(intent)
        if inner_violations:
            return []  # inner failed (had violations), so NOT is satisfied (no violations)
        else:
            return ["Condition not met (NOT policy)"]

# -----------------------------------------------------------------------------
# Context-Aware Deterministic Evaluator
# -----------------------------------------------------------------------------
class PolicyEvaluator:
    """
    Evaluates a policy tree with additional context (e.g., cost estimates).
    This allows policies that depend on external data to be evaluated.
    """

    def __init__(self, root_policy: Policy):
        self._root = root_policy

    def evaluate(self, intent: InfrastructureIntent, context: Optional[Dict[str, Any]] = None) -> EvalResult:
        """
        Evaluate the policy tree against the intent, using context for dynamic checks.
        Context may contain 'cost_estimate' for CostThresholdPolicy, etc.
        """
        return self._evaluate_recursive(self._root, intent, context or {})

    def _evaluate_recursive(self, policy: Policy, intent: InfrastructureIntent, context: Dict[str, Any]) -> EvalResult:
        # If policy is CostThresholdPolicy, we use context to evaluate.
        if isinstance(policy, CostThresholdPolicy):
            cost = context.get('cost_estimate')
            if cost is not None and cost > policy.max_cost_usd:
                return [f"Cost ${cost:.2f} exceeds threshold ${policy.max_cost_usd:.2f}"]
            return []
        # For atomic policies, just evaluate (they don't need context)
        if isinstance(policy, AtomicPolicy):
            return policy.evaluate(intent)
        # For composites, recurse
        if isinstance(policy, AndPolicy):
            return self._evaluate_recursive(policy.left, intent, context) + self._evaluate_recursive(policy.right, intent, context)
        if isinstance(policy, OrPolicy):
            left = self._evaluate_recursive(policy.left, intent, context)
            right = self._evaluate_recursive(policy.right, intent, context)
            if not left or not right:
                return []
            return left + right
        if isinstance(policy, NotPolicy):
            inner = self._evaluate_recursive(policy.inner, intent, context)
            if inner:
                return []
            return ["Condition not met (NOT policy)"]
        # Fallback
        return []

# -----------------------------------------------------------------------------
# Probabilistic Evaluation – Uncertain Numbers and Probabilistic Evaluator
# -----------------------------------------------------------------------------

class UncertainNumber:
    """
    Represents an uncertain numeric quantity with a probability distribution.
    Currently supports normal distribution; can be extended.
    """
    def __init__(self, mean: float, std: float = 0.0, dist_type='normal'):
        self.mean = mean
        self.std = std
        self.dist_type = dist_type
        if dist_type == 'normal' and std > 0:
            self.dist = stats.norm(loc=mean, scale=std)
        else:
            self.dist = None  # deterministic

    def probability_gt(self, threshold: float) -> float:
        """Return P(X > threshold)"""
        if self.dist is None:
            return 1.0 if self.mean > threshold else 0.0
        return 1 - self.dist.cdf(threshold)

    def probability_lt(self, threshold: float) -> float:
        """Return P(X < threshold)"""
        if self.dist is None:
            return 1.0 if self.mean < threshold else 0.0
        return self.dist.cdf(threshold)

    def sample(self, n: int = 1000) -> np.ndarray:
        """Draw samples for Monte Carlo."""
        if self.dist is None:
            return np.full(n, self.mean)
        return self.dist.rvs(size=n)


class ProbabilisticPolicyEvaluator:
    """
    Evaluates policies with uncertain inputs, returning probability of violation.
    """

    def __init__(self, root_policy: Policy):
        self._root = root_policy

    def evaluate_probabilistic(self, intent: InfrastructureIntent,
                               context: Optional[Dict[str, UncertainNumber]] = None,
                               n_samples: int = 1000) -> Dict[str, Any]:
        """
        Evaluate policy tree with uncertain context, returning violation probabilities.

        Args:
            intent: The infrastructure intent.
            context: Dict mapping context keys to UncertainNumber (e.g., cost_estimate).
            n_samples: Number of Monte Carlo samples for uncertainty propagation.

        Returns:
            Dictionary with:
                - violation_probability: overall probability of violation (0-1)
                - per_policy_probabilities: dict mapping policy names to violation probabilities
                - samples_used: number of samples
        """
        if context is None:
            context = {}

        # Prepare sample arrays for each uncertain context key
        samples = {}
        for key, unc in context.items():
            samples[key] = unc.sample(n_samples)

        # If no uncertain inputs, just do deterministic evaluation
        if not samples:
            violations = self._evaluate_deterministic(intent, context)
            return {
                'violation_probability': 1.0 if violations else 0.0,
                'per_policy_probabilities': {},
                'samples_used': 1
            }

        # Monte Carlo
        violation_counts = 0
        # We'll also track per-policy probabilities approximately by noting which policies contributed
        # This requires a more detailed evaluation that returns structured violations.
        # For simplicity, we'll just return overall probability here.
        # A more advanced version could record per-policy violations for each sample.

        for i in range(n_samples):
            # Build context dict for this sample
            sample_context = {}
            for key, arr in samples.items():
                sample_context[key] = arr[i]

            # Evaluate deterministic policy
            eval_det = PolicyEvaluator(self._root)
            violations = eval_det.evaluate(intent, sample_context)

            if violations:
                violation_counts += 1

        violation_prob = violation_counts / n_samples

        return {
            'violation_probability': violation_prob,
            'per_policy_probabilities': {},  # placeholder for future extension
            'samples_used': n_samples
        }

    def _evaluate_deterministic(self, intent: InfrastructureIntent,
                                 context: Dict[str, Any]) -> List[str]:
        """Helper to run deterministic evaluation with point values."""
        eval_det = PolicyEvaluator(self._root)
        return eval_det.evaluate(intent, context)

# -----------------------------------------------------------------------------
# Policy Builder (DSL) – convenience functions
# -----------------------------------------------------------------------------
def allow_all() -> Policy:
    """Policy that never produces violations."""
    class _AllowAll(AtomicPolicy):
        def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
            return []
    return _AllowAll()

def deny_all() -> Policy:
    """Policy that always produces a violation."""
    class _DenyAll(AtomicPolicy):
        def evaluate(self, intent: InfrastructureIntent) -> EvalResult:
            return ["Action denied by default policy"]
    return _DenyAll()

# Example: policy = (region_allowed({"eastus"}) & ~resource_type_restricted({ResourceType.KUBERNETES_CLUSTER})) | cost_threshold(500)

__all__ = [
    "Policy",
    "RegionAllowedPolicy",
    "ResourceTypeRestrictedPolicy",
    "MaxPermissionLevelPolicy",
    "CostThresholdPolicy",
    "AndPolicy",
    "OrPolicy",
    "NotPolicy",
    "PolicyEvaluator",
    "ProbabilisticPolicyEvaluator",
    "UncertainNumber",
    "allow_all",
    "deny_all",
]
