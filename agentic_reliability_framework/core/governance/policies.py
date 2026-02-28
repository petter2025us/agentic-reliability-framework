# agentic_reliability_framework/infrastructure/policies.py
"""
Policy Algebra – Composable, typed policies for infrastructure governance.

This module defines a composable policy system using a monoid-like structure.
Policies can be combined (AND, OR) and evaluated against intents. The algebra
enables building complex rules from simple primitives while maintaining
deterministic evaluation.

The design draws from knowledge engineering (rule-based systems), decision
engineering (explicit trade-offs), and platform engineering (pluggable backends).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, TypeVar, Union

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
    We evaluate it only when a cost is supplied via context.
    """
    max_cost_usd: float

    def evaluate(self, intent: InfrastructureIntent, cost: Optional[float] = None) -> EvalResult:
        # This is a special case – we need cost from the simulator.
        # We'll handle it by allowing context injection. For composition, we'll use a wrapper.
        # In practice, we'll evaluate this in the simulator and add violations manually.
        # For now, we keep it as a marker.
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
# Context-Aware Evaluation
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
        # For simplicity, we traverse the tree and apply context where needed.
        # A more sophisticated implementation would use a visitor pattern.
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
    "allow_all",
    "deny_all",
]
