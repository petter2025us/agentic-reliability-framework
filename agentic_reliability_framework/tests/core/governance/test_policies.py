import pytest
import numpy as np
from agentic_reliability_framework.core.governance.policies import (
    RegionAllowedPolicy,
    ResourceTypeRestrictedPolicy,
    MaxPermissionLevelPolicy,
    CostThresholdPolicy,
    AndPolicy,
    OrPolicy,
    NotPolicy,
    PolicyEvaluator,
    UncertainNumber,
    ProbabilisticPolicyEvaluator,
    allow_all,
    deny_all,
)
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
    PermissionLevel,
)


def test_region_allowed_policy():
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    policy = RegionAllowedPolicy({"eastus", "westeurope"})
    assert policy.evaluate(intent) == []
    # create a second intent with a disallowed region to test violation
    bad_intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        # choose a valid Azure region that is not in the allowlist
        region="westus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    violations = policy.evaluate(bad_intent)
    # message should mention 'not allowed' and the offending region
    assert "not allowed" in violations[0]
    assert "westus" in violations[0]


def test_resource_type_restricted():
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.KUBERNETES_CLUSTER,
        region="eastus",
        size="Small",
        requester="u",
        environment="dev",
    )
    policy = ResourceTypeRestrictedPolicy({ResourceType.KUBERNETES_CLUSTER})
    assert policy.evaluate(intent) == ["Resource type 'kubernetes_cluster' is forbidden."]
    # other type passes by constructing a fresh intent
    other_intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    assert policy.evaluate(other_intent) == []


def test_max_permission_level():
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/sub/123",
        requester="alice",
    )
    policy = MaxPermissionLevelPolicy(PermissionLevel.WRITE)
    violations = policy.evaluate(intent)
    assert "exceeds max allowed" in violations[0]
    # create a new intent with a lower permission level (immutable objects)
    intent2 = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.READ,
        resource_scope="/sub/123",
        requester="alice",
    )
    assert policy.evaluate(intent2) == []


def test_cost_threshold_with_evaluator():
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    policy = CostThresholdPolicy(100.0)
    evalr = PolicyEvaluator(policy)
    assert evalr.evaluate(intent, {"cost_estimate": 50.0}) == []
    result = evalr.evaluate(intent, {"cost_estimate": 150.0})
    assert "exceeds threshold" in result[0]


def test_combinators_and_or_not():
    p1 = allow_all()
    p2 = deny_all()
    assert (p1 & p2).evaluate(None) == ["Action denied by default policy"]
    assert (p1 | p2).evaluate(None) == []
    assert (~p1).evaluate(None) == ["Condition not met (NOT policy)"]
    # complex tree
    tree = (~ResourceTypeRestrictedPolicy({ResourceType.VM})) & RegionAllowedPolicy({"eastus"})
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    # ~resource_type_restricted should yield a violation (not because inner returns no violations) -> so NOT satisfied? check semantics
    # in this example inner returns violation (because type is VM and forbidden) so NOT yields [] and And because left [] returns [] overall
    assert tree.evaluate(intent) == []


def test_uncertain_number_probabilities():
    det = UncertainNumber(5.0)
    assert det.probability_gt(4.0) == 1.0
    assert det.probability_lt(6.0) == 1.0
    normal = UncertainNumber(0.0, std=1.0)
    # probability greater than 0 should be ~0.5
    assert pytest.approx(normal.probability_gt(0.0), rel=0.1) == 0.5


def test_probabilistic_policy_evaluator(monkeypatch):
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="u",
        environment="dev",
    )
    policy = CostThresholdPolicy(100.0)
    pe = ProbabilisticPolicyEvaluator(policy)
    # if cost uncertain with mean 50 std 10, probability of violation ~0
    context = {"cost_estimate": UncertainNumber(50, std=10)}
    res = pe.evaluate_probabilistic(intent, context, n_samples=200)
    assert res['violation_probability'] < 0.05


def test_allow_deny_all_helpers():
    assert allow_all().evaluate(None) == []
    assert deny_all().evaluate(None) == ["Action denied by default policy"]
