import pytest
from agentic_reliability_framework.infrastructure.policies import Policy, PolicyEvaluator
from agentic_reliability_framework.infrastructure.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
)


def test_policy_evaluator_region_restriction():
    policies = [
        Policy(
            name="restrict regions",
            description="Only eastus allowed",
            allowed_regions=["eastus"]
        )
    ]
    evaluator = PolicyEvaluator(policies)

    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="westus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "westus" in violations[0]


def test_policy_evaluator_restricted_resource_type():
    policies = [
        Policy(
            name="no k8s",
            description="Kubernetes clusters are expensive",
            restricted_resource_types=["kubernetes_cluster"]
        )
    ]
    evaluator = PolicyEvaluator(policies)

    intent = ProvisionResourceIntent(
        resource_type=ResourceType.KUBERNETES_CLUSTER,
        region="eastus",
        size="Medium",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "kubernetes_cluster" in violations[0]


def test_policy_evaluator_permission_limit():
    policies = [
        Policy(
            name="no admin",
            description="Admin access is forbidden",
            max_permission_level="write"
        )
    ]
    evaluator = PolicyEvaluator(policies)

    intent = GrantAccessIntent(
        principal="bob",
        permission_level="admin",
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "admin" in violations[0]

    intent_ok = GrantAccessIntent(
        principal="bob",
        permission_level="write",
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    violations_ok = evaluator.evaluate(intent_ok)
    assert len(violations_ok) == 0
