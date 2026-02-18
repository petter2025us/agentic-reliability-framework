import pytest
from agentic_reliability_framework.infrastructure.cost_estimator import CostEstimator
from agentic_reliability_framework.infrastructure.intents import ProvisionResourceIntent, ResourceType


def test_cost_estimator_known_size():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="prod"
    )
    cost = estimator.estimate_monthly_cost(intent)
    assert cost == 70.0


def test_cost_estimator_unknown_size():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="SuperLarge",
        requester="alice",
        environment="prod"
    )
    cost = estimator.estimate_monthly_cost(intent)
    assert cost is None


def test_cost_delta_with_baseline():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="prod"
    )
    baseline = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="bob",
        environment="prod"
    )
    delta = estimator.cost_delta_vs_baseline(intent, baseline)
    assert delta == 210.0  # 280 - 70


def test_cost_delta_without_baseline():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="prod"
    )
    delta = estimator.cost_delta_vs_baseline(intent)
    # smallest VM is 70, so delta = 280 - 70 = 210
    assert delta == 210.0
