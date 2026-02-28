import pytest
import tempfile
import yaml
import os
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.intents import ProvisionResourceIntent, ResourceType, Environment


def test_cost_estimator_known_size():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment=Environment.PROD
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
        environment=Environment.PROD
    )
    cost = estimator.estimate_monthly_cost(intent)
    assert cost is None


def test_cost_estimator_with_yaml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'vm': {
                'custom_size': 999.0
            }
        }, f)
        temp_path = f.name

    estimator = CostEstimator(pricing_file=temp_path)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="custom_size",
        requester="alice",
        environment=Environment.PROD
    )
    cost = estimator.estimate_monthly_cost(intent)
    assert cost == 999.0
    os.unlink(temp_path)


def test_cost_delta_with_baseline():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    baseline = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="bob",
        environment=Environment.PROD
    )
    delta = estimator.cost_delta_vs_baseline(intent, baseline)
    assert delta == 210.0


def test_cost_delta_without_baseline():
    estimator = CostEstimator()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    delta = estimator.cost_delta_vs_baseline(intent)
    assert delta == 210.0  # 280 - 70
