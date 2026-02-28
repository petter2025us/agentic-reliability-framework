import pytest
import time
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent as InfraHealingIntent,
    IntentSource,
    IntentStatus,
    RecommendedAction,
    ConfidenceDistribution,
    HealingIntentSerializer as NewSerializer,
)
from agentic_reliability_framework.core.models.healing_intent import (
    HealingIntent as OldHealingIntent,
    HealingIntentSerializer as OldSerializer,
)


@pytest.fixture
def sample_infra_healing_intent():
    conf_dist = ConfidenceDistribution(0.85, 0.05)
    return InfraHealingIntent(
        action="provision_vm",
        component="web-server",
        parameters={"size": "Standard_D2s_v3", "region": "eastus"},
        justification="Scale out due to high CPU",
        confidence=0.85,
        confidence_distribution=conf_dist.to_dict(),
        incident_id="inc_12345",
        detected_at=time.time(),
        risk_score=0.35,
        risk_factors={"cost": 0.2, "permission": 0.15},
        cost_projection=120.0,
        cost_confidence_interval=(110.0, 130.0),
        recommended_action=RecommendedAction.APPROVE,
        decision_tree=[{"step": "check_policy", "outcome": "pass"}],
        alternative_actions=[{"action": "scale_out", "confidence": 0.7}],
        reasoning_chain=["CPU > 80%", "Memory > 75%"],
        similar_incidents=[{"id": "inc_123", "similarity": 0.9}],
        rag_similarity_score=0.85,
        source=IntentSource.INFRASTRUCTURE_ANALYSIS,
        status=IntentStatus.OSS_ADVISORY_ONLY,
        policy_violations=["region_not_allowed"],
        infrastructure_intent_id="infra_001",
    )


def test_new_serializer_roundtrip(sample_infra_healing_intent):
    """Verify that the new HealingIntentSerializer can roundtrip a v2 intent."""
    serialized = NewSerializer.serialize(sample_infra_healing_intent, version="2.0.0")
    deserialized = NewSerializer.deserialize(serialized)
    assert deserialized.action == sample_infra_healing_intent.action
    assert deserialized.component == sample_infra_healing_intent.component
    assert deserialized.parameters == sample_infra_healing_intent.parameters
    assert deserialized.justification == sample_infra_healing_intent.justification
    assert deserialized.confidence == sample_infra_healing_intent.confidence
    assert deserialized.risk_score == sample_infra_healing_intent.risk_score
    assert deserialized.source == sample_infra_healing_intent.source


def test_downgrade_to_old_format(sample_infra_healing_intent):
    """
    Test that a v2 intent can be downgraded to the old format (v1.1.0) and still
    be deserialized by the old serializer, preserving core fields.
    """
    # Serialize with v1.1.0 (old format)
    serialized_v1 = NewSerializer.serialize(sample_infra_healing_intent, version="1.1.0")
    # Deserialize with old serializer
    old_intent = OldSerializer.deserialize(serialized_v1)
    assert old_intent.action == sample_infra_healing_intent.action
    assert old_intent.component == sample_infra_healing_intent.component
    assert old_intent.parameters == sample_infra_healing_intent.parameters
    assert old_intent.justification == sample_infra_healing_intent.justification
    assert old_intent.confidence == sample_infra_healing_intent.confidence
