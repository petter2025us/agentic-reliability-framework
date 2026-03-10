"""
Comprehensive tests for OSS MCP Client.
"""
import pytest
import json
import time
import uuid
from unittest.mock import MagicMock, patch, AsyncMock, ANY
from datetime import datetime

from agentic_reliability_framework.core.mcp.oss_client import (
    OSSMCPClient,
    OSSMCPResponse,
    OSSAnalysisResult,
    create_oss_mcp_client,
)
from agentic_reliability_framework.core.models.healing_intent import HealingIntent, IntentStatus
from agentic_reliability_framework.core.config.constants import OSS_EDITION, ENTERPRISE_UPGRADE_URL


@pytest.fixture
def client():
    """Fixture for OSSMCPClient with default config."""
    with patch("agentic_reliability_framework.core.mcp.oss_client.oss_config") as mock_oss_config:
        mock_oss_config.safety_guardrails = {
            "action_blacklist": [],
            "max_blast_radius": 3,
            "business_hours": None,
        }
        mock_oss_config.get.return_value = False  # rag_enabled
        yield OSSMCPClient()


class TestOSSMCPClient:
    """Test suite for OSSMCPClient."""

    def test_init_defaults(self, client):
        """Test client initialization with default config."""
        assert client.mode == "advisory"
        assert client.oss_edition == OSS_EDITION
        assert client.registered_tools is not None
        assert len(client.registered_tools) >= 5
        assert "rollback" in client.registered_tools
        assert "restart_container" in client.registered_tools
        assert client.metrics["requests_processed"] == 0

    def test_init_with_config(self):
        """Test client initialization with custom config."""
        config = {"execution_allowed": True, "mcp_mode": "execution"}
        with patch("agentic_reliability_framework.core.mcp.oss_client.oss_config") as mock_oss_config:
            mock_oss_config.safety_guardrails = {}
            mock_oss_config.get.return_value = False
            client = OSSMCPClient(config)
        # Should override invalid config
        assert client._config["execution_allowed"] is False
        assert client._config["mcp_mode"] == "advisory"

    def test_register_oss_tools(self, client):
        """Test that registered tools have correct OSS properties."""
        for tool_name, tool_info in client.registered_tools.items():
            assert tool_info["can_execute"] is False
            assert tool_info["analysis_only"] is True
            assert tool_info["oss_allowed"] is True
            assert "parameters" in tool_info

    def test_validate_request_valid(self, client):
        """Test request validation passes for valid input."""
        result = client._validate_request(
            tool_name="restart_container",
            component="api-server",
            parameters={"container_id": "abc123"},
            context={"justification": "Need to restart due to memory leak"}
        )
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_request_unknown_tool(self, client):
        """Test validation fails for unknown tool."""
        result = client._validate_request("unknown", "comp", {}, None)
        assert result["valid"] is False
        assert "Unknown tool" in result["errors"][0]

    def test_validate_request_missing_required_param(self, client):
        """Test validation fails when required parameter missing."""
        result = client._validate_request("scale_out", "comp", {}, None)
        assert result["valid"] is False
        assert "Missing required parameter: scale_factor" in result["errors"][0]

    def test_validate_request_type_conversion(self, client):
        """Test that parameters are converted to correct types."""
        params = {"scale_factor": "5"}
        result = client._validate_request("scale_out", "comp", params, None)
        assert result["valid"] is True
        assert params["scale_factor"] == 5

    def test_check_dangerous_parameters(self, client):
        """Test detection of dangerous parameter values."""
        result = client._check_dangerous_parameters("traffic_shift", {"percentage": 100})
        assert result["safe"] is False
        assert "dangerous" in result["reason"].lower()

        result = client._check_dangerous_parameters("traffic_shift", {"percentage": 50})
        assert result["safe"] is True

    @pytest.mark.asyncio
    async def test_perform_safety_checks_allowed(self, client):
        """Test safety checks pass for allowed actions."""
        context = {"affected_services": ["comp"], "environment": "staging"}
        result = await client._perform_safety_checks("restart_container", "comp", {}, context)
        assert result["allowed"] is True
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_perform_safety_checks_blacklist(self, client):
        """Test safety check fails if tool is blacklisted."""
        client.safety_guardrails["action_blacklist"] = ["RESTART_CONTAINER"]
        result = await client._perform_safety_checks("restart_container", "comp", {}, None)
        assert result["allowed"] is False
        assert "blacklist" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_perform_safety_checks_blast_radius_warning(self, client):
        """Test warning when blast radius exceeds max."""
        client.safety_guardrails["max_blast_radius"] = 2
        context = {"affected_services": ["a", "b", "c"]}
        result = await client._perform_safety_checks("restart_container", "comp", {}, context)
        assert result["allowed"] is True
        assert any("blast radius" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_query_rag_for_similar_incidents_no_rag(self, client):
        """Test RAG query returns empty when RAG not enabled."""
        client._get_rag_enabled = lambda: False
        result = await client._query_rag_for_similar_incidents("comp", {}, None)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_rag_for_similar_incidents_with_rag(self, client):
        """Test RAG query returns list of incidents."""
        client._get_rag_enabled = lambda: True
        # Patch the rag_graph.get_rag_graph function
        with patch("agentic_reliability_framework.runtime.memory.rag_graph.get_rag_graph", create=True) as mock_get:
            mock_graph = MagicMock()
            mock_graph.is_enabled.return_value = True
            mock_graph.find_similar.return_value = []
            mock_graph.get_outcomes.return_value = []
            mock_get.return_value = mock_graph

            # Now test the method
            result = await client._query_rag_for_similar_incidents("comp", {}, {"incident_id": "id"})
            assert result == []

    def test_calculate_rag_similarity_score(self, client):
        """Test aggregate similarity score calculation."""
        incidents = [{"similarity": 0.8}, {"similarity": 0.9}]
        score = client._calculate_rag_similarity_score(incidents)
        assert score == pytest.approx(0.85)

        incidents = [{"similarity": "high"}, {}]
        score = client._calculate_rag_similarity_score(incidents)
        assert score is None

    def test_calculate_confidence_basic(self, client):
        """Test confidence calculation without context."""
        conf = client._calculate_confidence("restart_container", "comp", {}, [], None)
        assert 0.8 <= conf <= 1.0

    def test_calculate_confidence_with_similar_incidents(self, client):
        """Test confidence boost from similar incidents."""
        incidents = [
            {"similarity": 0.9, "success_rate": 1.0},
            {"similarity": 0.8, "success_rate": 0.5},
        ]
        conf = client._calculate_confidence("restart_container", "comp", {}, incidents, None)
        assert conf > 0.85

    def test_generate_justification_with_similar(self, client):
        """Test justification generation with similar incidents."""
        incidents = [{"success_rate": 0.8}, {"success_rate": 0.9}]
        just = client._generate_justification("restart", "comp", {"param": 1}, incidents, None)
        assert "similar historical incidents" in just
        assert "success rate" in just

    def test_generate_justification_from_context(self, client):
        """Test justification uses context if available."""
        context = {"justification": "My custom reason"}
        just = client._generate_justification("restart", "comp", {}, [], context)
        assert just == "My custom reason"

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_success(self, client):
        """Test full analysis pipeline succeeds."""
        # Mock internal methods
        client._validate_request = MagicMock(return_value={"valid": True, "errors": [], "warnings": []})
        client._check_dangerous_parameters = MagicMock(return_value={"safe": True, "reason": ""})
        client._perform_safety_checks = AsyncMock(return_value={"allowed": True, "warnings": []})
        client._query_rag_for_similar_incidents = AsyncMock(return_value=[])
        client._calculate_rag_similarity_score = MagicMock(return_value=None)
        client._calculate_confidence = MagicMock(return_value=0.9)
        client._create_healing_intent = AsyncMock(return_value=MagicMock())

        result = await client.analyze_and_recommend(
            tool_name="restart_container",
            component="api",
            parameters={},
            context={"incident_id": "123"},
        )

        assert isinstance(result, OSSAnalysisResult)
        assert result.confidence == 0.9
        assert result.similar_incidents_count == 0
        assert result.requires_enterprise is True
        assert client.metrics["requests_processed"] == 1
        assert client.metrics["healing_intents_created"] == 1

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_validation_failure(self, client):
        """Test analysis returns fallback on validation error."""
        client._validate_request = MagicMock(return_value={"valid": False, "errors": ["bad"]})
        # We don't mock other methods; they shouldn't be called because validation fails first.
        result = await client.analyze_and_recommend("restart_container", "api", {}, None)
        assert isinstance(result, OSSAnalysisResult)
        assert result.confidence == 0.3
        assert any("Analysis error" in w for w in result.warnings)
        assert result.warnings[0].startswith("Analysis error:")

    @pytest.mark.asyncio
    async def test_analyze_and_recommend_exception_handling(self, client):
        """Test that exception during analysis returns fallback result."""
        client._validate_request = MagicMock(side_effect=Exception("Unexpected"))
        result = await client.analyze_and_recommend("restart_container", "api", {}, {"incident_id": "123"})
        assert isinstance(result, OSSAnalysisResult)
        assert result.confidence == 0.3
        assert result.warnings[0].startswith("Analysis error:")

    @pytest.mark.asyncio
    async def test_execute_tool_backward_compatibility(self, client):
        """Test execute_tool method for backward compatibility."""
        mock_intent = MagicMock(spec=HealingIntent)
        mock_intent.to_enterprise_request.return_value = {"intent": "data"}
        mock_intent.similar_incidents = []
        mock_intent.rag_similarity_score = None
        mock_intent.source = "oss"
        mock_intent.is_oss_advisory = True
        mock_intent.action = "restart"
        mock_intent.component = "api"
        mock_intent.confidence = 0.9

        mock_result = OSSAnalysisResult(
            healing_intent=mock_intent,
            confidence=0.9,
            similar_incidents_count=0,
            rag_similarity_score=None,
            analysis_time_ms=10.0,
            warnings=[],
            requires_enterprise=True,
        )
        client.analyze_and_recommend = AsyncMock(return_value=mock_result)

        request = {
            "request_id": "req-123",
            "tool": "restart_container",
            "component": "api",
            "parameters": {"container_id": "c1"},
            "justification": "test",
        }

        response_dict = await client.execute_tool(request)
        assert "request_id" in response_dict
        assert response_dict["status"] == "completed"
        assert response_dict["executed"] is False
        assert "result" in response_dict
        assert response_dict["result"]["requires_enterprise"] is True
        assert response_dict["result"]["oss_analysis"]["analysis_time_ms"] == 10.0

    def test_get_client_info(self, client):
        """Test client info retrieval."""
        info = client.get_client_info()
        assert info["mode"] == "advisory"
        assert info["edition"] == OSS_EDITION
        assert info["registered_tools"] == len(client.registered_tools)
        assert info["requires_enterprise_for_execution"] is True
        assert info["upgrade_url"] == ENTERPRISE_UPGRADE_URL

    def test_get_tool_info_all(self, client):
        """Test get_tool_info without name."""
        info = client.get_tool_info()
        assert isinstance(info, dict)
        assert "rollback" in info
        assert info["rollback"]["can_execute"] is False
        assert info["rollback"]["requires_enterprise"] is True

    def test_get_tool_info_specific(self, client):
        """Test get_tool_info for a specific tool."""
        info = client.get_tool_info("restart_container")
        assert info["name"] == "restart_container"
        assert info["can_execute"] is False
        assert info["requires_enterprise"] is True

    def test_get_tool_info_unknown(self, client):
        """Test get_tool_info for unknown tool."""
        info = client.get_tool_info("unknown")
        assert info == {}

    def test_create_cache_key(self, client):
        """Test cache key generation."""
        key1 = client._create_cache_key("comp", {"a": 1, "b": 2}, {"severity": "high"})
        key2 = client._create_cache_key("comp", {"b": 2, "a": 1}, {"severity": "high"})
        assert key1 == key2

    def test_clear_cache(self, client):
        """Test clearing similarity cache."""
        client.similarity_cache = {"key": ["data"]}
        client.clear_cache()
        assert client.similarity_cache == {}

    def test_reset_metrics(self, client):
        """Test resetting metrics."""
        client.metrics["requests_processed"] = 10
        client.reset_metrics()
        assert client.metrics["requests_processed"] == 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self, client):
        """Test async context manager entry/exit."""
        async with client as c:
            assert c is client


def test_create_oss_mcp_client():
    """Test factory function."""
    with patch("agentic_reliability_framework.core.mcp.oss_client.oss_config"):
        client = create_oss_mcp_client({"some": "config"})
    assert isinstance(client, OSSMCPClient)
