"""Bedrock model access + tool-capability probe.

Sends a 1-token Converse request with a minimal toolConfig to each candidate model.
Reports: invokable (200 vs 403/error) and tool-support (did it accept toolConfig without error).

Usage:
    aws sso login --profile tra-sso
    .venv/bin/python scripts/probe_bedrock_access.py
"""
import boto3
import json
from botocore.exceptions import ClientError

PROFILE = "tra-sso"
REGION = "us-east-1"

# All Tier 1 + Tier 2 candidates
CANDIDATES = [
    # Tier 1 — local↔Bedrock overlaps
    "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-safeguard-120b",
    "openai.gpt-oss-safeguard-20b",
    "qwen.qwen3-32b-v1:0",
    "zai.glm-4.7-flash",
    # Tier 2 — frontier breadth
    "anthropic.claude-opus-4-8",
    "deepseek.r1-v1:0",
    "qwen.qwen3-next-80b-a3b",
    "mistral.mistral-large-3-675b-instruct",
    "zai.glm-5",
    "nvidia.nemotron-super-3-120b",
    "amazon.nova-premier-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "moonshot.kimi-k2-thinking",
    "minimax.minimax-m2.5",
    # Also check cross-region inference profile versions
    "us.anthropic.claude-opus-4-8",
    "us.deepseek.r1-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
]

MINIMAL_TOOL = {
    "tools": [
        {
            "toolSpec": {
                "name": "test_tool",
                "description": "A test tool that does nothing.",
                "inputSchema": {
                    "json": {"type": "object", "properties": {"x": {"type": "string"}}}
                },
            }
        }
    ]
}


def probe_model(client, model_id: str) -> dict:
    """Probe a single model with 1-token Converse + toolConfig."""
    result = {"model_id": model_id, "accessible": False, "tool_support": False, "error": None}
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            inferenceConfig={"maxTokens": 1, "temperature": 0.0},
            toolConfig=MINIMAL_TOOL,
        )
        result["accessible"] = True
        result["tool_support"] = True
        # Check if response has any content
        output = response.get("output", {}).get("message", {}).get("content", [])
        result["response_type"] = output[0].get("text", "") if output else "empty"
        result["stop_reason"] = response.get("stopReason", "unknown")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"][:100]
        result["error"] = f"{error_code}: {error_msg}"
        # If it's a tool-config rejection but the model is accessible, try without tools
        if "tool" in error_msg.lower() or "toolConfig" in error_msg.lower():
            try:
                response = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": "hi"}]}],
                    inferenceConfig={"maxTokens": 1, "temperature": 0.0},
                )
                result["accessible"] = True
                result["tool_support"] = False
                result["error"] = f"accessible but no tool support: {error_code}"
            except ClientError as e2:
                result["error"] = f"{error_code} (tools); {e2.response['Error']['Code']} (no tools)"
    except Exception as e:
        result["error"] = str(e)[:100]
    return result


def main():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    client = session.client("bedrock-runtime")

    print(f"{'Model ID':<50} {'Access':<8} {'Tools':<7} {'Notes'}")
    print("-" * 110)

    tier1_ok = []
    tier2_ok = []

    for model_id in CANDIDATES:
        r = probe_model(client, model_id)
        access = "✅" if r["accessible"] else "❌"
        tools = "✅" if r["tool_support"] else "❌"
        notes = r.get("error", "") or r.get("stop_reason", "")
        print(f"{model_id:<50} {access:<8} {tools:<7} {notes}")

        if r["accessible"] and r["tool_support"]:
            if "openai" in model_id or "qwen3-32b" in model_id or "glm-4.7-flash" in model_id:
                tier1_ok.append(model_id)
            else:
                tier2_ok.append(model_id)

    print("\n" + "=" * 60)
    print(f"Tier 1 invokable + tool-capable: {len(tier1_ok)}")
    for m in tier1_ok:
        print(f"  {m}")
    print(f"\nTier 2 invokable + tool-capable: {len(tier2_ok)}")
    for m in tier2_ok:
        print(f"  {m}")


if __name__ == "__main__":
    main()
