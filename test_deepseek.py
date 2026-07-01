#!/usr/bin/env python
"""Test ET-Agent + DeepSeek V4 basic conversation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
from dotenv import load_dotenv
from hermes_constants import get_hermes_home
env_path = os.path.join(get_hermes_home(), ".env")
load_dotenv(env_path)

# Check API key
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key or "your-deepseek" in api_key:
    print("\n[ERROR] DEEPSEEK_API_KEY not set!")
    print(f"Please edit: {env_path}")
    print("Replace with your actual DeepSeek API Key (starts with sk-)")
    sys.exit(1)

print(f"[*] API Key: {api_key[:10]}...{api_key[-4:]}")

# Build agent
from run_agent import AIAgent

agent = AIAgent(
    model="deepseek-v4-pro",
    provider="deepseek",
    api_key=api_key,
    ephemeral_system_prompt="You are a helpful assistant. Please respond concisely.",
    enabled_toolsets=["safe"],
    max_iterations=3,
    quiet_mode=False,
)

print("[*] Agent initialized. Sending test message...")
print("=" * 50)

# Test conversation
response = agent.chat("Hello! Please say 'DeepSeek V4 integration with ET-Agent is working!' in one sentence.")

print("=" * 50)
print(f"\n[Response] {response}")
print("\n[DONE] DeepSeek V4 + ET-Agent integration verified!")
