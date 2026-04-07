import os
import time
import json
import requests
from openai import OpenAI

# Required env variables as per Pre-Submission Checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI-compatible client using HF_TOKEN and API_BASE_URL
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a Webmaster AI agent. Your goal is to fix a broken HTML website.
The working directory contains HTML files with these issues:
1. Missing/empty alt attributes on images in gallery.html
2. Broken heading hierarchy (h1 jumps to h4) in index.html
3. Missing sitemap.xml

Output ONLY valid JSON with this schema (no markdown, no explanation):
{"name": "execute_bash", "arguments": {"command": "your bash command here"}}
"""

ENV_URL = "http://127.0.0.1:8000"

def main():
    print("=== Web Auditor Baseline Inference ===")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("[START] task=web_auditor", flush=True)

    # Reset environment
    try:
        res = requests.post(f"{ENV_URL}/reset", timeout=30)
        res.raise_for_status()
        obs = res.json()
    except Exception as e:
        print(f"ERROR: Could not connect to environment at {ENV_URL}: {e}")
        print("Make sure the server is running: uvicorn server.app:app --port 8000")
        return

    print(f"Initial reward: {obs.get('reward', 0.0)}")
    print(f"Directory:\n{obs.get('current_directory_structure', '')}")

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"OBSERVATION: {json.dumps(obs)}"}
    ]

    scores = []
    for step_num in range(10):
        print(f"\n--- Step {step_num + 1} ---")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                max_tokens=256,
                temperature=0.0,
            )
            agent_reply = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": agent_reply})

            # Parse the JSON action
            clean = agent_reply.replace("```json", "").replace("```", "").strip()
            action_data = json.loads(clean)
            cmd = action_data.get("arguments", {}).get("command", "echo no-op")
            print(f"Command: {cmd}")

            # Execute step
            step_res = requests.post(f"{ENV_URL}/step", json={"command": cmd}, timeout=30)
            step_res.raise_for_status()
            new_obs = step_res.json()

            reward = new_obs.get("reward", 0.0)
            done = new_obs.get("done", False)
            meta = new_obs.get("metadata", {})
            scores.append(reward)
            print(f"Reward: {reward:.3f} | Task scores: {meta}")
            print(f"[STEP] step={step_num + 1} reward={reward:.3f}", flush=True)

            history.append({"role": "user", "content": f"OBSERVATION: {json.dumps(new_obs)}"})

            if done:
                print("Agent completed all tasks!")
                break

        except json.JSONDecodeError:
            print(f"Could not parse agent response as JSON: {agent_reply[:100]}")
        except Exception as e:
            print(f"Step error: {e}")
            break

    print(f"\n=== FINAL SCORES ===")
    if scores:
        print(f"Final reward: {scores[-1]:.3f}")
        print(f"Peak reward:  {max(scores):.3f}")
        final_score = scores[-1]
    else:
        print("No scores recorded.")
        final_score = 0.0
    
    print(f"[END] task=web_auditor score={final_score:.3f} steps={len(scores)}", flush=True)

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s (limit: 1200s / 20min)")
