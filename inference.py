import asyncio
import os
import json
import time
from typing import List, Optional
from openai import OpenAI

from models import WebAuditorAction
from client import WebAuditorEnv

# Required env variables as per Pre-Submission Checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME") # Important for Phase 2!

SYSTEM_PROMPT = """You are a Webmaster AI agent. Your goal is to fix a broken HTML website.
The working directory contains HTML files with these issues:
1. Missing/empty alt attributes on images in gallery.html
2. Broken heading hierarchy (h1 jumps to h4) in index.html
3. Missing sitemap.xml

Output ONLY valid JSON with this schema (no markdown, no explanation):
{"name": "execute_bash", "arguments": {"command": "your bash command here"}}
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    print("=== Web Auditor Baseline Inference ===")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")

    log_start(task="web_auditor", env="web_auditor", model=MODEL_NAME)

    scores = []
    success = False
    client = None
    steps_taken = 0

    try:
        # Avoid instant global crash if token isn't provided
        client = OpenAI(api_key=HF_TOKEN or "dummy_key", base_url=API_BASE_URL)
        
        env = await WebAuditorEnv.from_docker_image(IMAGE_NAME)
        
        try:
            result = await env.reset()
            obs = result.observation
            
            print(f"Initial reward: {result.reward or 0.0}")
            print(f"Directory:\n{obs.current_directory_structure}")

            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OBSERVATION: output={obs.output} files={obs.current_directory_structure}"}
            ]

            for step_num in range(1, 11):
                if result.done:
                    break

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=history,
                        max_tokens=256,
                        temperature=0.0,
                    )
                    agent_reply = response.choices[0].message.content.strip()
                    history.append({"role": "assistant", "content": agent_reply})

                    clean = agent_reply.replace("```json", "").replace("```", "").strip()
                    action_data = json.loads(clean)
                    cmd = action_data.get("arguments", {}).get("command", "echo no-op")
                    
                    cmd_single_line = cmd.replace('\n', '\\n').replace('\r', '')

                    # Execute
                    result = await env.step(WebAuditorAction(command=cmd))
                    obs = result.observation
                    
                    reward = result.reward or 0.0
                    done = result.done
                    error = None
                    
                    scores.append(reward)
                    steps_taken = step_num
                    
                    log_step(step=step_num, action=cmd_single_line, reward=reward, done=done, error=error)
                    
                    history.append({"role": "user", "content": f"OBSERVATION: output={obs.output} files={obs.current_directory_structure}"})

                except Exception as step_exc:
                    print(f"Step {step_num} error: {step_exc}")
                    break

            if scores:
                final_score = scores[-1]
                success = final_score >= 0.99

        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Global execution error: {e}")
    finally:
        print(f"\n=== FINAL SCORES ===")
        final_score = scores[-1] if scores else 0.0
        log_end(success=success, steps=steps_taken, score=final_score, rewards=scores)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")
