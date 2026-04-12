"""
Inference Script for Web Auditor Environment
Complies with the mandatory OpenEnv stdout format:

[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
from typing import List, Optional

from openai import OpenAI

# ── Mandatory environment variables ──────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")                    # set by evaluator
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("TASK_NAME",    "fix_alt_tags")
BENCHMARK    = os.getenv("BENCHMARK",    "web_auditor")
SPACE_URL    = os.getenv("SPACE_URL",    "http://localhost:8000")

# All 3 task IDs that must each produce an [END] line
TASK_IDS = ["fix_alt_tags", "fix_heading_hierarchy", "create_sitemap"]

MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a Webmaster AI agent fixing a broken HTML website.
The site has 3 problems:
1. Images in gallery.html are missing alt attributes
2. Headings in index.html skip levels (h1 -> h4)
3. No sitemap.xml exists

Execute bash commands to fix these issues. Respond ONLY with valid JSON:
{"name": "execute_bash", "arguments": {"command": "your bash command here"}}
"""

# ── Log helpers ───────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Sanitise action: remove newlines so everything stays on one line
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── Simple HTTP client (no Docker needed) ────────────────────────────────────

class SimpleHTTPClient:
    """Talks to the FastAPI environment over plain HTTP."""

    def __init__(self, base_url: str):
        import requests as _requests
        self._s  = _requests.Session()
        self._url = base_url.rstrip("/")

    def reset(self) -> dict:
        r = self._s.post(f"{self._url}/reset", timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, command: str) -> dict:
        r = self._s.post(
            f"{self._url}/step",
            json={"action": {"command": command}},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self._s.close()


# ── One episode for a single task ────────────────────────────────────────────

async def run_task(task_id: str, client: OpenAI) -> None:
    """Run one episode and emit [START] … [STEP]… [END] for task_id."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Connect to environment ──────────────────────────────────────────
        if IMAGE_NAME:
            # Evaluator provides a docker image
            from client import WebAuditorEnv
            from models import WebAuditorAction
            env_docker = await WebAuditorEnv.from_docker_image(IMAGE_NAME)
        else:
            env_docker = None

        http_client = SimpleHTTPClient(SPACE_URL) if not env_docker else None

        # ── Reset ───────────────────────────────────────────────────────────
        if env_docker:
            from models import WebAuditorAction
            result = await env_docker.reset()
            obs_dir  = result.observation.current_directory_structure
            obs_out  = result.observation.output
        else:
            data     = http_client.reset()
            obs      = data.get("observation", {})
            obs_dir  = obs.get("current_directory_structure", "")
            obs_out  = obs.get("output", "")

        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": f"Task: {task_id}\nFiles:\n{obs_dir}\nOutput:\n{obs_out}"},
        ]

        # ── Steps ───────────────────────────────────────────────────────────
        for step_num in range(1, MAX_STEPS + 1):
            steps_taken = step_num
            error_msg: Optional[str] = None
            command = "ls"

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.0,
                )
                reply = (completion.choices[0].message.content or "").strip()
                messages.append({"role": "assistant", "content": reply})

                # Parse JSON command
                clean = reply[reply.find("{"):reply.rfind("}")+1] if "{" in reply else ""
                command = json.loads(clean).get("arguments", {}).get("command", "ls")
            except Exception as parse_err:
                error_msg = str(parse_err)[:80]

            # Execute step
            try:
                if env_docker:
                    from models import WebAuditorAction
                    result  = await env_docker.step(WebAuditorAction(command=command))
                    reward  = float(result.reward or 0.0)
                    done    = bool(result.done)
                    obs_out = result.observation.output
                    obs_dir = result.observation.current_directory_structure
                else:
                    data    = http_client.step(command)
                    reward  = float(data.get("reward") or 0.0)
                    done    = bool(data.get("done", False))
                    obs     = data.get("observation", {})
                    obs_out = obs.get("output", "")
                    obs_dir = obs.get("current_directory_structure", "")
            except Exception as step_err:
                reward    = 0.0
                done      = False
                error_msg = str(step_err)[:80]

            rewards.append(reward)
            log_step(step=step_num, action=command, reward=reward, done=done, error=error_msg)

            messages.append({"role": "user", "content": f"Output:\n{obs_out}"})

            if done:
                break

        # Score = last reward (already in [0,1] from environment)
        score   = max(0.0, min(1.0, rewards[-1] if rewards else 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] run_task({task_id}) error: {e}", flush=True)

    finally:
        # Clean up
        if env_docker is not None:
            try:
                if asyncio.iscoroutinefunction(getattr(env_docker, "close", None)):
                    await env_docker.close()
                else:
                    env_docker.close()
            except Exception:
                pass
        if http_client is not None:
            try:
                http_client.close()
            except Exception:
                pass

        # Mandatory [END] line
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main: run all 3 tasks sequentially ───────────────────────────────────────

async def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    for task_id in TASK_IDS:
        await run_task(task_id, client)


if __name__ == "__main__":
    asyncio.run(main())
