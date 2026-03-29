import os
import time
from openai import OpenAI
import json

# Fetch credentials from environment securely fulfilling baseline script compliance
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy_key")
API_BASE_URL = os.environ.get("API_BASE_URL", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

# Create OpenAI client pointing to any provider
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a Webmaster AI agent connected to an OpenEnv simulation.
Your goal is to fix a structurally flawed HTML website (Travel Scrapbook).
You must output ONLY valid JSON representing your tool action using this schema:
{
    "name": "execute_bash",
    "arguments": {
        "command": "Your bash command here"
    }
}
Do not include any other markdown formatting outside the JSON block.
"""

def main():
    # To run inference against an endpoint, you should invoke standard logic hitting the FastAPI endpoint
    # E.g. Using the OpenEnv client logic to ping local 'uv run server' (localhost:7860)
    print("Initializing Agent Inference Loop...")
    import requests
    base_engine = "http://127.0.0.1:8000"
    
    # 1) Reset Environment
    res = requests.post(f"{base_engine}/reset")
    if res.status_code != 200:
        print("Error resetting space:", res.text)
        return
        
    obs = res.json()
    print("Initial State:", obs)
    
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"OBSERVATION: {json.dumps(obs)}"}
    ]
    
    # Simple loop
    for step in range(5):
        print(f"\n--- STEP {step+1} ---")
        try:
            # LLM Engine call matching criteria
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
            )
            
            agent_response = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": agent_response})
            
            try:
                # Clean prompt formatting
                clean_json = agent_response.replace('```json', '').replace('```', '').strip()
                action_data = json.loads(clean_json)
                
                cmd = action_data.get("arguments", {}).get("command", "")
                print(f"Action Output: {cmd}")
                
                # 2) Step Environment Endpoints
                step_res = requests.post(
                    f"{base_engine}/step", 
                    json={"command": cmd}
                )
                new_obs = step_res.json()
                print("Reward:", new_obs.get('reward'), " Done:", new_obs.get('done'))
                
                history.append({"role": "user", "content": f"OBSERVATION: {json.dumps(new_obs)}"})
                
                if new_obs.get('done'):
                    print("Agent fully succeeded!")
                    break
                    
            except Exception as e:
                print("Failed to parse agent action as JSON:", e)
                break
                
        except Exception as e:
            print("Inference error mapping LLM parameters:", e)
            break

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution finished in {time.time() - start_time:.2f}s (Constraint: < 20min)")
