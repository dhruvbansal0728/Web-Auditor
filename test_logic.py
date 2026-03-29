import os
import sys

# Append the current directory to python path so we can import server
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from server.web_auditor_environment import WebAuditorEnvironment
from models import WebAuditorAction

print("--- 1. Initializing Environment ---")
env = WebAuditorEnvironment()

print("\n--- 2. Resetting Clean State ---")
initial_obs = env.reset()
print("Observation Output: ", initial_obs.output)
print("Observation Reward:", initial_obs.reward)     # Should be 0.0
print("Observation Done:  ", initial_obs.done)       # Should be False

print("\n--- 3. Testing Action (Reading index.html) ---")
action = WebAuditorAction(command="cat index.html")
obs = env.step(action)
print("File Content Header:", obs.file_content[:150] if obs.file_content else "None")
print("Reward after read:", obs.reward)

print("\n--- 4. Testing Action (Fixing Alt Tags via sed) ---")
# The easy task grader checks for valid alt tags > 3 chars
fix_command = 'sed -i \'s/<img src="assets\\/paris_trip.png" class="scrapbook-photo">/<img src="assets\\/paris_trip.png" class="scrapbook-photo" alt="Beautiful Paris">/g\' gallery.html'
action_fix = WebAuditorAction(command=fix_command)
obs_fix = env.step(action_fix)

print("Reward after fixing one alt tag:", obs_fix.reward)
print("Metadata Grader Scores:", obs_fix.metadata)
