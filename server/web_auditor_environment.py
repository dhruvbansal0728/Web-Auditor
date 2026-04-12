import os
import shutil
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import WebAuditorAction, WebAuditorObservation
except ImportError:
    from models import WebAuditorAction, WebAuditorObservation


def extract_heading_level(tag_name):
    try:
        return int(tag_name[1])
    except:
        return 0

def grade_alt_tags(work_dir) -> float:
    total_images = 0
    valid_alts = 0
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                    images = soup.find_all('img')
                    total_images += len(images)
                    for img in images:
                        alt_val = img.get('alt', '')
                        if len(alt_val.strip()) > 3:
                            valid_alts += 1
                except:
                    pass
    if total_images == 0:
        return 1.0  
    return valid_alts / total_images

def grade_headings(work_dir) -> float:
    violations = 0
    total_transitions = 0
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if not headings: continue
                    for i in range(len(headings) - 1):
                        current_h = extract_heading_level(headings[i].name)
                        next_h = extract_heading_level(headings[i+1].name)
                        total_transitions += 1
                        if next_h > current_h + 1:
                            violations += 1
                except:
                    pass
    if total_transitions == 0:
        return 1.0
    score = 1.0 - (violations / total_transitions)
    return max(0.0, float(score))

def grade_sitemap(work_dir) -> float:
    sitemap_path = os.path.join(work_dir, "sitemap.xml")
    if not os.path.exists(sitemap_path):
        return 0.0
    try:
        tree = ET.parse(sitemap_path)
        xml_root = tree.getroot()
    except:
        return 0.0 
    urls = [elem.text for elem in xml_root.findall(".//loc") if elem.text]
    if not urls:
        return 0.0
    valid_links = 0
    for url in urls:
        filename = url.split('/')[-1]
        if not filename: filename = "index.html"
        local_target = os.path.join(work_dir, filename)
        if os.path.exists(local_target):
            valid_links += 1
    return valid_links / len(urls)


class WebAuditorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Path references (assumes running from web_auditor directory boundary)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.join(base_dir, "template_site")
        # Use tempfile's cross-platform temporary directory for writable workspaces to avoid HF Spaces Permission Errors!
        self.work_dir_base = os.path.join(tempfile.gettempdir(), "working_directory")


    def get_directory_structure(self, instance_work_dir) -> str:
        structure = []
        for root, dirs, files in os.walk(instance_work_dir):
            level = root.replace(instance_work_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            structure.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                structure.append(f"{subindent}{f}")
        return "\n".join(structure)

    def _get_instance_dir(self):
        return f"{self.work_dir_base}_{self._state.episode_id}"

    def reset(self) -> WebAuditorObservation:
        # Create a clean state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        instance_dir = self._get_instance_dir()
        
        if os.path.exists(instance_dir):
            shutil.rmtree(instance_dir)
            
        shutil.copytree(self.template_dir, instance_dir)
        dir_struct = self.get_directory_structure(instance_dir)
        
        return WebAuditorObservation(
            output="Environment Ready: Clean Travel Scrapbook initialized. 3 HTML files contain structural flaws. Fix the alt tags, heading hierarchies, and create a sitemap.",
            current_directory_structure=dir_struct,
            done=False,
            reward=0.0
        )

    def step(self, action: WebAuditorAction) -> WebAuditorObservation:
        self._state.step_count += 1
        instance_dir = self._get_instance_dir()
        
        if not os.path.exists(instance_dir):
            return self.reset()

        result = subprocess.run(
            action.command, 
            cwd=instance_dir, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        output = (result.stdout + result.stderr).strip()
        dir_struct = self.get_directory_structure(instance_dir)
        
        file_content = None
        if action.command.startswith("cat") and ".html" in action.command:
            file_content = output
            
        # Partial progress rewards calculated dynamically via 3 deterministic graders
        t1, t2, t3 = grade_alt_tags(instance_dir), grade_headings(instance_dir), grade_sitemap(instance_dir)
        total_score = (t1 + t2 + t3) / 3.0
        
        # Conclude episode if max score reached
        done = total_score >= 0.99 

        return WebAuditorObservation(
            output=output,
            current_directory_structure=dir_struct,
            file_content=file_content,
            done=done,
            reward=total_score,
            metadata={"step": self._state.step_count, "fix_alt_tags": t1, "fix_heading_hierarchy": t2, "create_sitemap": t3}
        )

    @property
    def state(self) -> State:
        return self._state
