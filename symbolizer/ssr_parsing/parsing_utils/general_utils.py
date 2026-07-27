import os 
from typing import List, Dict, Any, Set, Collection, Union, Literal

import yaml

def load_env(env_file=""):
    # Missing .env is fine: credentials may come from the process environment
    # (e.g. docker compose `environment:`). API clients fail at call time if unset.
    if not env_file or not os.path.exists(env_file):
        return
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                # Don't override env vars already set (e.g. by run_phase1_ssr.py subprocess env)
                if key not in os.environ:
                    os.environ[key] = value

#next char
def next_letter(char):
    if char.lower() == 'z':
        return 'a' if char.islower() else 'A'
    else:
        return chr(ord(char) + 1)

#load yaml config
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
