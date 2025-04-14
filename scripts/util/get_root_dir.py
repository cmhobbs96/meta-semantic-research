import os
from pathlib import Path

def get_project_root():
    cwd = Path(os.getcwd())
    while cwd.name != "meta-semantic-research":
        if cwd.parent == cwd:
            raise RuntimeError("Project root 'meta-semantic-research' not found")
        cwd = cwd.parent
    return cwd