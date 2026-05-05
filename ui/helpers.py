"""
Shared helpers for the TorchTune UI.

Provides project-wide utilities:
  - get_project_root()        -- canonical project root
  - get_torchtune_root()      -- torchtune package root (for recipes/configs)
  - list_recipes()            -- available recipes from the registry
  - list_configs_for_recipe() -- configs for a given recipe
  - list_model_families()     -- model families (llama3, qwen2, etc.)
  - list_checkpoints()        -- checkpoint files in output dirs
  - get_output_dir()          -- default output directory
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(ui_dir)


def get_torchtune_root() -> str:
    """Return the torchtune package installation root."""
    try:
        import torchtune
        return str(Path(torchtune.__file__).parent.parent)
    except ImportError:
        return get_project_root()


_PROJECT_ROOT = get_project_root()
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")


def list_recipes() -> List[Dict]:
    """Return list of available torchtune recipes with metadata."""
    try:
        from torchtune._recipe_registry import get_all_recipes
        recipes = get_all_recipes()
        result = []
        for r in recipes:
            result.append({
                "name": r.name,
                "file_path": r.file_path,
                "configs": [{"name": c.name, "file_path": c.file_path} for c in r.configs],
                "supports_distributed": r.supports_distributed,
            })
        return result
    except ImportError:
        pass

    # Fallback: import the registry module directly (skips torchtune.__init__)
    try:
        import importlib.util
        registry_path = os.path.join(_PROJECT_ROOT, "torchtune", "_recipe_registry.py")
        if os.path.isfile(registry_path):
            spec = importlib.util.spec_from_file_location("_recipe_registry", registry_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            recipes = mod.get_all_recipes()
            result = []
            for r in recipes:
                result.append({
                    "name": r.name,
                    "file_path": r.file_path,
                    "configs": [{"name": c.name, "file_path": c.file_path} for c in r.configs],
                    "supports_distributed": r.supports_distributed,
                })
            return result
    except Exception:
        pass

    return []


def list_configs_for_recipe(recipe_name: str) -> List[Dict]:
    """Return configs available for a given recipe name."""
    for r in list_recipes():
        if r["name"] == recipe_name:
            return r["configs"]
    return []


def list_model_families() -> List[str]:
    """Extract unique model family names from available configs."""
    families = set()
    for r in list_recipes():
        for c in r["configs"]:
            parts = c["name"].split("/")
            if len(parts) >= 2:
                families.add(parts[0])
    return sorted(families)


def list_checkpoints(search_dir: Optional[str] = None) -> List[str]:
    """Scan for checkpoint files (.pt, .pth, .safetensors, .bin)."""
    dirs_to_scan = []
    if search_dir:
        dirs_to_scan.append(search_dir)
    else:
        dirs_to_scan.extend([
            CHECKPOINTS_DIR,
            OUTPUT_DIR,
            os.path.join(_PROJECT_ROOT, "workspace"),
        ])

    found = []
    exts = ("*.pt", "*.pth", "*.safetensors", "*.bin")
    for d in dirs_to_scan:
        if not os.path.isdir(d):
            continue
        try:
            for ext in exts:
                for p in Path(d).rglob(ext):
                    found.append(str(p))
        except OSError:
            pass

    try:
        found.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    except OSError:
        pass
    return found


def get_output_dir() -> str:
    """Return the default output directory, creating it if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def run_tune_command(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run a `tune` CLI command and return (returncode, output)."""
    cmd = [sys.executable, "-m", "torchtune._cli.tune"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or get_project_root(),
            timeout=30,
        )
        return result.returncode, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, str(e)


def list_hf_models_for_download() -> List[Dict]:
    """Return popular model repos for the download tab."""
    return [
        {"repo": "meta-llama/Llama-3.2-1B-Instruct", "family": "llama3_2", "size": "1B"},
        {"repo": "meta-llama/Llama-3.2-3B-Instruct", "family": "llama3_2", "size": "3B"},
        {"repo": "meta-llama/Llama-3.1-8B-Instruct", "family": "llama3_1", "size": "8B"},
        {"repo": "meta-llama/Llama-3.3-70B-Instruct", "family": "llama3_3", "size": "70B"},
        {"repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "family": "llama4", "size": "17Bx16E"},
        {"repo": "mistralai/Mistral-7B-Instruct-v0.3", "family": "mistral", "size": "7B"},
        {"repo": "google/gemma-2-2b-it", "family": "gemma2", "size": "2B"},
        {"repo": "google/gemma-2-9b-it", "family": "gemma2", "size": "9B"},
        {"repo": "microsoft/Phi-3-mini-4k-instruct", "family": "phi3", "size": "3.8B"},
        {"repo": "microsoft/phi-4", "family": "phi4", "size": "14B"},
        {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen2_5", "size": "0.5B"},
        {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen2_5", "size": "1.5B"},
        {"repo": "Qwen/Qwen2.5-3B-Instruct", "family": "qwen2_5", "size": "3B"},
        {"repo": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen2_5", "size": "7B"},
        {"repo": "Qwen/Qwen3-0.6B", "family": "qwen3", "size": "0.6B"},
        {"repo": "Qwen/Qwen3-1.7B", "family": "qwen3", "size": "1.7B"},
        {"repo": "Qwen/Qwen3-4B", "family": "qwen3", "size": "4B"},
        {"repo": "Qwen/Qwen3-8B", "family": "qwen3", "size": "8B"},
    ]
