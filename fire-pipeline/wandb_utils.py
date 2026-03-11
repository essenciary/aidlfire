"""
Weights & Biases (W&B) utilities for training scripts.

Handles login, API key, offline mode, and headless/cloud environments.
Fails by default if W&B cannot be initialized; use --skip-wandb to disable.
"""

import os
import sys
from pathlib import Path

WANDB_SKIP_MSG = "Use --skip-wandb to disable W&B logging, or run 'wandb login' to authenticate."


def setup_wandb(
    config: dict,
    project: str,
    run_name: str | None = None,
    wandb_dir: Path | None = None,
    *,
    api_key: str | None = None,
    offline: bool = False,
):
    """
    Initialize Weights & Biases logging. Raises SystemExit if W&B cannot be initialized.

    Supports:
    - wandb login credentials (~/.netrc) when no key is provided
    - WANDB_API_KEY env var or --wandb-api-key (for headless/cloud)
    - --wandb-offline for offline mode (no login required)

    Args:
        config: Config dict to log
        project: W&B project name
        run_name: Optional run name
        wandb_dir: Directory for wandb run files (avoids ./wandb shadowing)
        api_key: API key (overrides WANDB_API_KEY if provided)
        offline: Force offline mode (no login required)

    Returns:
        wandb module if successful

    Raises:
        SystemExit: If wandb not installed or not logged in (and not offline)
    """
    if wandb_dir is not None:
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    key = api_key or os.environ.get("WANDB_API_KEY")
    if offline:
        os.environ["WANDB_MODE"] = "offline"

    try:
        site_packages = [p for p in sys.path if "site-packages" in p]
        if site_packages:
            sys.path.insert(0, site_packages[0])
        import wandb
        if site_packages:
            sys.path.pop(0)
    except ImportError:
        raise SystemExit(
            f"W&B logging failed: wandb not installed. Install with: uv sync --extra train\n{WANDB_SKIP_MSG}"
        )

    if key:
        wandb.login(key=key, relogin=True)

    wandb.init(project=project, name=run_name, config=config)

    run = wandb.run
    if run:
        try:
            url = run.url
            print(f"W&B: Logging to project '{project}'. View run at {url}")
        except Exception:
            if offline or os.environ.get("WANDB_MODE") == "offline":
                print("W&B: Offline mode. Run 'wandb sync' in the run directory to upload.")
            else:
                print(f"W&B: Logging to project '{project}'")

    return wandb
