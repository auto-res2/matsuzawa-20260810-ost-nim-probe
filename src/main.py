"""Orchestrator for the shared OST / PLINDER feasibility probe.

Applies mode overrides and invokes src/inference.py as a subprocess, per the
repository's execution contract. It holds no probe logic of its own.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# The probe is I/O-bound and cheap, so the modes differ only in how many
# PLINDER systems are actually scored end to end.
MODE_SYSTEMS = {"sanity": 2, "pilot": 10, "full": 50}


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)

    mode = str(resolved.get("mode", "sanity"))
    resolved["scoring"]["n_systems"] = MODE_SYSTEMS.get(mode, 3)

    run_id = str(resolved.get("run", "probe"))
    results_dir = str(resolved.get("results_dir", ".research/results"))

    print(f"[main] mode={mode} run_id={run_id} results_dir={results_dir}")
    print(f"[main] scoring {resolved['scoring']['n_systems']} PLINDER system(s)")

    # Hydra changes the working directory; the results path is relative to
    # where the run was launched from, so resolve it before handing it over.
    orig_cwd = hydra.utils.get_original_cwd()
    results_dir = str((Path(orig_cwd) / results_dir).resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"cfg": resolved, "results_dir": results_dir, "run_id": run_id}, f)
        spec_path = f.name

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "src.inference", spec_path],
        cwd=orig_cwd,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
