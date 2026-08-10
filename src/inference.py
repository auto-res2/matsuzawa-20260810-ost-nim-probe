"""Feasibility probe: from inside a Seyval-launched Apptainer job on the
cluster, can we (1) run the shared OpenStructure CLI on real PLINDER data,
(2) launch the Boltz-2 NIM API server as a nested Apptainer container and get
a real prediction out of it, and (3) run DiffDock as a nested Apptainer
container end to end?

The question is not "do these tools work" — they demonstrably work when
launched by hand on the cluster — but whether the Seyval job container can
reach them: whether the account-level shared directory is bound in, whether
the cluster's apptainer binary is visible and can nest, and whether `--nv`
GPU passthrough survives the nesting.

Every check records what it saw rather than raising, so one failure does not
hide the results of the checks after it.

A hard-won detail encoded here: the shared `ost` launcher prints its usage
and exits 255 for `--help`, so "does the CLI run" must be judged by an
actual compare-ligand-structures invocation, never by the help exit code.
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _run(cmd, timeout, cwd=None, env=None):
    """Run a command, capturing everything. Never raises."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return {
            "cmd": " ".join(str(c) for c in cmd),
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:],
        }
    except FileNotFoundError as e:
        return {"cmd": " ".join(str(c) for c in cmd), "returncode": None, "error": f"not found: {e}"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(str(c) for c in cmd), "returncode": None, "error": f"timeout after {timeout}s"}
    except Exception as e:  # noqa: BLE001 - a probe must not die on an unexpected error
        return {"cmd": " ".join(str(c) for c in cmd), "returncode": None, "error": repr(e)}


def _path_facts(p):
    path = Path(p)
    facts = {"path": str(path), "exists": path.exists()}
    if not facts["exists"]:
        return facts
    facts["is_dir"] = path.is_dir()
    facts["readable"] = os.access(path, os.R_OK)
    facts["executable"] = os.access(path, os.X_OK)
    facts["writable"] = os.access(path, os.W_OK)
    if path.is_file():
        try:
            facts["size_bytes"] = path.stat().st_size
        except OSError:
            pass
    return facts


def check_environment():
    """What machine are we actually on, and are we inside Apptainer?

    The runtime variables recorded here (Slurm's and Apptainer's) are set by
    the scheduler and the container runtime themselves when present. None of
    them is required: every read has a None fallback and the probe runs to
    completion on a bare laptop. They are observations, not configuration.
    """
    runtime_snapshot = dict(os.environ)
    out = {
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "in_apptainer": "APPTAINER_CONTAINER" in runtime_snapshot,
        "apptainer_container": runtime_snapshot.get("APPTAINER_CONTAINER"),
        "slurm_job_id": runtime_snapshot.get("SLURM_JOB_ID"),
        "slurm_node": runtime_snapshot.get("SLURMD_NODENAME"),
    }
    try:
        out["os_release"] = Path("/etc/os-release").read_text().splitlines()[0]
    except OSError:
        out["os_release"] = None
    # GPU visibility in the OUTER container. Both nested tools need --nv to
    # survive nesting, and that has no chance if the job itself has no GPU.
    out["nvidia_smi"] = _run(["nvidia-smi", "-L"], timeout=60)
    return out


def check_mounts(account_dir):
    """Which filesystems are actually bound in?"""
    out = {}
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
        out["data1_mounts"] = [m for m in mounts if "/data1" in m]
        out["shared_software_mounts"] = [m for m in mounts if "/shared" in m]
    except OSError as e:
        out["error"] = repr(e)
    out["path_walk"] = [_path_facts(p) for p in ["/data1", account_dir]]
    return out


def pick_systems(test_dir, n):
    """First n PLINDER systems that have every file the probes need."""
    tdir = Path(test_dir)
    chosen = []
    try:
        entries = sorted(os.listdir(tdir))
    except OSError:
        return chosen
    for e in entries:
        s = tdir / e
        lig = s / "ligand_files"
        if not (s.is_dir() and (s / "system.cif").exists() and (s / "receptor.pdb").exists() and lig.is_dir()):
            continue
        sdfs = sorted(lig.glob("*.sdf"))
        if not sdfs:
            continue
        chosen.append({"system_id": e, "dir": str(s), "sdf": str(sdfs[0])})
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------- OST


def score_one_system(ost_bin, system_dir, workdir, radius, timeout):
    """Run a real lDDT-PLI computation end to end.

    The reference is scored against itself, so a working toolchain must
    return ~1.0. That distinguishes "the command ran" from "the command
    produced a meaningful number".
    """
    system_dir = Path(system_dir)
    ref = system_dir / "system.cif"
    out = {"system_id": system_dir.name}
    result_json = Path(workdir) / f"ost_{system_dir.name}.json"
    cmd = [
        ost_bin, "compare-ligand-structures",
        "-m", str(ref),
        "-r", str(ref),
        "--lddt-pli",
        "--lddt-pli-radius", str(radius),
        "--lddt-pli-add-mdl-contacts",
        "-o", str(result_json),
    ]
    out["run"] = _run(cmd, timeout=timeout, cwd=workdir)

    if not result_json.exists():
        out["verdict"] = "no_output_file"
        return out
    try:
        data = json.loads(result_json.read_text())
    except (OSError, json.JSONDecodeError) as e:
        out["verdict"] = "unparsable_output"
        out["error"] = repr(e)
        return out
    pli = data.get("lddt_pli") or {}
    scores = pli.get("assigned_scores") or []
    out["ost_version"] = data.get("ost_version")
    out["n_assigned"] = len(scores)
    out["scores"] = [
        {"model_ligand": s.get("model_ligand"), "reference_ligand": s.get("reference_ligand"), "score": s.get("score")}
        for s in scores
    ]
    vals = [s["score"] for s in out["scores"] if isinstance(s["score"], (int, float))]
    out["max_score"] = max(vals) if vals else None
    # Self-comparison must be ~1.0; anything less means the toolchain ran
    # but is not computing what we think it is.
    out["verdict"] = "scored" if (vals and max(vals) > 0.99) else ("ran_but_no_scores" if not scores else "scored_but_unexpected")
    return out


def check_ost(cfg, systems, workdir):
    sh, sc = cfg["shared"], cfg["scoring"]
    out = {"bin_facts": _path_facts(sh["ost_bin"]), "systems": []}
    if not out["bin_facts"]["exists"]:
        out["verdict"] = "absent"
        return out
    # Recorded for diagnosis only — the launcher exits 255 even on success
    # here, so nothing is decided from this.
    out["version"] = _run([sh["ost_bin"], "--version"], timeout=120)
    for s in systems:
        out["systems"].append(
            score_one_system(sh["ost_bin"], s["dir"], workdir, sc["lddt_pli_radius"], int(sc["timeout_sec"]))
        )
    scored = [s for s in out["systems"] if s.get("verdict") == "scored"]
    out["verdict"] = "scored" if scored else "present_but_fails"
    return out


# ---------------------------------------------------------------- nested Apptainer


def check_apptainer(ap):
    """Can the cluster's apptainer binary run at all from inside the job's
    own Apptainer container? Nesting is the single riskiest assumption in
    this whole probe, so it gets its own check before anything depends on it."""
    out = {"bin_facts": _path_facts(ap["bin"])}
    if not out["bin_facts"]["exists"]:
        out["verdict"] = "absent"
        return out
    out["version"] = _run([ap["bin"], "--version"], timeout=120)
    out["verdict"] = "runs" if out["version"].get("returncode") == 0 else "present_but_fails"
    return out


def _nested_env(ap):
    env = dict(os.environ)
    env["APPTAINER_CACHEDIR"] = ap["cachedir"]
    return env


def prepare_nested_session(workdir):
    """Point the nested apptainer's session base at the Lustre job directory.

    The inner apptainer wants its session under /var/lib/apptainer (compiled
    SESSIONDIR base). Inside the job container that path lands on the
    writable overlay, and squashfuse mounting the inner sif there produces
    an empty rootfs — the tool "runs" against a container with no /bin/sh.
    Verified on the login node: putting the session base on a real bound
    filesystem makes the same nested invocation work, and a symlink is
    enough because apptainer resolves the path before using it. So: replace
    /var/lib/apptainer with a symlink to a directory on the bound Lustre.
    """
    out = {}
    target = Path(workdir).resolve() / "apptainer_var"
    base = Path("/var/lib/apptainer")
    try:
        target.mkdir(parents=True, exist_ok=True)
        if base.is_symlink():
            base.unlink()
        elif base.exists():
            shutil.rmtree(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.symlink_to(target)
        out["session_base"] = str(target)
        out["verdict"] = "linked"
    except OSError as e:
        # A read-only outer rootfs would land here; the nested checks then
        # run anyway and their own errors say what actually happened.
        out["error"] = repr(e)
        out["verdict"] = "failed"
    return out


# ---------------------------------------------------------------- Boltz-2 NIM


def sample_memory(label):
    """GPU and host memory at one instant, for sizing the NIM server.

    Sampled at baseline / ready / after-predict rather than continuously: the
    interesting quantity is how much the resident server holds, and the
    difference between baseline and ready is exactly that. nvidia-smi reports
    the whole device, so on a shared node this includes other tenants — the
    baseline sample is what makes the delta meaningful anyway.
    """
    out = {"label": label}
    r = _run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        timeout=60,
    )
    if r.get("returncode") == 0:
        first = r["stdout"].strip().splitlines()[0]
        used, total = (p.strip() for p in first.split(","))
        out["gpu_used_mib"] = int(used)
        out["gpu_total_mib"] = int(total)
    # MemAvailable alone cannot say WHERE host memory went: page cache from
    # reading 16 GB of weights off Lustre, pinned CUDA buffers, and ordinary
    # process memory all move it. Record the components that separate them.
    wanted = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers",
              "Shmem", "SReclaimable", "SUnreclaim", "Mapped", "PageTables")
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, val = line.partition(":")
            if key in wanted:
                out[f"host_{key.lower()}_kb"] = int(val.split()[0])
    except OSError:
        pass

    # Resident memory of every process we can see, so "the server's own
    # footprint" is distinguishable from cache. Apptainer shares the PID
    # namespace by default, so the nested server's processes appear here.
    procs = []
    total_rss = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except (OSError, ProcessLookupError):
            continue
        name = rss = None
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(maxsplit=1)[1]
            elif line.startswith("VmRSS:"):
                rss = int(line.split()[1])
            if name and rss is not None:
                break
        if rss:
            total_rss += rss
            procs.append((rss, name, entry.name))
    out["host_total_rss_kb"] = total_rss
    procs.sort(reverse=True)
    out["top_rss"] = [{"pid": p, "name": n, "rss_kb": r} for r, n, p in procs[:6]]
    return out


def _http_get(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return {"status": r.status, "body": r.read(2000).decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _http_post_json(url, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"status": r.status, "json": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read()[:2000].decode("utf-8", "replace")}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def check_boltz2_api(cfg, workdir):
    """Launch the Boltz-2 NIM server exactly as the project README does,
    wait for /v1/health/ready, then send one real (small) predict request.

    "Connectivity" here means the full chain: sif readable -> nested
    apptainer launches -> model loads on the GPU -> HTTP answers -> a real
    prediction comes back.
    """
    nim, ap = cfg["nim"], cfg["apptainer"]
    out = {
        "sif": _path_facts(nim["sif"]),
        "env_file": _path_facts(nim["env_file"]),
        "cache_dir": _path_facts(nim["cache_dir"]),
        "workspace_dir": _path_facts(nim["workspace_dir"]),
    }
    if not (out["sif"]["exists"] and out["env_file"]["exists"]):
        out["verdict"] = "assets_missing"
        return out

    out["warmup"] = warmup_nested_mount(ap, nim["sif"])
    if out["warmup"]["verdict"] != "mounted":
        out["verdict"] = "mount_never_ready"
        return out

    cmd = [
        ap["bin"], "exec", "--nv",
        "--env-file", nim["env_file"],
        "--bind", f"{nim['cache_dir']}:/opt/nim/.cache",
        "--bind", f"{nim['workspace_dir']}:/opt/nim/workspace",
        nim["sif"],
        "/opt/nim/start_server.sh",
    ]
    out["server_cmd"] = " ".join(cmd)
    out["memory"] = [sample_memory("baseline_before_server")]
    log_path = Path(workdir) / "boltz2_server.log"
    ready_url = nim["base_url"] + "/v1/health/ready"
    deadline = time.time() + int(nim["ready_timeout_sec"])
    proc = None
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=_nested_env(ap), start_new_session=True)
            while time.time() < deadline:
                if proc.poll() is not None:
                    out["server_exit_code"] = proc.returncode
                    break
                r = _http_get(ready_url, timeout=10)
                if r.get("status") == 200:
                    out["ready"] = r
                    out["ready_after_sec"] = round(int(nim["ready_timeout_sec"]) - (deadline - time.time()))
                    break
                time.sleep(10)

        if "ready" not in out:
            out["verdict"] = "server_died" if "server_exit_code" in out else "ready_timeout"
            return out

        out["memory"].append(sample_memory("server_ready"))

        # One real prediction, sized to finish quickly: a short protein and
        # aspirin. Anything with structures[] in the response proves the
        # whole inference path, which is all this probe claims.
        payload = {
            "polymers": [{
                "id": "A",
                "molecule_type": "protein",
                "sequence": "MTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPT",
            }],
            "ligands": [{"id": "L1", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}],
            "recycling_steps": 1,
            "sampling_steps": 30,
            "diffusion_samples": 1,
            "output_format": "mmcif",
        }
        t0 = time.time()
        resp = _http_post_json(nim["base_url"] + "/biology/mit/boltz2/predict", payload, int(nim["request_timeout_sec"]))
        out["predict_sec"] = round(time.time() - t0, 1)
        out["memory"].append(sample_memory("after_predict"))
        if resp.get("status") == 200 and resp.get("json", {}).get("structures"):
            j = resp["json"]
            out["predict"] = {
                "status": 200,
                "n_structures": len(j["structures"]),
                "confidence_scores": j.get("confidence_scores"),
                "structure_chars": len(j["structures"][0].get("structure", "")),
            }
            (Path(workdir) / "boltz2_structure_1.cif").write_text(j["structures"][0].get("structure", ""))
            out["verdict"] = "predicted"
        else:
            out["predict"] = {k: v for k, v in resp.items() if k != "json"}
            out["verdict"] = "ready_but_predict_failed"
        return out
    finally:
        if proc is not None and proc.poll() is None:
            # The server is a process group (start_server.sh spawns workers);
            # kill the group or the job hangs at teardown until walltime.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            out["server_log_tail"] = log_path.read_text()[-3000:]
        except OSError:
            pass


# ---------------------------------------------------------------- DiffDock


def warmup_nested_mount(ap, sif, tries=3, delay_sec=15):
    """Mount the inner sif until a file from its rootfs is actually visible.

    Observed on this cluster: the same nested invocation intermittently sees
    an EMPTY inner rootfs ("No runscript and no /bin/sh") and then works on
    a later attempt — consistent with the squashfuse mount not being ready
    when apptainer proceeds, e.g. on a cold Lustre read of the sif. So the
    probe warms the mount with a trivial exec, retrying with a delay, and
    records every attempt: a pass/fail pattern across attempts is exactly
    the evidence a one-shot check cannot give.
    """
    out = {"attempts": []}
    for i in range(tries):
        if i:
            time.sleep(delay_sec)
        r = _run([ap["bin"], "exec", sif, "/bin/true"], timeout=300, env=_nested_env(ap))
        out["attempts"].append(r)
        if r.get("returncode") == 0:
            out["verdict"] = "mounted"
            out["attempts_needed"] = i + 1
            return out
    out["verdict"] = "never_mounted"
    return out


def check_diffdock(cfg, system, workdir):
    """One real docking run via nested `apptainer run --nv`, following the
    project README: receptor.pdb + one ligand SDF, rank1.sdf as proof.

    Bind paths must be absolute, and the output directory must be bound
    explicitly — inside the nested container only what we bind exists.
    """
    dd, ap, sh = cfg["diffdock"], cfg["apptainer"], cfg["shared"]
    out = {"sif": _path_facts(dd["sif"])}
    if not out["sif"]["exists"]:
        out["verdict"] = "absent"
        return out
    if system is None:
        out["verdict"] = "no_input_system"
        return out

    out["warmup"] = warmup_nested_mount(cfg["apptainer"], dd["sif"])
    if out["warmup"]["verdict"] != "mounted":
        out["verdict"] = "mount_never_ready"
        return out

    out_dir = (Path(workdir) / "diffdock_out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    complex_name = "probe1"
    cmd = [
        ap["bin"], "run", "--nv",
        "--bind", str(out_dir),
        "--bind", sh["plinder_test_dir"],
        dd["sif"],
        "--config", "/opt/DiffDock/default_inference_args.yaml",
        "--protein_path", str(Path(system["dir"]) / "receptor.pdb"),
        "--ligand_description", system["sdf"],
        "--complex_name", complex_name,
        "--out_dir", str(out_dir),
    ]
    out["system_id"] = system["system_id"]
    out["run"] = _run(cmd, timeout=int(dd["timeout_sec"]), env=_nested_env(ap))

    rank1 = out_dir / complex_name / "rank1.sdf"
    poses = sorted((out_dir / complex_name).glob("rank*_confidence*.sdf")) if (out_dir / complex_name).is_dir() else []
    out["rank1_exists"] = rank1.exists()
    out["n_poses"] = len(poses)
    if poses:
        # The confidence is in the filename (rankN_confidence<val>.sdf).
        out["pose_files"] = [p.name for p in poses[:3]]
    # DiffDock can swallow per-complex failures and exit 0, so the product
    # file — not the exit code — is the verdict.
    out["verdict"] = "docked" if rank1.exists() else "no_pose_produced"
    return out


# ---------------------------------------------------------------- driver


def run(cfg, results_dir, run_id):
    sh = cfg["shared"]

    outdir = Path(results_dir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    report = {"run_id": run_id}
    report["environment"] = check_environment()
    report["mounts"] = check_mounts(sh["account_dir"])
    report["apptainer"] = check_apptainer(cfg["apptainer"])

    systems = pick_systems(sh["plinder_test_dir"], int(cfg["scoring"]["n_systems"]))
    report["plinder_systems"] = [s["system_id"] for s in systems]

    report["ost"] = check_ost(cfg, systems, outdir)

    nested_ok = report["apptainer"]["verdict"] == "runs"
    if nested_ok:
        report["nested_session"] = prepare_nested_session(outdir)
        report["diffdock"] = check_diffdock(cfg, systems[0] if systems else None, outdir)
        report["boltz2_api"] = check_boltz2_api(cfg, outdir)
    else:
        report["diffdock"] = {"verdict": "skipped_no_apptainer"}
        report["boltz2_api"] = {"verdict": "skipped_no_apptainer"}

    (outdir / "probe_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def summarize(report):
    """Print a readable verdict block plus the machine-parsed line."""
    env = report["environment"]
    print("=" * 70)
    print("OST / DIFFDOCK / BOLTZ-2 NIM CONNECTIVITY PROBE")
    print("=" * 70)
    print(f"arch={env['arch']}  in_apptainer={env['in_apptainer']}  node={env['slurm_node']}")
    gpu = env["nvidia_smi"]
    print(f"[gpu]        rc={gpu.get('returncode')} {gpu.get('stdout', gpu.get('error', '')).strip()[:120]}")
    print(f"[mounts]     account dir visible: {report['mounts']['path_walk'][1]['exists']}")
    print(f"[apptainer]  {report['apptainer']['verdict']}")
    ost = report["ost"]
    for s in ost.get("systems", []):
        print(f"[ost]        {s['system_id']}: {s['verdict']} max={s.get('max_score')}")
    print(f"[ost]        verdict: {ost['verdict']}")
    dd = report["diffdock"]
    print(f"[diffdock]   {dd['verdict']} n_poses={dd.get('n_poses')}")
    api = report["boltz2_api"]
    ready = api.get("ready_after_sec")
    pred = api.get("predict", {})
    print(f"[boltz2 api] {api['verdict']} ready_after={ready}s predict_sec={api.get('predict_sec')} "
          f"confidence={pred.get('confidence_scores')}")
    for m in api.get("memory", []):
        mib = lambda k: m.get(k, 0) // 1024  # noqa: E731 - local formatting shorthand
        print(f"[memory]     {m['label']}: gpu={m.get('gpu_used_mib')} MiB  "
              f"avail={mib('host_memavailable_kb')}  free={mib('host_memfree_kb')}  "
              f"cached={mib('host_cached_kb')}  shmem={mib('host_shmem_kb')}  "
              f"rss_total={mib('host_total_rss_kb')} MiB")
        for p in m.get("top_rss", [])[:3]:
            print(f"[memory]       top rss: {p['name']} {p['rss_kb'] // 1024} MiB")
    print("=" * 70)

    ok = (
        ost["verdict"] == "scored"
        and dd["verdict"] == "docked"
        and api["verdict"] == "predicted"
    )
    if ok:
        print("SANITY_VALIDATION: PASS")
    else:
        failed = [
            f"{name}_{v['verdict']}"
            for name, v in (("ost", ost), ("diffdock", dd), ("boltz2_api", api))
            if v["verdict"] not in ("scored", "docked", "predicted")
        ]
        print(f"SANITY_VALIDATION: FAIL reason={','.join(failed)}")
    print("SANITY_VALIDATION_SUMMARY: " + json.dumps({
        "account_dir_visible": report["mounts"]["path_walk"][1]["exists"],
        "apptainer_nested": report["apptainer"]["verdict"],
        "ost": ost["verdict"],
        "diffdock": dd["verdict"],
        "boltz2_api": api["verdict"],
    }))
    return ok


if __name__ == "__main__":
    # Invoked as a subprocess by src/main.py, which passes the resolved
    # config as a JSON file so this module needs no configuration framework
    # of its own.
    spec = json.loads(Path(sys.argv[1]).read_text())
    rep = run(spec["cfg"], spec["results_dir"], spec["run_id"])
    sys.exit(0 if summarize(rep) else 1)
