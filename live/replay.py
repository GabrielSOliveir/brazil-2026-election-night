"""Replays election night: feeds the 10 saved TSE snapshots to live.py in order, from a clean state.
Usage (repo root): python live/replay.py   -> writes data/work/live_log.csv; compare with data/derived/live_projection_log.csv"""
import glob, os, subprocess, sys
for f in ("data/work/live_state.pkl", "data/work/live_log.csv"):
    if os.path.exists(f): os.remove(f)
os.makedirs("data/work", exist_ok=True)
for snap in sorted(glob.glob("data/derived/live_snapshots/t2_*.csv")):
    hhmm = os.path.basename(snap)[3:7]; hora = f"{hhmm[:2]}:{hhmm[2:]}"
    print(f"--- {hora}", flush=True)
    subprocess.run([sys.executable, "-I", "live/live.py", snap, "--hora", hora], check=True)
