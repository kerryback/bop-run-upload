"""
noipca2 - Master Orchestration Script

Runs the complete workflow for a range of panel identifiers:
1. Generate panel data
2. Calculate SDF moments
3. Compute Fama factors
4. Compute DKKM factors (all nfeatures in one run — matches root approach)

Usage:
    python main.py [model] [start] [end] [--koyeb]

Arguments:
    model: Model name (bgn, kp14, gs21)
    start: Starting index (optional, default: 0)
    end: Ending index exclusive (optional, default: 1)
    --koyeb: Configure for Koyeb deployment (N_JOBS=24, TEMP_DIR=DATA_DIR)

ROOT REFERENCE: main_revised.py
  The root runs everything in one script (generate panel, compute factors,
  evaluate portfolio stats in run_month). noipca2 splits into separate scripts
  but the computation logic matches root exactly.

Examples:
    python main.py kp14                    # Single panel, index 0
    python main.py kp14 0 5                # Indices 0-4
    python main.py kp14 0 10 --koyeb       # Koyeb deployment
"""

import sys
import os
import time
import subprocess
import pickle
from datetime import datetime

# Add this directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Parse command-line arguments
if len(sys.argv) < 2:
    print("Usage: python main.py [model] [start] [end] [--koyeb]")
    print("  model: bgn, kp14, or gs21")
    print("  start: Starting index (default: 0)")
    print("  end:   Ending index exclusive (default: 1)")
    sys.exit(1)

model = sys.argv[1].lower()
start_idx = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 0
end_idx = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else start_idx + 1
koyeb_mode = '--koyeb' in sys.argv

# Validate model
if model not in ['bgn', 'kp14', 'gs21']:
    print(f"ERROR: Invalid model '{model}'. Use: bgn, kp14, gs21")
    sys.exit(1)

# Configure environment
if koyeb_mode:
    config.set_n_jobs(24)
    # On Koyeb, use DATA_DIR for everything (no separate scratch)
elif os.path.exists('/opt/scratch/keb7'):
    config.set_jgsrc1_config()

# Get references to config values
DATA_DIR = config.DATA_DIR
TEMP_DIR = config.TEMP_DIR
N_DKKM_FEATURES_LIST = config.N_DKKM_FEATURES_LIST
KEEP_PANEL = config.KEEP_PANEL
KEEP_MOMENTS = config.KEEP_MOMENTS

# S3 upload support
S3_CONFIGURED = bool(os.environ.get('AWS_ACCESS_KEY_ID'))


def upload_file(filepath):
    """Upload a file to S3 if configured. Raises on failure."""
    if not S3_CONFIGURED or not os.path.exists(filepath):
        return
    from utils.upload_to_aws import upload_file as _upload
    _upload(filepath)


def run_script(script_name, args, description):
    """Run a Python script as subprocess."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, script_name] + args
    start_time = time.time()

    # Use Popen with PIPE to handle TeeOutput (which lacks fileno())
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end='')
    process.wait()

    elapsed = time.time() - start_time

    if process.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {process.returncode}")
        sys.exit(1)

    return elapsed


def run_workflow_for_index(panel_id):
    """
    Run complete workflow for a single panel index.

    Steps:
    1. Generate panel → {model}_{id}_panel.pkl
    2. Calculate SDF moments → {model}_{id}_moments.pkl
    3. Compute Fama factors → {model}_{id}_fama.pkl
    4. Compute DKKM factors (all nfeatures at once) → {model}_{id}_dkkm_*.pkl
    """
    full_panel_id = f"{model}_{panel_id}"

    print(f"\n{'='*70}")
    print(f"RUNNING WORKFLOW FOR {full_panel_id.upper()}")
    print(f"{'='*70}")

    timings = {}

    # Step 1: Generate panel
    timings['panel'] = run_script(
        "utils/generate_panel.py",
        [model, str(panel_id)],
        f"STEP 1: Generating {model.upper()} panel data (index={panel_id})"
    )
    upload_file(os.path.join(TEMP_DIR, f"{full_panel_id}_panel.pkl"))

    # Step 2: Calculate SDF moments
    timings['moments'] = run_script(
        "utils/calculate_moments.py",
        [full_panel_id],
        "STEP 2: Calculating SDF conditional moments"
    )
    upload_file(os.path.join(TEMP_DIR, f"{full_panel_id}_moments.pkl"))

    # Step 3: Fama factors
    timings['fama'] = run_script(
        "utils/run_fama.py",
        [full_panel_id],
        "STEP 3: Computing Fama-French and Fama-MacBeth factors"
    )
    fama_file = os.path.join(DATA_DIR, f"{full_panel_id}_fama.pkl")
    upload_file(fama_file)

    # Step 4: DKKM factors (all nfeatures in one run — matches root approach)
    # ROOT: main_revised.py generates max_features once, then subsets for each nfeatures
    # noipca2/run_dkkm.py does the same: one W matrix, all nfeatures evaluated
    timings['dkkm'] = run_script(
        "utils/run_dkkm.py",
        [full_panel_id],
        f"STEP 4: Computing DKKM factors (all nfeatures: {N_DKKM_FEATURES_LIST})"
    )

    # Upload all DKKM output files
    for nfeatures in N_DKKM_FEATURES_LIST:
        dkkm_file = os.path.join(DATA_DIR, f"{full_panel_id}_dkkm_{nfeatures}.pkl")
        upload_file(dkkm_file)
    dkkm_all_file = os.path.join(DATA_DIR, f"{full_panel_id}_dkkm.pkl")
    upload_file(dkkm_all_file)

    # Cleanup
    if not KEEP_MOMENTS:
        moments_file = os.path.join(TEMP_DIR, f"{full_panel_id}_moments.pkl")
        if os.path.exists(moments_file):
            os.remove(moments_file)
            print(f"[CLEANUP] Deleted moments file")

    if not KEEP_PANEL:
        panel_file = os.path.join(TEMP_DIR, f"{full_panel_id}_panel.pkl")
        if os.path.exists(panel_file):
            os.remove(panel_file)
            print(f"[CLEANUP] Deleted panel file")

    # Summary
    total = sum(timings.values())
    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE FOR {full_panel_id.upper()}")
    print(f"{'='*70}")
    print(f"  1. Panel:   {timings['panel']:7.1f}s ({timings['panel']/60:.1f}min)")
    print(f"  2. Moments: {timings['moments']:7.1f}s ({timings['moments']/60:.1f}min)")
    print(f"  3. Fama:    {timings['fama']:7.1f}s ({timings['fama']/60:.1f}min)")
    print(f"  4. DKKM:    {timings['dkkm']:7.1f}s ({timings['dkkm']/60:.1f}min)")
    print(f"  Total:      {total:7.1f}s ({total/60:.1f}min)")


def delete_koyeb_service():
    """Delete the Koyeb service (for auto-cleanup after completion)."""
    app_name = os.environ.get('KOYEB_APP_NAME')
    service_name = os.environ.get('KOYEB_SERVICE_NAME')
    token = os.environ.get('KOYEB_API_TOKEN')

    if not all([app_name, service_name, token]):
        return

    print(f"\n[KOYEB] Deleting service {service_name} from app {app_name}...")
    try:
        result = subprocess.run(
            ['koyeb', 'services', 'delete', service_name,
             '--app', app_name, '--token', token],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(f"[KOYEB] Service deleted successfully")
        else:
            print(f"[KOYEB] Delete failed: {result.stderr}")
    except Exception as e:
        print(f"[KOYEB] Delete error: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    overall_start = time.time()

    # Set up logging
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model}_{start_idx}_{end_idx}.log")

    # Redirect stdout/stderr to log file
    import io
    class TeeOutput(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except ValueError:
                    pass
            return len(data)
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except ValueError:
                    pass

    log_fh = open(log_file, 'w')
    sys.stdout = TeeOutput(sys.__stdout__, log_fh)
    sys.stderr = TeeOutput(sys.__stderr__, log_fh)

    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Index range: {start_idx} to {end_idx-1} (inclusive)")
    print(f"  Total runs: {end_idx - start_idx}")
    print(f"  DKKM features: {N_DKKM_FEATURES_LIST}")
    print(f"  N_JOBS: {config.N_JOBS}")
    print(f"  Log file: {log_file}")
    print(f"\nDirectories:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  TEMP_DIR: {TEMP_DIR}")

    # Run workflow for each index
    for idx in range(start_idx, end_idx):
        run_workflow_for_index(idx)

    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"ALL WORKFLOWS COMPLETE")
    print(f"{'='*70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    # Koyeb auto-cleanup
    if koyeb_mode:
        delete_koyeb_service()

    # Restore original stdout/stderr before closing log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_fh.close()
