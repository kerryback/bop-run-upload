"""
noipca2 - Master Orchestration Script

Runs the complete workflow for a range of panel identifiers:
1.  Generate panel data
1b. Generate 25 double-sorted portfolios (mve x bm)
2.  Compute Fama factors (FF and FM)
3.  Compute DKKM factors (all nfeatures in one run)
4.  Estimate SDFs (compute stock weights via ridge regression)
5.  Calculate SDF moments
6.  Evaluate SDFs (compute portfolio statistics)

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
import shutil
from datetime import datetime

# Format seconds as h:m:s
def fmt(s):
    h, m, sec = int(s // 3600), int(s % 3600 // 60), int(s % 60)
    return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s"

def now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('America/Chicago')).strftime('%a %d %b %Y, %I:%M%p %Z')

# Add this directory and utils/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

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
KEEP_WEIGHTS = config.KEEP_WEIGHTS

# S3 upload support
S3_CONFIGURED = bool(os.environ.get('AWS_ACCESS_KEY_ID'))


def upload_file(filepath):
    """Upload a file to S3 if configured. Raises on failure."""
    if not S3_CONFIGURED or not os.path.exists(filepath):
        return
    from utils.upload_to_aws import upload_file as _upload
    _upload(filepath)


class ScriptError(Exception):
    """Raised when a subprocess script fails."""
    def __init__(self, script_name, returncode, output_tail=""):
        self.script_name = script_name
        self.returncode = returncode
        self.output_tail = output_tail
        super().__init__(f"{script_name} failed with return code {returncode}")


def run_script(script_name, args, description):
    """Run a Python script as subprocess."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, script_name] + args
    start_time = time.time()

    # Use Popen with PIPE to handle TeeOutput (which lacks fileno())
    # Keep last 100 lines for crash reporting
    output_lines = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)
        if len(output_lines) > 100:
            output_lines.pop(0)
    process.wait()

    elapsed = time.time() - start_time

    if process.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {process.returncode}")
        raise ScriptError(script_name, process.returncode, "".join(output_lines))

    return elapsed


def run_workflow_for_index(panel_id):
    """
    Run complete workflow for a single panel index.

    Steps:
    1.  Generate panel → {model}_{id}_panel.pkl
    1b. Generate 25 portfolios → {model}_{id}_25_portfolios.pkl
    2.  Compute Fama factors (in-process, kept in memory for Step 4)
    3.  Compute DKKM factors (in-process, kept in memory for Step 4)
    4.  Estimate SDFs (in-process, receives Fama + DKKM data) → {model}_{id}_stock_weights.pkl
    5.  Calculate SDF moments → {model}_{id}_moments.pkl
    6.  Evaluate SDFs (compute stats) → {model}_{id}_results.pkl
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
    if KEEP_PANEL:
        upload_file(os.path.join(TEMP_DIR, f"{full_panel_id}_panel.pkl"))

    # Step 1b: Generate 25 portfolios
    timings['25_portfolios'] = run_script(
        "utils/generate_25_portfolios.py",
        [full_panel_id],
        "STEP 1b: Computing 25 double-sorted portfolios"
    )
    upload_file(os.path.join(DATA_DIR, f"{full_panel_id}_25_portfolios.pkl"))

    # Diagnostic checkpoint: upload a trivial file to S3 to confirm process is alive
    checkpoint_file = os.path.join(DATA_DIR, f"{full_panel_id}_checkpoint.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'status': 'pre_fama', 'timestamp': now()}, f)
    upload_file(checkpoint_file)
    os.remove(checkpoint_file)
    print(f"[CHECKPOINT] Pre-fama checkpoint uploaded at {now()}")

    # Step 2: Fama factors
    timings['fama'] = run_script(
        "utils/generate_fama_factors.py",
        [full_panel_id],
        "STEP 2: Computing Fama-French and Fama-MacBeth factors"
    )
    # Factor files not uploaded to S3

    # Step 3: DKKM factors
    timings['dkkm'] = run_script(
        "utils/generate_dkkm_factors.py",
        [full_panel_id],
        f"STEP 3: Computing DKKM factors (max_features={config.MAX_FEATURES})"
    )
    # Factor files not uploaded to S3

    # Step 4: Estimate SDFs (compute stock weights)
    timings['estimate'] = run_script(
        "utils/estimate_sdfs.py",
        [full_panel_id],
        "STEP 4: Estimating SDFs (computing stock weights)"
    )

    if KEEP_WEIGHTS:
        upload_file(os.path.join(DATA_DIR, f"{full_panel_id}_stock_weights.pkl"))

    # Step 5: Calculate SDF moments
    timings['moments'] = run_script(
        "utils/calculate_moments.py",
        [full_panel_id],
        "STEP 5: Calculating SDF conditional moments"
    )
    if KEEP_MOMENTS:
        upload_file(os.path.join(TEMP_DIR, f"{full_panel_id}_moments.pkl"))

    # arr_tuple is no longer needed after moments are computed
    arr_dir = os.path.join(TEMP_DIR, f"{full_panel_id}_arr")
    if os.path.exists(arr_dir):
        shutil.rmtree(arr_dir)
        print(f"[CLEANUP] Deleted arr_tuple directory")

    # Step 6: Evaluate SDFs (compute portfolio statistics)
    timings['evaluate'] = run_script(
        "utils/evaluate_sdfs.py",
        [full_panel_id],
        "STEP 6: Evaluating SDFs (computing statistics)"
    )
    upload_file(os.path.join(DATA_DIR, f"{full_panel_id}_results.pkl"))

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

    if not KEEP_WEIGHTS:
        weights_file = os.path.join(DATA_DIR, f"{full_panel_id}_stock_weights.pkl")
        if os.path.exists(weights_file):
            os.remove(weights_file)
            print(f"[CLEANUP] Deleted weights file")

    # Summary
    total = sum(timings.values())
    print(f"\n{'='*70}")
    print(f"WORKFLOW COMPLETE FOR {full_panel_id.upper()} at {now()}")
    print(f"{'='*70}")
    print(f"  1.  Panel:      {fmt(timings['panel'])}")
    print(f"  1b. Portfolios: {fmt(timings['25_portfolios'])}")
    print(f"  2.  Fama:       {fmt(timings['fama'])}")
    print(f"  3.  DKKM:       {fmt(timings['dkkm'])}")
    print(f"  4.  Estimate:   {fmt(timings['estimate'])}")
    print(f"  5.  Moments:    {fmt(timings['moments'])}")
    print(f"  6.  Evaluate:   {fmt(timings['evaluate'])}")
    print(f"  Total:          {fmt(total)}")


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

    print(f"Started at {now()}")
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
        try:
            run_workflow_for_index(idx)
        except Exception as e:
            import traceback
            crash_tb = traceback.format_exc()
            print(f"\n[CRASH] Workflow failed for index {idx}:\n{crash_tb}")

            # Upload crash report to S3 for diagnosis
            crash_file = os.path.join(DATA_DIR, f"{model}_{idx}_crash.pkl")
            with open(crash_file, 'wb') as f:
                pickle.dump({
                    'error': str(e),
                    'type': type(e).__name__,
                    'traceback': crash_tb,
                    'output_tail': getattr(e, 'output_tail', ''),
                    'timestamp': now(),
                }, f)
            upload_file(crash_file)
            print(f"[CRASH] Crash report uploaded to S3")
            sys.exit(1)

    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"ALL WORKFLOWS COMPLETE at {now()}")
    print(f"{'='*70}")
    print(f"Total runtime: {fmt(total_time)}")

    # Koyeb auto-cleanup
    if koyeb_mode:
        delete_koyeb_service()

    # Restore original stdout/stderr before closing log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_fh.close()
