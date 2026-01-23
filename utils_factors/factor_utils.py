"""
Shared utilities for factor computation scripts.

Contains common functionality used by run_fama.py, run_dkkm.py, etc.

ROOT REFERENCE: These utilities wrap common patterns from main_revised.py:
  - prepare_panel: lines 62-70 of main_revised.py
  - parse_panel_arguments: command-line parsing (noipca addition)
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

try:
    from config import DATA_DIR, TEMP_DIR
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the noipca2 directory")
    sys.exit(1)


def parse_panel_arguments(script_name: str = "script",
                         additional_args: Optional[Dict[str, str]] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse panel_id and other arguments from command line.

    Args:
        script_name: Name of calling script (for error messages)
        additional_args: Dict of additional args to parse, e.g. {'nfeatures': 'int'}

    Returns:
        (panel_id, model_name, parsed_args_dict)
    """
    if len(sys.argv) < 2:
        print(f"ERROR: Panel ID required")
        print(f"\nUsage: python {script_name}.py [panel_id] [additional_args...]")
        sys.exit(1)

    panel_id = sys.argv[1]
    model_name = panel_id.split('_')[0].lower()

    parsed_args = {}
    if additional_args:
        arg_idx = 2
        for arg_name, arg_type in additional_args.items():
            if len(sys.argv) > arg_idx:
                try:
                    if arg_type == 'int':
                        parsed_args[arg_name] = int(sys.argv[arg_idx])
                    elif arg_type == 'float':
                        parsed_args[arg_name] = float(sys.argv[arg_idx])
                    else:
                        parsed_args[arg_name] = sys.argv[arg_idx]
                except (ValueError, IndexError):
                    print(f"ERROR: Invalid value for {arg_name}: {sys.argv[arg_idx]}")
                    sys.exit(1)
                arg_idx += 1

    return panel_id, model_name, parsed_args


def prepare_panel(panel: pd.DataFrame, chars: list) -> Tuple[pd.DataFrame, int, int]:
    """
    Clean and prepare panel data for factor computation.

    ROOT: main_revised.py lines 62-70
        panel["size"] = np.log(panel.mve)
        panel = panel[panel.month>=2]
        panel.replace([np.inf, -np.inf], np.nan, inplace=True)
        panel.set_index(["month", "firmid"], inplace=True)
        nans = panel[chars+["mve", "xret"]].isnull().any(axis=1)
        keep = nans[~nans].index
        panel = panel.loc[keep]
        start = panel.index.unique("month").min()
        end = panel.index.unique("month").max()

    Args:
        panel: Raw panel DataFrame
        chars: List of characteristic column names

    Returns:
        (cleaned_panel, start_month, end_month)
    """
    # ROOT line 62: add size = log(mve)
    panel = panel.reset_index()
    panel["size"] = np.log(panel.mve)

    # ROOT line 63: filter early months
    panel = panel[panel.month >= 2]

    # ROOT line 64: replace inf with NaN
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ROOT line 65: set multi-index
    panel.set_index(["month", "firmid"], inplace=True)

    # ROOT lines 66-68: remove NaN rows
    nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
    keep = nans[~nans].index
    panel = panel.loc[keep]

    # ROOT lines 69-70: get time range
    start = panel.index.unique("month").min()
    end = panel.index.unique("month").max()

    print(f"\nPanel after cleaning:")
    print(f"  Start month: {start}, End month: {end}")
    print(f"  Total observations: {len(panel)}")

    return panel, start, end


def save_factor_results(results: Dict[str, Any],
                        output_file: str,
                        verbose: bool = True) -> None:
    """Save factor computation results to pickle file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"\n[OK] Results saved to: {output_file}")
        print(f"     Keys: {list(results.keys())}")


def print_script_header(title: str, model: str, panel_id: str,
                       config, additional_info: Optional[Dict] = None) -> None:
    """Print standardized script header."""
    from datetime import datetime
    print("="*70)
    print(title)
    print("="*70)
    print(f"Model: {model}")
    print(f"Panel ID: {panel_id}")
    print(f"Configuration: N={config['N']}, T={config['T']}, n_jobs={config['n_jobs']}")
    if additional_info:
        for key, value in additional_info.items():
            print(f"{key}: {value}")
    print(f"Started at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    print("="*70)


def print_script_footer(panel_id: str = None,
                       usage_examples: Optional[list] = None) -> None:
    """Print standardized script footer."""
    from datetime import datetime
    print(f"\n{'='*70}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    if usage_examples:
        print(f"\nUsage examples:")
        for example in usage_examples:
            print(f"  {example}")
    print(f"{'='*70}")
