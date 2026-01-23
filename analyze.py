"""
noipca2 - Analysis Script

Produces LaTeX tables and figures from noipca2 results.

This script automatically discovers and analyzes all panels for each model (BGN, KP14, GS21):
- Uses pre-computed HJD² from the stats (errs @ second_moment_inv @ errs, no sqrt)
- Computes sharpe = mean / stdev for each month
- Aggregates by panel, then across panels

ROOT REFERENCE: The HJD values stored are HJD² (main_revised.py line 257-258):
  errs = rp - second_moment @ weights_on_stocks
  hjd = errs @ second_moment_inv @ errs  # no sqrt!
For display, we take sqrt(mean(hjd)) across months.

Creates 9 LaTeX tables (3 for each model):
1. Fama table (FFC/FMR sharpe and hjd)
2-3. DKKM sharpe and hjd tables

Usage:
    python analyze.py

Outputs:
    - LaTeX tables saved to tables/
    - Boxplot figures saved to figures/
    - Combined PDF in tables/all_results.pdf
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Output directories
SCRIPT_DIR = Path(__file__).parent
TABLES_DIR = SCRIPT_DIR / "tables"
FIGURES_DIR = SCRIPT_DIR / "figures"
KOYEB_RESULTS_DIR = SCRIPT_DIR / "s3_results"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# Create output directories if they don't exist
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Configuration
MODELS = ['bgn', 'kp14', 'gs21']


def get_results_dir() -> Path:
    """
    Determine which results directory to use.
    Prefers s3_results/ if it exists and has files, otherwise uses outputs/.
    """
    if KOYEB_RESULTS_DIR.exists() and any(KOYEB_RESULTS_DIR.glob("*.pkl")):
        return KOYEB_RESULTS_DIR
    return OUTPUTS_DIR


def discover_panels(model: str, results_dir: Path) -> List[int]:
    """
    Discover all available panel indices for a given model.

    Scans results directory for files matching: {model}_{panel_idx}_fama.pkl

    Args:
        model: Model name ('bgn', 'kp14', 'gs21')
        results_dir: Directory to scan for results

    Returns:
        Sorted list of panel_idx integers
    """
    panels = set()

    if not results_dir.exists():
        print(f"  WARNING: {results_dir} does not exist")
        return []

    # Pattern: model_*_fama.pkl
    pattern = str(results_dir / f"{model}_*_fama.pkl")

    for filepath in glob.glob(pattern):
        # Extract panel_idx from filename
        filename = Path(filepath).stem  # "kp14_0_fama"
        parts = filename.split('_')

        # Format: model_panelid_fama
        if len(parts) >= 3 and parts[-1] == 'fama':
            try:
                panel_idx = int(parts[-2])
                panels.add(panel_idx)
            except ValueError:
                continue

    return sorted(list(panels))


def load_and_process_fama(model: str, panels: List[int], results_dir: Path) -> pd.DataFrame:
    """
    Load Fama results and compute sharpe from the stored stats.

    ROOT REFERENCE:
      The fama_stats DataFrame from run_fama.py contains:
        month, method, alpha, stdev, mean, xret, sdf_ret, hjd
      where hjd = errs @ second_moment_inv @ errs (HJD², no sqrt)

    For each month:
    - sharpe = mean / stdev
    - hjd is already HJD² (stored directly from root computation)

    Args:
        model: Model name
        panels: List of panel_idx integers
        results_dir: Directory containing result files

    Returns DataFrame with panel-level data.
    """
    all_data = []

    for panel_id, panel_idx in enumerate(panels):
        filename = f"{model}_{panel_idx}_fama.pkl"
        fama_file = results_dir / filename

        if not fama_file.exists():
            continue

        with open(fama_file, 'rb') as f:
            fama_data = pickle.load(f)

        fama_stats = fama_data.get('fama_stats')
        if fama_stats is None or len(fama_stats) == 0:
            continue

        # Compute sharpe for each month
        fama_stats = fama_stats.copy()
        fama_stats['sharpe'] = fama_stats['mean'] / fama_stats['stdev']

        # ROOT: hjd column is already HJD² (errs @ second_moment_inv @ errs)
        # Rename to hjd_sq for consistency with aggregation logic
        fama_stats['hjd_sq'] = fama_stats['hjd']

        # Use sequential panel_id for aggregation
        fama_stats['panel'] = panel_id

        all_data.append(fama_stats[['panel', 'month', 'method', 'alpha', 'sharpe', 'hjd_sq']])

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def load_and_process_dkkm(model: str, panels: List[int], results_dir: Path) -> pd.DataFrame:
    """
    Load DKKM results and compute sharpe from the stored stats.

    ROOT REFERENCE:
      The dkkm_stats DataFrame from run_dkkm.py contains:
        month, matrix, nfeatures, alpha, include_mkt, stdev, mean, xret, sdf_ret, hjd
      where hjd = errs @ second_moment_inv @ errs (HJD², no sqrt)

    Args:
        model: Model name
        panels: List of panel_idx integers
        results_dir: Directory containing result files

    Returns DataFrame with panel-level data.
    """
    all_data = []

    for panel_id, panel_idx in enumerate(panels):
        # Try combined file first: model_panelid_dkkm.pkl
        combined_file = results_dir / f"{model}_{panel_idx}_dkkm.pkl"

        if combined_file.exists():
            with open(combined_file, 'rb') as f:
                dkkm_data = pickle.load(f)

            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is not None and len(dkkm_stats) > 0:
                dkkm_stats = dkkm_stats.copy()
                dkkm_stats['sharpe'] = dkkm_stats['mean'] / dkkm_stats['stdev']
                # ROOT: hjd column is already HJD²
                dkkm_stats['hjd_sq'] = dkkm_stats['hjd']
                dkkm_stats['panel'] = panel_id
                # Column is 'nfeatures' in noipca2 (renamed to 'num_factors' for table display)
                dkkm_stats['num_factors'] = dkkm_stats['nfeatures']
                all_data.append(dkkm_stats[['panel', 'month', 'alpha', 'num_factors', 'sharpe', 'hjd_sq']])
                continue

        # Fallback: per-nfeatures files: model_panelid_dkkm_*.pkl
        pattern = str(results_dir / f"{model}_{panel_idx}_dkkm_*.pkl")

        for filepath in glob.glob(pattern):
            # Skip weight matrix files
            if filepath.endswith('_W.pkl'):
                continue

            # Extract nfeatures from filename: model_n_dkkm_features.pkl
            filename = Path(filepath).stem
            parts = filename.split('_')

            if len(parts) < 4:
                continue
            try:
                nfeatures = int(parts[3])
            except ValueError:
                continue

            with open(filepath, 'rb') as f:
                dkkm_data = pickle.load(f)

            dkkm_stats = dkkm_data.get('dkkm_stats')
            if dkkm_stats is None or len(dkkm_stats) == 0:
                continue

            dkkm_stats = dkkm_stats.copy()
            dkkm_stats['sharpe'] = dkkm_stats['mean'] / dkkm_stats['stdev']
            # ROOT: hjd column is already HJD²
            dkkm_stats['hjd_sq'] = dkkm_stats['hjd']
            dkkm_stats['panel'] = panel_id
            dkkm_stats['num_factors'] = nfeatures

            all_data.append(dkkm_stats[['panel', 'month', 'alpha', 'num_factors', 'sharpe', 'hjd_sq']])

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def create_fama_table(fama_df: pd.DataFrame, model: str, output_path: str):
    """
    Create Fama table for a single model.

    Rows: alpha values
    Columns: (FFC, sharpe), (FMR, sharpe), (FFC, hjd), (FMR, hjd)
    Values: Mean across panels

    HJD display: sqrt(mean(hjd_sq)) per panel, then mean across panels
    (since hjd_sq is the stored HJD² from root computation)
    """
    if len(fama_df) == 0:
        print(f"  WARNING: No Fama data for {model}")
        return

    alphas = sorted(fama_df['alpha'].unique())

    table_data = []
    for alpha in alphas:
        alpha_data = fama_df[fama_df['alpha'] == alpha]

        row = {'alpha': alpha}

        # FFC sharpe: mean across panels of (mean across months)
        ffc_data = alpha_data[alpha_data['method'] == 'ff']
        if len(ffc_data) > 0:
            ffc_panel_sharpe = ffc_data.groupby('panel')['sharpe'].mean()
            row['FFC_sharpe'] = ffc_panel_sharpe.mean()
        else:
            row['FFC_sharpe'] = np.nan

        # FMR sharpe: mean across panels of (mean across months)
        fmr_data = alpha_data[alpha_data['method'] == 'fm']
        if len(fmr_data) > 0:
            fmr_panel_sharpe = fmr_data.groupby('panel')['sharpe'].mean()
            row['FMR_sharpe'] = fmr_panel_sharpe.mean()
        else:
            row['FMR_sharpe'] = np.nan

        # FFC hjd: mean across panels of sqrt(mean(hjd_sq))
        # ROOT: hjd_sq is already errs @ M_inv @ errs (HJD²)
        if len(ffc_data) > 0:
            ffc_panel_hjd = ffc_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
            row['FFC_hjd'] = ffc_panel_hjd.mean()
        else:
            row['FFC_hjd'] = np.nan

        # FMR hjd: mean across panels of sqrt(mean(hjd_sq))
        if len(fmr_data) > 0:
            fmr_panel_hjd = fmr_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
            row['FMR_hjd'] = fmr_panel_hjd.mean()
        else:
            row['FMR_hjd'] = np.nan

        table_data.append(row)

    df = pd.DataFrame(table_data)
    df = df.set_index('alpha')

    # Create LaTeX table
    latex = df.to_latex(float_format="%.4f", na_rep="--",
                        column_format='r' + 'r'*len(df.columns),
                        escape=False)

    caption = f"""\\caption{{\\textbf{{Fama-French Performance - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show mean Sharpe ratio and
    Hansen-Jagannathan distance across panels for FFC and FMR methods.}}
\\label{{tab:fama_{model}}}"""

    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"  Saved {output_path}")


def create_dkkm_tables(dkkm_df: pd.DataFrame, model: str, sharpe_path: str, hjd_path: str):
    """
    Create DKKM sharpe and hjd tables for a single model.

    Rows: alpha values
    Columns: num_factors (nfeatures from root)
    Values: Mean across panels

    HJD display: sqrt(mean(hjd_sq)) per panel, then mean across panels
    """
    if len(dkkm_df) == 0:
        print(f"  WARNING: No DKKM data for {model}")
        return

    alphas = sorted(dkkm_df['alpha'].unique())
    num_factors_vals = sorted(dkkm_df['num_factors'].unique())

    # Create sharpe table: mean across panels of (mean across months)
    sharpe_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            if len(nf_data) > 0:
                panel_sharpe = nf_data.groupby('panel')['sharpe'].mean()
                row[nf] = panel_sharpe.mean()
            else:
                row[nf] = np.nan
        sharpe_table.append(row)

    sharpe_df = pd.DataFrame(sharpe_table).set_index('alpha')

    # Create hjd table: mean across panels of sqrt(mean(hjd_sq))
    hjd_table = []
    for alpha in alphas:
        row = {'alpha': alpha}
        alpha_data = dkkm_df[dkkm_df['alpha'] == alpha]
        for nf in num_factors_vals:
            nf_data = alpha_data[alpha_data['num_factors'] == nf]
            if len(nf_data) > 0:
                panel_hjd = nf_data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                row[nf] = panel_hjd.mean()
            else:
                row[nf] = np.nan
        hjd_table.append(row)

    hjd_df = pd.DataFrame(hjd_table).set_index('alpha')

    # Save sharpe table
    latex = sharpe_df.to_latex(float_format="%.4f", na_rep="--",
                                column_format='r' + 'r'*len(sharpe_df.columns),
                                escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Sharpe Ratios - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean Sharpe ratios across panels.}}
\\label{{tab:dkkm_sharpe_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(sharpe_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {sharpe_path}")

    # Save hjd table
    latex = hjd_df.to_latex(float_format="%.4f", na_rep="--",
                             column_format='r' + 'r'*len(hjd_df.columns),
                             escape=False)
    caption = f"""\\caption{{\\textbf{{DKKM Hansen-Jagannathan Distances - {model.upper()}}}.
    Rows show different $\\alpha$ values. Columns show number of DKKM factors.
    Values are mean HJ distances across panels.}}
\\label{{tab:dkkm_hjd_{model}}}"""
    latex = latex.replace(r'\end{tabular}', r'\end{tabular}' + '\n' + caption)
    latex = r'\begin{table}[h]' + '\n' + r'\centering' + '\n' + latex + '\n' + r'\end{table}'

    with open(hjd_path, 'w') as f:
        f.write(latex)
    print(f"  Saved {hjd_path}")


def create_fama_boxplots(fama_df: pd.DataFrame, model: str):
    """Create Fama boxplots for sharpe and hjd distributions across panels."""
    if len(fama_df) == 0:
        return

    alphas = sorted(fama_df['alpha'].unique())

    # Sharpe boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for method, method_label in [('ff', 'FFC'), ('fm', 'FMR')]:
            data = fama_df[(fama_df['alpha'] == alpha) & (fama_df['method'] == method)]
            if len(data) > 0:
                panel_sharpe = data.groupby('panel')['sharpe'].mean()
                boxplot_data.append(panel_sharpe.values)
                labels.append(f"{method_label}\n$\\alpha$={alpha:.1e}")

    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='steelblue',
                markerfacecolor='lightblue', markeredgewidth=1.5, markeredgecolor='steelblue')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(f'{model.upper()} - Fama Sharpe Ratios (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sharpe_path = FIGURES_DIR / f"{model}_fama_sharpe_boxplot.pdf"
    plt.savefig(sharpe_path, bbox_inches='tight')
    print(f"  Saved {sharpe_path}")
    plt.close()

    # HJD boxplot
    fig, ax = plt.subplots(figsize=(12, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for method, method_label in [('ff', 'FFC'), ('fm', 'FMR')]:
            data = fama_df[(fama_df['alpha'] == alpha) & (fama_df['method'] == method)]
            if len(data) > 0:
                panel_hjd = data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                boxplot_data.append(panel_hjd.values)
                labels.append(f"{method_label}\n$\\alpha$={alpha:.1e}")

    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='indianred',
                markerfacecolor='lightcoral', markeredgewidth=1.5, markeredgecolor='indianred')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        plt.xticks(rotation=45, ha='right')

    ax.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax.set_title(f'{model.upper()} - Fama HJ Distances (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    hjd_path = FIGURES_DIR / f"{model}_fama_hjd_boxplot.pdf"
    plt.savefig(hjd_path, bbox_inches='tight')
    print(f"  Saved {hjd_path}")
    plt.close()


def create_dkkm_boxplots(dkkm_df: pd.DataFrame, model: str):
    """Create DKKM boxplots for sharpe and hjd distributions across panels."""
    if len(dkkm_df) == 0:
        return

    alphas = sorted(dkkm_df['alpha'].unique())
    num_factors_vals = sorted(dkkm_df['num_factors'].unique())

    # Sharpe boxplot
    fig, ax = plt.subplots(figsize=(14, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for nf in num_factors_vals:
            data = dkkm_df[(dkkm_df['alpha'] == alpha) & (dkkm_df['num_factors'] == nf)]
            if len(data) > 0:
                panel_sharpe = data.groupby('panel')['sharpe'].mean()
                boxplot_data.append(panel_sharpe.values)
                labels.append(f"$\\alpha$={alpha:.1e}\nn={nf}")

    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='steelblue',
                markerfacecolor='lightblue', markeredgewidth=1.5, markeredgecolor='steelblue')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(f'{model.upper()} - DKKM Sharpe Ratios (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    sharpe_path = FIGURES_DIR / f"{model}_dkkm_sharpe_boxplot.pdf"
    plt.savefig(sharpe_path, bbox_inches='tight')
    print(f"  Saved {sharpe_path}")
    plt.close()

    # HJD boxplot
    fig, ax = plt.subplots(figsize=(14, 6))

    boxplot_data = []
    labels = []

    for alpha in alphas:
        for nf in num_factors_vals:
            data = dkkm_df[(dkkm_df['alpha'] == alpha) & (dkkm_df['num_factors'] == nf)]
            if len(data) > 0:
                panel_hjd = data.groupby('panel')['hjd_sq'].apply(lambda x: np.sqrt(x.mean()))
                boxplot_data.append(panel_hjd.values)
                labels.append(f"$\\alpha$={alpha:.1e}\nn={nf}")

    all_single = all(len(d) == 1 for d in boxplot_data)

    if all_single:
        positions = range(1, len(boxplot_data) + 1)
        values = [d[0] for d in boxplot_data]
        ax.plot(positions, values, 'o', markersize=8, color='indianred',
                markerfacecolor='lightcoral', markeredgewidth=1.5, markeredgecolor='indianred')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    else:
        bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        plt.xticks(rotation=45, ha='right', fontsize=8)

    ax.set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    ax.set_title(f'{model.upper()} - DKKM HJ Distances (Distribution Across Panels)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    hjd_path = FIGURES_DIR / f"{model}_dkkm_hjd_boxplot.pdf"
    plt.savefig(hjd_path, bbox_inches='tight')
    print(f"  Saved {hjd_path}")
    plt.close()


def generate_pdf():
    """Generate a PDF containing all LaTeX tables and figures."""
    import subprocess

    master_tex = TABLES_DIR / "all_results.tex"

    with open(master_tex, 'w') as f:
        f.write(r"\documentclass[11pt]{article}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{graphicx}" + "\n")
        f.write(r"\usepackage{geometry}" + "\n")
        f.write(r"\geometry{margin=1in}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write("\n")
        f.write(r"\title{noipca2 Factor Model Results}" + "\n")
        f.write(r"\author{}" + "\n")
        f.write(r"\date{\today}" + "\n")
        f.write(r"\maketitle" + "\n")
        f.write(r"\tableofcontents" + "\n")
        f.write(r"\clearpage" + "\n")
        f.write("\n")

        for model in MODELS:
            f.write(f"\n\\section{{{model.upper()} Model Results}}\n\n")

            # Fama results
            f.write(f"\\subsection{{Fama-French Results}}\n\n")

            fama_file = f"{model}_fama.tex"
            if (TABLES_DIR / fama_file).exists():
                f.write(f"\\input{{{fama_file}}}\n")
                f.write("\\clearpage\n\n")

            fama_sharpe_fig = f"../figures/{model}_fama_sharpe_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_fama_sharpe_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{fama_sharpe_fig}}}\n")
                f.write(f"\\caption{{Fama Sharpe Ratio Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            fama_hjd_fig = f"../figures/{model}_fama_hjd_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_fama_hjd_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{fama_hjd_fig}}}\n")
                f.write(f"\\caption{{Fama HJ Distance Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            # DKKM results
            f.write(f"\\subsection{{DKKM Results}}\n\n")

            dkkm_sharpe_file = f"{model}_dkkm_sharpe.tex"
            if (TABLES_DIR / dkkm_sharpe_file).exists():
                f.write(f"\\input{{{dkkm_sharpe_file}}}\n")
                f.write("\\clearpage\n\n")

            dkkm_hjd_file = f"{model}_dkkm_hjd.tex"
            if (TABLES_DIR / dkkm_hjd_file).exists():
                f.write(f"\\input{{{dkkm_hjd_file}}}\n")
                f.write("\\clearpage\n\n")

            dkkm_sharpe_fig = f"../figures/{model}_dkkm_sharpe_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_dkkm_sharpe_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{dkkm_sharpe_fig}}}\n")
                f.write(f"\\caption{{DKKM Sharpe Ratio Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

            dkkm_hjd_fig = f"../figures/{model}_dkkm_hjd_boxplot.pdf"
            if (FIGURES_DIR / f"{model}_dkkm_hjd_boxplot.pdf").exists():
                f.write("\\begin{figure}[h]\n")
                f.write("\\centering\n")
                f.write(f"\\includegraphics[width=0.9\\textwidth]{{{dkkm_hjd_fig}}}\n")
                f.write(f"\\caption{{DKKM HJ Distance Distribution - {model.upper()}}}\n")
                f.write("\\end{figure}\n")
                f.write("\\clearpage\n\n")

        f.write(r"\end{document}" + "\n")

    print(f"  Created master LaTeX file: {master_tex}")

    # Compile PDF
    try:
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', 'all_results.tex'],
                cwd=str(TABLES_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

        pdf_file = TABLES_DIR / "all_results.pdf"
        if pdf_file.exists():
            print(f"  PDF generated successfully: {pdf_file}")

            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out', '.toc']:
                aux_file = TABLES_DIR / f"all_results{ext}"
                if aux_file.exists():
                    aux_file.unlink()
        else:
            print(f"  WARNING: PDF generation failed (pdflatex may not be installed)")
            print(f"  You can manually compile {master_tex} with pdflatex")

    except FileNotFoundError:
        print(f"  WARNING: pdflatex not found - skipping PDF generation")
        print(f"  You can manually compile {master_tex} with pdflatex")
    except subprocess.TimeoutExpired:
        print(f"  WARNING: pdflatex timed out")
    except Exception as e:
        print(f"  WARNING: PDF generation failed: {e}")


def main():
    """Main analysis function."""
    print("="*70)
    print("noipca2 ANALYSIS - Creating LaTeX Tables")
    print("="*70)
    print()

    # Determine results directory
    results_dir = get_results_dir()
    print(f"Using results from: {results_dir}")
    print()

    for model in MODELS:
        print(f"Processing {model.upper()}...")

        panels = discover_panels(model, results_dir)
        if not panels:
            print(f"  WARNING: No panels found for {model}")
            print()
            continue

        print(f"  Found {len(panels)} panels: {panels}")

        # Load and process data
        print("  Loading Fama data...")
        fama_df = load_and_process_fama(model, panels, results_dir)

        print("  Loading DKKM data...")
        dkkm_df = load_and_process_dkkm(model, panels, results_dir)

        # Create tables
        print("  Creating tables...")

        fama_path = TABLES_DIR / f"{model}_fama.tex"
        create_fama_table(fama_df, model, str(fama_path))

        dkkm_sharpe_path = TABLES_DIR / f"{model}_dkkm_sharpe.tex"
        dkkm_hjd_path = TABLES_DIR / f"{model}_dkkm_hjd.tex"
        create_dkkm_tables(dkkm_df, model, str(dkkm_sharpe_path), str(dkkm_hjd_path))

        # Create figures
        print("  Creating figures...")
        create_fama_boxplots(fama_df, model)
        create_dkkm_boxplots(dkkm_df, model)

        print()

    # Generate PDF
    print("Generating PDF...")
    generate_pdf()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Tables: {TABLES_DIR}/")
    print(f"Figures: {FIGURES_DIR}/")
    print(f"PDF: {TABLES_DIR / 'all_results.pdf'}")


if __name__ == "__main__":
    main()
