#!/usr/bin/env python3
"""
LRET Scientific Benchmarking Report Generator
==============================================

Converts structured CSV output from LRET quantum simulator to a professional
Excel report suitable for scientific publication and analysis.

Based on the LRET (Low-Rank Evolution with Truncation) paper methodology for
benchmarking quantum simulation algorithms.

Output Structure (6-7 sheets max):
1. DASHBOARD    - Executive summary with key findings and navigation
2. CONFIG       - Simulation configuration and parameters  
3. SWEEP_DATA   - All sweep results consolidated (main data table)
4. STATISTICS   - Aggregated statistics with mean/std per sweep parameter
5. MODE_PERF    - Mode performance comparison (all parallelization modes)
6. LOGS         - Execution progress logs (optional, can be hidden)

Author: LRET Benchmarking Suite
Version: 3.0 (Scientific Format)
"""

import pandas as pd
import numpy as np
import io
import re
import sys
from datetime import datetime


def create_scientific_report(input_csv: str, output_xlsx: str) -> bool:
    """
    Generate a professional scientific benchmarking Excel report.
    
    Args:
        input_csv: Path to the structured CSV file from LRET simulator
        output_xlsx: Path for output Excel file
        
    Returns:
        True on success, False on error
    """
    try:
        # =====================================================================
        # PHASE 1: PARSE CSV INTO SECTIONS
        # =====================================================================
        with open(input_csv, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        sections = {}  # name -> list of data lines (excluding SECTION/END_SECTION markers)
        progress_logs = []
        
        current_section = None
        current_data = []
        
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue
                
            # Section start
            if raw.startswith('SECTION,'):
                if current_section and current_data:
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].extend(current_data)
                current_section = raw.split(',', 1)[1].strip()
                current_data = []
                continue
            
            # Section end
            if raw.startswith('END_SECTION,'):
                if current_section and current_data:
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].extend(current_data)
                current_section = None
                current_data = []
                continue
            
            # Data line
            if current_section:
                current_data.append(raw)
            else:
                # Orphaned data - likely progress logs
                # Check if it looks like a timestamp-prefixed log
                if raw[:4].isdigit():
                    progress_logs.append(raw)
        
        # Save any remaining data
        if current_section and current_data:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].extend(current_data)
        
        # =====================================================================
        # PHASE 2: CONVERT SECTIONS TO DATAFRAMES
        # =====================================================================
        
        def lines_to_df(lines):
            """Convert CSV lines to DataFrame."""
            if not lines:
                return pd.DataFrame()
            try:
                return pd.read_csv(io.StringIO('\n'.join(lines)))
            except:
                return pd.read_csv(io.StringIO('\n'.join(lines)), header=None)
        
        # Parse key sections
        config_data = {}
        sweep_dfs = []
        modes_dfs = []
        
        for name, lines in sections.items():
            if name in ['HEADER', 'SWEEP_CONFIG', 'METADATA']:
                df = lines_to_df(lines)
                if 'parameter' in df.columns and 'value' in df.columns:
                    for _, row in df.iterrows():
                        config_data[str(row['parameter'])] = str(row['value'])
                elif 'key' in df.columns and 'value' in df.columns:
                    for _, row in df.iterrows():
                        config_data[str(row['key'])] = str(row['value'])
            
            elif name.startswith('SWEEP_RESULTS_'):
                df = lines_to_df(lines)
                if not df.empty:
                    df['sweep_type'] = name.replace('SWEEP_RESULTS_', '')
                    sweep_dfs.append(df)
            
            elif name.startswith('ALL_MODES_'):
                df = lines_to_df(lines)
                if not df.empty:
                    df['sweep_type'] = name.replace('ALL_MODES_', '')
                    modes_dfs.append(df)
            
            elif name == 'SUMMARY':
                df = lines_to_df(lines)
                if 'metric' in df.columns and 'value' in df.columns:
                    for _, row in df.iterrows():
                        config_data[f"summary_{row['metric']}"] = str(row['value'])
        
        # Merge all sweep data
        all_sweep_data = pd.concat(sweep_dfs, ignore_index=True) if sweep_dfs else pd.DataFrame()
        all_modes_data = pd.concat(modes_dfs, ignore_index=True) if modes_dfs else pd.DataFrame()
        
        # =====================================================================
        # PHASE 3: COMPUTE STATISTICS
        # =====================================================================
        
        def compute_sweep_statistics(df):
            """Compute mean/std/min/max grouped by sweep parameter."""
            if df.empty:
                return pd.DataFrame()
            
            # Identify sweep parameter column
            sweep_cols = ['epsilon', 'noise_prob', 'num_qubits', 'n', 'd', 'depth', 'initial_rank']
            param_col = None
            for col in sweep_cols:
                if col in df.columns:
                    param_col = col
                    break
            
            if param_col is None:
                param_col = df.columns[0]
            
            # Metrics to aggregate
            metric_cols = ['lret_time_s', 'final_rank', 'lret_final_rank', 'purity', 'lret_purity',
                          'entropy', 'lret_entropy', 'fidelity_vs_fdm', 'speedup', 
                          'fdm_time_s', 'gate_time_s', 'noise_time_s', 'truncation_time_s']
            
            available_metrics = [c for c in metric_cols if c in df.columns]
            
            if not available_metrics:
                return pd.DataFrame()
            
            # Group and aggregate
            stats_rows = []
            for param_val, group in df.groupby(param_col):
                row = {
                    'sweep_param': param_col,
                    'param_value': param_val,
                    'n_trials': len(group),
                }
                
                for col in available_metrics:
                    data = pd.to_numeric(group[col], errors='coerce').dropna()
                    if len(data) > 0:
                        row[f'{col}_mean'] = data.mean()
                        row[f'{col}_std'] = data.std() if len(data) > 1 else 0
                        row[f'{col}_min'] = data.min()
                        row[f'{col}_max'] = data.max()
                
                stats_rows.append(row)
            
            return pd.DataFrame(stats_rows)
        
        def compute_mode_statistics(df):
            """Compute mode performance statistics grouped by sweep param and mode."""
            if df.empty or 'mode' not in df.columns:
                return pd.DataFrame()
            
            # Identify sweep parameter column
            sweep_cols = ['epsilon', 'noise_prob', 'num_qubits', 'n', 'd', 'depth', 'initial_rank']
            param_col = None
            for col in sweep_cols:
                if col in df.columns:
                    param_col = col
                    break
            
            if param_col is None:
                return pd.DataFrame()
            
            # Metrics to aggregate
            metric_cols = ['time_s', 'final_rank', 'purity', 'entropy', 
                          'speedup_vs_seq', 'fidelity_vs_fdm']
            available_metrics = [c for c in metric_cols if c in df.columns]
            
            # Group by sweep param and mode
            stats_rows = []
            for (param_val, mode), group in df.groupby([param_col, 'mode']):
                row = {
                    param_col: param_val,
                    'mode': mode,
                    'n_trials': len(group),
                }
                
                for col in available_metrics:
                    data = pd.to_numeric(group[col], errors='coerce').dropna()
                    if len(data) > 0:
                        row[f'{col}_mean'] = data.mean()
                        row[f'{col}_std'] = data.std() if len(data) > 1 else 0
                
                stats_rows.append(row)
            
            return pd.DataFrame(stats_rows)
        
        sweep_stats = compute_sweep_statistics(all_sweep_data)
        mode_stats = compute_mode_statistics(all_modes_data)
        
        # =====================================================================
        # PHASE 4: CREATE EXCEL WORKBOOK
        # =====================================================================
        
        with pd.ExcelWriter(output_xlsx, engine='xlsxwriter', 
                           engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
            workbook = writer.book
            
            # -----------------------------------------------------------------
            # STYLES (Professional Scientific Look)
            # -----------------------------------------------------------------
            NAVY = '#1F4E79'
            LIGHT_BLUE = '#D6DCE4'
            LIGHT_GREEN = '#E2EFDA'
            LIGHT_ORANGE = '#FCE4D6'
            WHITE = '#FFFFFF'
            
            fmt_title = workbook.add_format({
                'bold': True, 'font_size': 20, 'font_color': NAVY,
                'font_name': 'Calibri'
            })
            fmt_subtitle = workbook.add_format({
                'bold': True, 'font_size': 14, 'font_color': NAVY,
                'bottom': 2, 'bottom_color': NAVY, 'font_name': 'Calibri'
            })
            fmt_header = workbook.add_format({
                'bold': True, 'font_color': WHITE, 'bg_color': NAVY,
                'border': 1, 'align': 'center', 'valign': 'vcenter',
                'font_name': 'Calibri', 'font_size': 10, 'text_wrap': True
            })
            fmt_data = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'valign': 'vcenter'
            })
            fmt_data_alt = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'bg_color': '#F2F2F2', 'valign': 'vcenter'
            })
            fmt_number = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'num_format': '0.000000', 'valign': 'vcenter'
            })
            fmt_number_alt = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'num_format': '0.000000', 'bg_color': '#F2F2F2', 'valign': 'vcenter'
            })
            fmt_int = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'num_format': '0', 'valign': 'vcenter'
            })
            fmt_int_alt = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10,
                'num_format': '0', 'bg_color': '#F2F2F2', 'valign': 'vcenter'
            })
            fmt_good = workbook.add_format({
                'border': 1, 'bg_color': '#C6EFCE', 'font_color': '#006100',
                'font_name': 'Calibri', 'font_size': 10
            })
            fmt_bad = workbook.add_format({
                'border': 1, 'bg_color': '#FFC7CE', 'font_color': '#9C0006',
                'font_name': 'Calibri', 'font_size': 10
            })
            fmt_link = workbook.add_format({
                'font_color': 'blue', 'underline': 1, 'font_name': 'Calibri'
            })
            fmt_config_key = workbook.add_format({
                'bold': True, 'font_name': 'Calibri', 'font_size': 10,
                'bg_color': LIGHT_BLUE, 'border': 1
            })
            fmt_config_val = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'border': 1
            })
            
            sheet_names = []
            
            # -----------------------------------------------------------------
            # HELPER: Write DataFrame to Sheet
            # -----------------------------------------------------------------
            def write_data_sheet(name, df, description=""):
                """Write a DataFrame to an Excel sheet with professional formatting."""
                if df.empty:
                    return None
                
                # Sanitize sheet name
                safe_name = re.sub(r'[\[\]:*?/\\]', '', name)[:31]
                
                # Handle duplicates
                base_name = safe_name
                counter = 1
                while safe_name in sheet_names:
                    safe_name = f"{base_name[:28]}_{counter}"
                    counter += 1
                
                df.to_excel(writer, sheet_name=safe_name, index=False, startrow=2)
                ws = writer.sheets[safe_name]
                ws.hide_gridlines(2)
                
                # Title and description
                ws.write(0, 0, name, fmt_subtitle)
                if description:
                    ws.write(1, 0, description, workbook.add_format({
                        'italic': True, 'font_color': '#666666', 'font_name': 'Calibri'
                    }))
                
                nrows, ncols = df.shape
                
                # Format headers
                for c, col_name in enumerate(df.columns):
                    ws.write(2, c, col_name, fmt_header)
                    # Auto-width with min/max constraints
                    max_len = max(len(str(col_name)), 
                                 df[col_name].astype(str).str.len().max() if len(df) > 0 else 0)
                    ws.set_column(c, c, min(max(max_len + 2, 10), 25))
                
                # Format data rows
                for r in range(nrows):
                    is_alt = r % 2 == 1
                    for c in range(ncols):
                        val = df.iloc[r, c]
                        col_name = df.columns[c].lower()
                        
                        if pd.isna(val):
                            ws.write_blank(r + 3, c, None, fmt_data_alt if is_alt else fmt_data)
                        elif isinstance(val, (int, np.integer)):
                            ws.write_number(r + 3, c, int(val), fmt_int_alt if is_alt else fmt_int)
                        elif isinstance(val, (float, np.floating)):
                            ws.write_number(r + 3, c, float(val), fmt_number_alt if is_alt else fmt_number)
                        else:
                            ws.write(r + 3, c, str(val), fmt_data_alt if is_alt else fmt_data)
                
                # Conditional formatting for speedup columns
                for c, col_name in enumerate(df.columns):
                    if 'speedup' in col_name.lower():
                        ws.conditional_format(3, c, nrows + 2, c, {
                            'type': 'cell', 'criteria': '>', 'value': 1, 'format': fmt_good
                        })
                        ws.conditional_format(3, c, nrows + 2, c, {
                            'type': 'cell', 'criteria': '<', 'value': 1, 'format': fmt_bad
                        })
                    elif 'fidelity' in col_name.lower():
                        ws.conditional_format(3, c, nrows + 2, c, {
                            'type': 'cell', 'criteria': '>=', 'value': 0.99, 'format': fmt_good
                        })
                        ws.conditional_format(3, c, nrows + 2, c, {
                            'type': 'cell', 'criteria': '<', 'value': 0.9, 'format': fmt_bad
                        })
                
                # Freeze panes
                ws.freeze_panes(3, 0)
                
                sheet_names.append(safe_name)
                return safe_name
            
            # -----------------------------------------------------------------
            # SHEET 1: DASHBOARD
            # -----------------------------------------------------------------
            dash = workbook.add_worksheet('DASHBOARD')
            sheet_names.append('DASHBOARD')
            dash.hide_gridlines(2)
            dash.set_column('A:A', 3)
            dash.set_column('B:B', 25)
            dash.set_column('C:C', 30)
            dash.set_column('D:D', 40)
            
            # Title
            dash.write('B2', 'LRET Quantum Simulation', fmt_title)
            dash.write('B3', 'Scientific Benchmarking Report', workbook.add_format({
                'font_size': 14, 'font_color': '#666666', 'font_name': 'Calibri'
            }))
            
            row = 5
            
            # Key Metrics Summary
            dash.write(row, 1, 'Key Configuration', fmt_subtitle)
            row += 1
            
            key_params = [
                ('Qubits (n)', config_data.get('num_qubits', config_data.get('base_qubits', 'N/A'))),
                ('Depth (d)', config_data.get('depth', config_data.get('base_depth', 'N/A'))),
                ('Sweep Type', config_data.get('sweep_type', 'N/A')),
                ('Noise Probability', config_data.get('noise_probability', config_data.get('base_noise', 'N/A'))),
                ('Truncation Threshold (ε)', config_data.get('truncation_threshold', 'N/A')),
                ('Parallel Mode', config_data.get('parallel_mode', 'N/A')),
                ('FDM Enabled', config_data.get('fdm_enabled', 'N/A')),
                ('Timestamp', config_data.get('timestamp', config_data.get('generated', 'N/A'))),
            ]
            
            for key, val in key_params:
                dash.write(row, 1, key, fmt_config_key)
                dash.write(row, 2, val, fmt_config_val)
                row += 1
            
            row += 1
            
            # Summary statistics
            if not all_sweep_data.empty:
                dash.write(row, 1, 'Summary Statistics', fmt_subtitle)
                row += 1
                
                n_points = len(all_sweep_data)
                n_trials = all_sweep_data['trial_id'].nunique() if 'trial_id' in all_sweep_data.columns else 1
                
                summary_stats = [
                    ('Total Data Points', str(n_points)),
                    ('Unique Trials', str(n_trials)),
                ]
                
                if 'lret_time_s' in all_sweep_data.columns:
                    avg_time = all_sweep_data['lret_time_s'].mean()
                    summary_stats.append(('Avg LRET Time (s)', f'{avg_time:.4f}'))
                
                if 'fidelity_vs_fdm' in all_sweep_data.columns:
                    avg_fid = all_sweep_data['fidelity_vs_fdm'].mean()
                    summary_stats.append(('Avg Fidelity vs FDM', f'{avg_fid:.6f}'))
                
                if 'speedup' in all_sweep_data.columns:
                    avg_speedup = all_sweep_data['speedup'].mean()
                    summary_stats.append(('Avg Speedup vs FDM', f'{avg_speedup:.2f}x'))
                
                for key, val in summary_stats:
                    dash.write(row, 1, key, fmt_config_key)
                    dash.write(row, 2, val, fmt_config_val)
                    row += 1
            
            row += 2
            
            # Navigation
            dash.write(row, 1, 'Report Navigation', fmt_subtitle)
            row += 1
            
            # -----------------------------------------------------------------
            # SHEET 2: CONFIG (Full Configuration)
            # -----------------------------------------------------------------
            if config_data:
                config_df = pd.DataFrame([
                    {'Parameter': k, 'Value': v} for k, v in config_data.items()
                ])
                write_data_sheet('CONFIG', config_df, 'Full simulation configuration and metadata')
                dash.write_url(row, 1, "internal:'CONFIG'!A1", string='→ Configuration', cell_format=fmt_link)
                row += 1
            
            # -----------------------------------------------------------------
            # SHEET 3: SWEEP_DATA (Main Results)
            # -----------------------------------------------------------------
            if not all_sweep_data.empty:
                # Reorder columns for clarity
                priority_cols = ['sweep_type', 'epsilon', 'noise_prob', 'n', 'd', 'num_qubits', 
                                'depth', 'initial_rank', 'trial_id', 'lret_time_s', 'final_rank',
                                'lret_final_rank', 'fidelity_vs_fdm', 'speedup', 'purity', 'lret_purity',
                                'entropy', 'lret_entropy']
                
                ordered_cols = [c for c in priority_cols if c in all_sweep_data.columns]
                remaining_cols = [c for c in all_sweep_data.columns if c not in ordered_cols]
                all_sweep_data = all_sweep_data[ordered_cols + remaining_cols]
                
                write_data_sheet('SWEEP_DATA', all_sweep_data, 
                               'Complete sweep results - one row per trial per parameter value')
                dash.write_url(row, 1, "internal:'SWEEP_DATA'!A1", string='→ Sweep Data (Raw)', cell_format=fmt_link)
                row += 1
            
            # -----------------------------------------------------------------
            # SHEET 4: STATISTICS (Aggregated)
            # -----------------------------------------------------------------
            if not sweep_stats.empty:
                write_data_sheet('STATISTICS', sweep_stats,
                               'Aggregated statistics: mean, std, min, max per sweep parameter')
                dash.write_url(row, 1, "internal:'STATISTICS'!A1", string='→ Statistics (Aggregated)', cell_format=fmt_link)
                row += 1
            
            # -----------------------------------------------------------------
            # SHEET 5: MODE_PERF (Mode Performance)
            # -----------------------------------------------------------------
            if not all_modes_data.empty:
                write_data_sheet('MODE_DATA', all_modes_data,
                               'Parallelization mode performance - all modes, all trials')
                dash.write_url(row, 1, "internal:'MODE_DATA'!A1", string='→ Mode Performance (Raw)', cell_format=fmt_link)
                row += 1
            
            if not mode_stats.empty:
                write_data_sheet('MODE_STATS', mode_stats,
                               'Mode performance statistics: mean, std per sweep parameter and mode')
                dash.write_url(row, 1, "internal:'MODE_STATS'!A1", string='→ Mode Statistics', cell_format=fmt_link)
                row += 1
            
            # -----------------------------------------------------------------
            # SHEET 6: LOGS (Progress Logs)
            # -----------------------------------------------------------------
            if progress_logs:
                # Normalize log lines
                max_cols = max(len(line.split(',')) for line in progress_logs)
                normalized = []
                for line in progress_logs:
                    parts = line.split(',')
                    while len(parts) < max_cols:
                        parts.append('')
                    normalized.append(','.join(parts))
                
                log_df = pd.read_csv(io.StringIO('\n'.join(normalized)), header=None)
                # Try to assign meaningful headers
                log_headers = ['timestamp', 'elapsed_s', 'step', 'operation', 'detail', 
                              'time_s', 'memory_mb', 'cumulative_s', 'extra']
                if len(log_df.columns) <= len(log_headers):
                    log_df.columns = log_headers[:len(log_df.columns)]
                else:
                    log_df.columns = [f'col_{i}' for i in range(len(log_df.columns))]
                
                write_data_sheet('LOGS', log_df, 'Execution progress logs')
                dash.write_url(row, 1, "internal:'LOGS'!A1", string='→ Progress Logs', cell_format=fmt_link)
                row += 1
            
            # -----------------------------------------------------------------
            # FINALIZE DASHBOARD
            # -----------------------------------------------------------------
            row += 2
            dash.write(row, 1, 'Notes', fmt_subtitle)
            row += 1
            notes = [
                '• fdm_executed = true: FDM actually ran this trial (typically trial 0 only)',
                '• fdm_run = true: FDM data available (may be copied from trial 0)',
                '• Speedup > 1 (green): LRET is faster than FDM',
                '• Fidelity ≥ 0.99 (green): High accuracy vs reference FDM simulation',
                '• Statistics sheets contain mean ± std for multi-trial experiments',
            ]
            for note in notes:
                dash.write(row, 1, note, workbook.add_format({
                    'font_name': 'Calibri', 'font_size': 10, 'font_color': '#666666'
                }))
                row += 1
            
            row += 2
            dash.write(row, 1, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                      workbook.add_format({'italic': True, 'font_color': '#999999', 'font_name': 'Calibri'}))
        
        print(f"SUCCESS: Scientific report saved to '{output_xlsx}'")
        print(f"         Total sheets: {len(sheet_names)}")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command-line entry point."""
    if len(sys.argv) < 3:
        print("LRET Scientific Benchmarking Report Generator")
        print("=" * 50)
        print()
        print("Usage: python ultimato4.py <input.csv> <output.xlsx>")
        print()
        print("Converts LRET structured CSV output to a professional")
        print("Excel report suitable for scientific publication.")
        print()
        print("Output includes:")
        print("  • Dashboard with key findings")
        print("  • Consolidated sweep data")
        print("  • Aggregated statistics (mean/std/min/max)")
        print("  • Mode performance comparison")
        print("  • Execution logs")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_xlsx = sys.argv[2]
    
    success = create_scientific_report(input_csv, output_xlsx)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
