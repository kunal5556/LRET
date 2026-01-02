#!/usr/bin/env python3
"""
LRET Scientific Benchmarking Report Generator
==============================================

Converts structured CSV output from LRET quantum simulator to a professional
Excel report suitable for scientific publication and analysis.

Version: 4.0 - Reader-friendly with sweep-separated tabs and comprehensive guide
"""

import pandas as pd
import numpy as np
import io
import re
import sys
from datetime import datetime
from collections import OrderedDict


def create_scientific_report(input_csv: str, output_xlsx: str) -> bool:
    """
    Generate a professional scientific benchmarking Excel report.
    """
    try:
        # =====================================================================
        # PHASE 1: PARSE CSV INTO SECTIONS
        # =====================================================================
        with open(input_csv, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        sections = {}
        progress_logs = []
        
        current_section = None
        current_data = []
        
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue
                
            if raw.startswith('SECTION,'):
                if current_section and current_data:
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].extend(current_data)
                current_section = raw.split(',', 1)[1].strip()
                current_data = []
                continue
            
            if raw.startswith('END_SECTION,'):
                if current_section and current_data:
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].extend(current_data)
                current_section = None
                current_data = []
                continue
            
            if current_section:
                current_data.append(raw)
            else:
                if raw[:4].isdigit():
                    progress_logs.append(raw)
        
        if current_section and current_data:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].extend(current_data)
        
        # =====================================================================
        # PHASE 2: CONVERT SECTIONS TO DATAFRAMES
        # =====================================================================
        
        def lines_to_df(lines):
            if not lines:
                return pd.DataFrame()
            try:
                return pd.read_csv(io.StringIO('\n'.join(lines)))
            except:
                return pd.read_csv(io.StringIO('\n'.join(lines)), header=None)
        
        # Parse sections into organized data
        config_data = OrderedDict()
        sweep_data_by_type = {}  # sweep_type -> DataFrame
        modes_data_by_type = {}  # sweep_type -> DataFrame
        
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
                sweep_type = name.replace('SWEEP_RESULTS_', '')
                df = lines_to_df(lines)
                if not df.empty:
                    sweep_data_by_type[sweep_type] = df
            
            elif name.startswith('ALL_MODES_'):
                sweep_type = name.replace('ALL_MODES_', '')
                df = lines_to_df(lines)
                if not df.empty:
                    modes_data_by_type[sweep_type] = df
            
            elif name == 'SUMMARY':
                df = lines_to_df(lines)
                if 'metric' in df.columns and 'value' in df.columns:
                    for _, row in df.iterrows():
                        config_data[f"summary_{row['metric']}"] = str(row['value'])
        
        # =====================================================================
        # PHASE 3: COMPUTE STATISTICS PER SWEEP TYPE
        # =====================================================================
        
        def get_sweep_param_col(df, sweep_type):
            """Determine the sweep parameter column based on sweep type."""
            type_to_col = {
                'EPSILON': 'epsilon',
                'NOISE_PROB': 'noise_prob',
                'QUBITS': 'n',
                'DEPTH': 'd',
                'INITIAL_RANK': 'initial_rank',
                'CROSSOVER': 'n',
            }
            
            # First try the mapped column
            col = type_to_col.get(sweep_type.upper())
            if col and col in df.columns:
                return col
            
            # Fallback: check common columns
            for c in ['epsilon', 'noise_prob', 'n', 'd', 'num_qubits', 'depth', 'initial_rank']:
                if c in df.columns:
                    return c
            
            return df.columns[0] if len(df.columns) > 0 else None
        
        def compute_statistics_for_sweep(df, sweep_type):
            """Compute statistics for a single sweep type."""
            if df.empty:
                return pd.DataFrame()
            
            param_col = get_sweep_param_col(df, sweep_type)
            if param_col is None:
                return pd.DataFrame()
            
            # Metrics to aggregate
            metric_cols = ['lret_time_s', 'final_rank', 'lret_final_rank', 'purity', 'lret_purity',
                          'entropy', 'lret_entropy', 'fidelity_vs_fdm', 'speedup', 
                          'fdm_time_s', 'gate_time_s', 'noise_time_s', 'truncation_time_s',
                          'lret_memory_bytes', 'fdm_memory_bytes', 'truncation_count']
            
            available_metrics = [c for c in metric_cols if c in df.columns]
            
            if not available_metrics:
                return pd.DataFrame()
            
            stats_rows = []
            for param_val, group in df.groupby(param_col):
                row = OrderedDict()
                row[param_col] = param_val
                row['n_trials'] = len(group)
                
                # Add qubit/depth info if available and not the sweep param
                if 'n' in df.columns and param_col != 'n':
                    row['n'] = group['n'].iloc[0] if len(group) > 0 else ''
                if 'd' in df.columns and param_col != 'd':
                    row['d'] = group['d'].iloc[0] if len(group) > 0 else ''
                
                for col in available_metrics:
                    data = pd.to_numeric(group[col], errors='coerce').dropna()
                    if len(data) > 0:
                        row[f'{col}_mean'] = data.mean()
                        row[f'{col}_std'] = data.std() if len(data) > 1 else 0.0
                
                stats_rows.append(row)
            
            return pd.DataFrame(stats_rows)
        
        def compute_mode_statistics_for_sweep(df, sweep_type):
            """Compute mode statistics for a single sweep type."""
            if df.empty or 'mode' not in df.columns:
                return pd.DataFrame()
            
            param_col = get_sweep_param_col(df, sweep_type)
            if param_col is None:
                return pd.DataFrame()
            
            metric_cols = ['time_s', 'final_rank', 'purity', 'entropy', 
                          'speedup_vs_seq', 'fidelity_vs_fdm', 'trace_distance_vs_fdm']
            available_metrics = [c for c in metric_cols if c in df.columns]
            
            stats_rows = []
            for (param_val, mode), group in df.groupby([param_col, 'mode']):
                row = OrderedDict()
                row[param_col] = param_val
                row['mode'] = mode
                row['n_trials'] = len(group)
                
                for col in available_metrics:
                    data = pd.to_numeric(group[col], errors='coerce').dropna()
                    if len(data) > 0:
                        row[f'{col}_mean'] = data.mean()
                        row[f'{col}_std'] = data.std() if len(data) > 1 else 0.0
                
                stats_rows.append(row)
            
            return pd.DataFrame(stats_rows)
        
        # Compute statistics per sweep type
        stats_by_type = {}
        mode_stats_by_type = {}
        
        for sweep_type, df in sweep_data_by_type.items():
            stats_by_type[sweep_type] = compute_statistics_for_sweep(df, sweep_type)
        
        for sweep_type, df in modes_data_by_type.items():
            mode_stats_by_type[sweep_type] = compute_mode_statistics_for_sweep(df, sweep_type)
        
        # =====================================================================
        # PHASE 4: CREATE EXCEL WORKBOOK
        # =====================================================================
        
        with pd.ExcelWriter(output_xlsx, engine='xlsxwriter', 
                           engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
            workbook = writer.book
            
            # -----------------------------------------------------------------
            # STYLES
            # -----------------------------------------------------------------
            NAVY = '#1F4E79'
            LIGHT_BLUE = '#D6DCE4'
            LIGHT_GREEN = '#E2EFDA'
            WHITE = '#FFFFFF'
            
            fmt_title = workbook.add_format({
                'bold': True, 'font_size': 22, 'font_color': NAVY, 'font_name': 'Calibri'
            })
            fmt_subtitle = workbook.add_format({
                'bold': True, 'font_size': 14, 'font_color': NAVY,
                'bottom': 2, 'bottom_color': NAVY, 'font_name': 'Calibri'
            })
            fmt_section = workbook.add_format({
                'bold': True, 'font_size': 12, 'font_color': NAVY, 'font_name': 'Calibri'
            })
            fmt_header = workbook.add_format({
                'bold': True, 'font_color': WHITE, 'bg_color': NAVY,
                'border': 1, 'align': 'center', 'valign': 'vcenter',
                'font_name': 'Calibri', 'font_size': 10, 'text_wrap': True
            })
            fmt_data = workbook.add_format({
                'border': 1, 'font_name': 'Calibri', 'font_size': 10, 'valign': 'vcenter'
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
                'font_color': 'blue', 'underline': 1, 'font_name': 'Calibri', 'font_size': 11
            })
            fmt_key = workbook.add_format({
                'bold': True, 'font_name': 'Calibri', 'font_size': 10,
                'bg_color': LIGHT_BLUE, 'border': 1
            })
            fmt_val = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'border': 1
            })
            fmt_guide = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'text_wrap': True, 'valign': 'top'
            })
            fmt_guide_header = workbook.add_format({
                'bold': True, 'font_name': 'Calibri', 'font_size': 11, 
                'font_color': NAVY, 'valign': 'top'
            })
            fmt_note = workbook.add_format({
                'font_name': 'Calibri', 'font_size': 10, 'font_color': '#666666',
                'text_wrap': True
            })
            
            sheet_info = []  # List of (sheet_name, description) for navigation
            
            # -----------------------------------------------------------------
            # HELPER: Write DataFrame to Sheet
            # -----------------------------------------------------------------
            def write_data_sheet(name, df, description="", start_row=2):
                if df.empty:
                    return None
                
                safe_name = re.sub(r'[\[\]:*?/\\]', '', name)[:31]
                existing = [s[0] for s in sheet_info]
                base_name = safe_name
                counter = 1
                while safe_name in existing:
                    safe_name = f"{base_name[:28]}_{counter}"
                    counter += 1
                
                df.to_excel(writer, sheet_name=safe_name, index=False, startrow=start_row)
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
                    ws.write(start_row, c, col_name, fmt_header)
                    max_len = max(len(str(col_name)), 
                                 df[col_name].astype(str).str.len().max() if len(df) > 0 else 0)
                    ws.set_column(c, c, min(max(max_len + 2, 10), 22))
                
                # Format data rows
                for r in range(nrows):
                    is_alt = r % 2 == 1
                    for c in range(ncols):
                        val = df.iloc[r, c]
                        
                        if pd.isna(val):
                            ws.write_blank(r + start_row + 1, c, None, fmt_data_alt if is_alt else fmt_data)
                        elif isinstance(val, (int, np.integer)):
                            ws.write_number(r + start_row + 1, c, int(val), fmt_int_alt if is_alt else fmt_int)
                        elif isinstance(val, (float, np.floating)):
                            ws.write_number(r + start_row + 1, c, float(val), fmt_number_alt if is_alt else fmt_number)
                        else:
                            ws.write(r + start_row + 1, c, str(val), fmt_data_alt if is_alt else fmt_data)
                
                # Conditional formatting
                for c, col_name in enumerate(df.columns):
                    if 'speedup' in col_name.lower():
                        ws.conditional_format(start_row + 1, c, nrows + start_row, c, {
                            'type': 'cell', 'criteria': '>', 'value': 1, 'format': fmt_good
                        })
                        ws.conditional_format(start_row + 1, c, nrows + start_row, c, {
                            'type': 'cell', 'criteria': '<', 'value': 1, 'format': fmt_bad
                        })
                    elif 'fidelity' in col_name.lower():
                        ws.conditional_format(start_row + 1, c, nrows + start_row, c, {
                            'type': 'cell', 'criteria': '>=', 'value': 0.99, 'format': fmt_good
                        })
                        ws.conditional_format(start_row + 1, c, nrows + start_row, c, {
                            'type': 'cell', 'criteria': '<', 'value': 0.9, 'format': fmt_bad
                        })
                
                ws.freeze_panes(start_row + 1, 0)
                
                sheet_info.append((safe_name, description))
                return safe_name
            
            # -----------------------------------------------------------------
            # SHEET 1: DASHBOARD (Created last, placeholder now)
            # -----------------------------------------------------------------
            dash = workbook.add_worksheet('DASHBOARD')
            sheet_info.append(('DASHBOARD', 'Executive summary, configuration, and user guide'))
            
            # -----------------------------------------------------------------
            # SHEET 2: CONFIG
            # -----------------------------------------------------------------
            if config_data:
                config_df = pd.DataFrame([
                    {'Parameter': k, 'Value': v} for k, v in config_data.items()
                ])
                write_data_sheet('CONFIG', config_df, 'Full simulation configuration and metadata')
            
            # -----------------------------------------------------------------
            # SHEETS: SWEEP DATA (One per sweep type)
            # -----------------------------------------------------------------
            for sweep_type, df in sweep_data_by_type.items():
                # Select most important columns for readability
                param_col = get_sweep_param_col(df, sweep_type)
                
                priority_cols = [param_col, 'trial_id', 'n', 'd', 'epsilon', 'noise_prob',
                                'lret_time_s', 'lret_final_rank', 'lret_purity', 'lret_entropy',
                                'fidelity_vs_fdm', 'speedup', 'fdm_time_s', 'fdm_executed']
                
                ordered_cols = [c for c in priority_cols if c in df.columns]
                remaining = [c for c in df.columns if c not in ordered_cols]
                display_df = df[ordered_cols + remaining].copy()
                
                sheet_name = f"DATA_{sweep_type[:20]}"
                desc = f"Raw sweep data for {sweep_type} - one row per trial"
                write_data_sheet(sheet_name, display_df, desc)
            
            # -----------------------------------------------------------------
            # SHEETS: STATISTICS (One per sweep type)
            # -----------------------------------------------------------------
            for sweep_type, df in stats_by_type.items():
                if not df.empty:
                    sheet_name = f"STATS_{sweep_type[:19]}"
                    desc = f"Aggregated statistics for {sweep_type} sweep (mean ± std)"
                    write_data_sheet(sheet_name, df, desc)
            
            # -----------------------------------------------------------------
            # SHEETS: MODE DATA (One per sweep type, if exists)
            # -----------------------------------------------------------------
            for sweep_type, df in modes_data_by_type.items():
                if not df.empty:
                    sheet_name = f"MODES_{sweep_type[:19]}"
                    desc = f"Parallelization mode performance for {sweep_type}"
                    write_data_sheet(sheet_name, df, desc)
            
            # -----------------------------------------------------------------
            # SHEETS: MODE STATS (One per sweep type, if exists)
            # -----------------------------------------------------------------
            for sweep_type, df in mode_stats_by_type.items():
                if not df.empty:
                    sheet_name = f"MSTAT_{sweep_type[:19]}"
                    desc = f"Mode statistics for {sweep_type} (mean ± std per mode)"
                    write_data_sheet(sheet_name, df, desc)
            
            # -----------------------------------------------------------------
            # SHEET: LOGS
            # -----------------------------------------------------------------
            if progress_logs:
                max_cols = max(len(line.split(',')) for line in progress_logs)
                normalized = []
                for line in progress_logs:
                    parts = line.split(',')
                    while len(parts) < max_cols:
                        parts.append('')
                    normalized.append(','.join(parts))
                
                log_df = pd.read_csv(io.StringIO('\n'.join(normalized)), header=None)
                log_headers = ['timestamp', 'elapsed_s', 'step', 'operation', 'detail', 
                              'time_s', 'memory_mb', 'cumulative_s', 'extra']
                if len(log_df.columns) <= len(log_headers):
                    log_df.columns = log_headers[:len(log_df.columns)]
                else:
                    log_df.columns = [f'col_{i}' for i in range(len(log_df.columns))]
                
                write_data_sheet('LOGS', log_df, 'Execution progress and timing logs')
            
            # -----------------------------------------------------------------
            # FINALIZE DASHBOARD
            # -----------------------------------------------------------------
            dash.hide_gridlines(2)
            dash.set_column('A:A', 3)
            dash.set_column('B:B', 28)
            dash.set_column('C:C', 35)
            dash.set_column('D:D', 60)
            
            row = 1
            
            # Title
            dash.write(row, 1, 'LRET Quantum Simulation Report', fmt_title)
            row += 1
            dash.write(row, 1, 'Scientific Benchmarking Analysis', workbook.add_format({
                'font_size': 14, 'font_color': '#666666', 'font_name': 'Calibri', 'italic': True
            }))
            row += 2
            
            # ----- SECTION 1: SIMULATION CONFIGURATION -----
            dash.write(row, 1, 'Simulation Configuration', fmt_subtitle)
            row += 1
            
            # Gather comprehensive config info
            sweep_types_found = list(sweep_data_by_type.keys())
            all_qubits = set()
            all_depths = set()
            all_trials = set()
            total_points = 0
            
            for df in sweep_data_by_type.values():
                total_points += len(df)
                if 'n' in df.columns:
                    all_qubits.update(df['n'].unique())
                if 'd' in df.columns:
                    all_depths.update(df['d'].unique())
                if 'trial_id' in df.columns:
                    all_trials.update(df['trial_id'].unique())
            
            # Build config display
            config_display = [
                ('Sweep Type(s)', ', '.join(sweep_types_found) if sweep_types_found else 'N/A'),
                ('Qubits (n)', ', '.join(map(str, sorted(all_qubits))) if all_qubits else config_data.get('num_qubits', 'N/A')),
                ('Depth (d)', ', '.join(map(str, sorted(all_depths))) if all_depths else config_data.get('depth', 'N/A')),
                ('Number of Trials', str(len(all_trials)) if all_trials else '1'),
                ('Total Data Points', str(total_points)),
                ('Noise Probability', config_data.get('noise_probability', config_data.get('base_noise', 'See data sheets'))),
                ('Truncation Threshold (ε)', config_data.get('truncation_threshold', 'Varies by sweep')),
                ('FDM Reference', 'Enabled (trial 0 only)' if any('fdm_executed' in df.columns for df in sweep_data_by_type.values()) else 'Check data'),
                ('Timestamp', config_data.get('timestamp', config_data.get('generated', datetime.now().strftime('%Y-%m-%d %H:%M')))),
            ]
            
            for key, val in config_display:
                dash.write(row, 1, key, fmt_key)
                dash.write(row, 2, val, fmt_val)
                row += 1
            
            row += 1
            
            # ----- SECTION 2: KEY RESULTS SUMMARY -----
            dash.write(row, 1, 'Key Results Summary', fmt_subtitle)
            row += 1
            
            # Compute summary across all sweep data
            all_lret_times = []
            all_fidelities = []
            all_speedups = []
            
            for df in sweep_data_by_type.values():
                if 'lret_time_s' in df.columns:
                    all_lret_times.extend(df['lret_time_s'].dropna().tolist())
                if 'fidelity_vs_fdm' in df.columns:
                    all_fidelities.extend(df['fidelity_vs_fdm'].dropna().tolist())
                if 'speedup' in df.columns:
                    all_speedups.extend(df['speedup'].dropna().tolist())
            
            results_display = []
            if all_lret_times:
                results_display.append(('Avg LRET Time', f'{np.mean(all_lret_times):.6f} s'))
                results_display.append(('LRET Time Range', f'{np.min(all_lret_times):.6f} - {np.max(all_lret_times):.6f} s'))
            if all_fidelities:
                results_display.append(('Avg Fidelity vs FDM', f'{np.mean(all_fidelities):.6f}'))
                results_display.append(('Min Fidelity', f'{np.min(all_fidelities):.6f}'))
            if all_speedups:
                results_display.append(('Avg Speedup vs FDM', f'{np.mean(all_speedups):.2f}x'))
                results_display.append(('Max Speedup', f'{np.max(all_speedups):.2f}x'))
            
            if not results_display:
                results_display.append(('Status', 'See data sheets for detailed results'))
            
            for key, val in results_display:
                dash.write(row, 1, key, fmt_key)
                dash.write(row, 2, val, fmt_val)
                row += 1
            
            row += 1
            
            # ----- SECTION 3: SHEET NAVIGATION -----
            dash.write(row, 1, 'Sheet Navigation', fmt_subtitle)
            row += 1
            
            for sheet_name, desc in sheet_info:
                if sheet_name != 'DASHBOARD':
                    dash.write_url(row, 1, f"internal:'{sheet_name}'!A1", 
                                  string=f"→ {sheet_name}", cell_format=fmt_link)
                    dash.write(row, 2, desc, fmt_note)
                    row += 1
            
            row += 1
            
            # ----- SECTION 4: USER GUIDE -----
            dash.write(row, 1, 'How to Use This Report', fmt_subtitle)
            row += 1
            
            guide_sections = [
                ("Understanding Sheet Names", 
                 "• DATA_* sheets contain raw trial data for each sweep type\n"
                 "• STATS_* sheets contain aggregated statistics (mean, std) per sweep parameter\n"
                 "• MODES_* sheets show performance of different parallelization modes\n"
                 "• MSTAT_* sheets contain mode performance statistics\n"
                 "• CONFIG contains full simulation parameters\n"
                 "• LOGS contains execution timing details"),
                
                ("Reading the Data",
                 "• Each DATA sheet has one row per trial per parameter value\n"
                 "• 'trial_id' identifies which trial (0, 1, 2, ...) the row belongs to\n"
                 "• 'n' = number of qubits, 'd' = circuit depth\n"
                 "• 'lret_time_s' = LRET simulation time in seconds\n"
                 "• 'fidelity_vs_fdm' = accuracy compared to full density matrix (1.0 = perfect)\n"
                 "• 'speedup' = how much faster LRET is vs FDM (>1 means LRET is faster)"),
                
                ("Understanding FDM Columns",
                 "• 'fdm_executed' = true means FDM actually ran (typically trial 0 only)\n"
                 "• 'fdm_run' = true means FDM data is available (may be copied from trial 0)\n"
                 "• FDM runs once per sweep point; other trials reuse the same FDM reference"),
                
                ("Comparing Results",
                 "• Use STATS sheets for quick comparison - they show mean ± std\n"
                 "• Green highlighting: speedup > 1 (LRET faster) or fidelity ≥ 0.99 (high accuracy)\n"
                 "• Red highlighting: speedup < 1 (LRET slower) or fidelity < 0.9 (lower accuracy)\n"
                 "• Compare across sweep parameters to see scaling behavior"),
                
                ("Mode Performance Analysis",
                 "• MODES sheets compare parallelization strategies (sequential, omp, etc.)\n"
                 "• 'speedup_vs_seq' shows speedup relative to sequential mode\n"
                 "• Use MSTAT sheets to see which mode performs best on average"),
                
                ("Statistical Analysis",
                 "• '_mean' columns show average across trials\n"
                 "• '_std' columns show standard deviation (variability)\n"
                 "• Low std relative to mean indicates consistent performance\n"
                 "• Use trial data for detailed statistical tests"),
            ]
            
            for title, content in guide_sections:
                dash.write(row, 1, title, fmt_guide_header)
                row += 1
                # Write multi-line content
                for line in content.split('\n'):
                    dash.write(row, 1, line, fmt_guide)
                    row += 1
                row += 1
            
            # ----- SECTION 5: LEGEND -----
            dash.write(row, 1, 'Color Legend', fmt_subtitle)
            row += 1
            dash.write(row, 1, '■ Green', fmt_good)
            dash.write(row, 2, 'Good: Speedup > 1 or Fidelity ≥ 0.99', fmt_note)
            row += 1
            dash.write(row, 1, '■ Red', fmt_bad)
            dash.write(row, 2, 'Warning: Speedup < 1 or Fidelity < 0.9', fmt_note)
            row += 2
            
            # Footer
            dash.write(row, 1, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                      workbook.add_format({'italic': True, 'font_color': '#999999', 'font_name': 'Calibri'}))
            dash.write(row + 1, 1, 'LRET Scientific Benchmarking Suite v4.0',
                      workbook.add_format({'italic': True, 'font_color': '#999999', 'font_name': 'Calibri'}))
        
        print(f"SUCCESS: Scientific report saved to '{output_xlsx}'")
        print(f"         Sheets created: {len(sheet_info)}")
        for name, desc in sheet_info:
            print(f"           - {name}: {desc[:50]}...")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 3:
        print("LRET Scientific Benchmarking Report Generator v4.0")
        print("=" * 55)
        print()
        print("Usage: python ultimato4.py <input.csv> <output.xlsx>")
        print()
        print("Features:")
        print("  • Separate sheets per sweep type for readability")
        print("  • Aggregated statistics with mean/std")
        print("  • Comprehensive dashboard with user guide")
        print("  • Color-coded results (green=good, red=warning)")
        print("  • Mode performance comparison")
        sys.exit(1)
    
    create_scientific_report(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
