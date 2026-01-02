import pandas as pd
import io
import re
import numpy as np
import sys
from datetime import datetime

def create_ultimate_report(input_csv, output_xlsx):
    try:
        with open(input_csv, 'r') as f:
            lines = f.readlines()

        sections = []
        combined_progress_logs = [] 
        
        current_section_name = "UNNAMED_SECTION"
        current_data = []
        current_col_count = -1
        in_named_section = False
        
        # --- HELPER: Detect if a block looks like Progress Logs ---
        def is_progress_block(data_lines):
            if not data_lines: return False
            first_line = data_lines[0]
            # Check for date start (e.g., 2025-...)
            if first_line.startswith("202") or first_line.startswith("203"): 
                return True
            # Fallback: Check column count (logs usually have many columns, e.g. > 7)
            cols = len(first_line.split(','))
            return cols >= 8

        # --- PARSING LOOP ---
        for line in lines:
            raw_line = line.strip()
            # Ignore empty lines
            if not raw_line:
                continue
            
            # 1. Handle SECTION Start
            if raw_line.startswith('SECTION,'):
                # Save previous data if it exists
                if current_data:
                    sections.append((current_section_name, current_data))
                
                current_section_name = raw_line.split(',')[1].strip()
                current_data = []
                current_col_count = -1
                in_named_section = True
                continue
            
            # 2. Handle SECTION End
            if raw_line.startswith('END_SECTION,'):
                # Close current section
                if current_data:
                    sections.append((current_section_name, current_data))
                
                # Switch to "Orphaned" mode (expecting logs or loose data)
                current_section_name = "ORPHANED_BLOCK"
                current_data = []
                current_col_count = -1
                in_named_section = False
                continue
            
            # Ignore comments
            if raw_line.startswith('#'):
                continue

            # 3. Process Data Lines
            cols = raw_line.split(',')
            count = len(cols)
            
            if in_named_section:
                # Inside a named section: just accumulate data
                current_data.append(raw_line)
            else:
                # Outside a named section (Orphaned): Check consistency
                if current_col_count == -1:
                    current_col_count = count
                    current_data.append(raw_line)
                elif count == current_col_count:
                    current_data.append(raw_line)
                else:
                    # Column count changed! This implies a new block of data started.
                    sections.append((current_section_name, current_data))
                    current_section_name = "ORPHANED_BLOCK" # Reset name
                    current_data = [raw_line]
                    current_col_count = count

        # Save the very last block
        if current_data:
            sections.append((current_section_name, current_data))

        # --- MERGING & ORGANIZING ---
        final_sections_map = {} # Map: SectionName -> List of DataChunks
        
        # Header definition for logs (since they usually lack one in the CSV)
        progress_headers = ["timestamp", "elapsed_s", "step_id", "operation", "detail", "val1", "val2", "time_s", "memory_mb"]

        for name, data_lines in sections:
            # Check if this block is a Progress Log
            if (name == "ORPHANED_BLOCK" or name == "UNNAMED_SECTION") and is_progress_block(data_lines):
                combined_progress_logs.extend(data_lines)
            elif name == "ORPHANED_BLOCK":
                # Unknown orphan data? Group into UNKNOWN
                if "UNKNOWN_DATA" not in final_sections_map: final_sections_map["UNKNOWN_DATA"] = []
                final_sections_map["UNKNOWN_DATA"].append(data_lines)
            else:
                # Standard Named Section
                if name not in final_sections_map: final_sections_map[name] = []
                final_sections_map[name].append(data_lines)

        # --- EXCEL WRITING ---
        writer_engine_kwargs = {'options': {'nan_inf_to_errors': True}}
        
        with pd.ExcelWriter(output_xlsx, engine='xlsxwriter', engine_kwargs=writer_engine_kwargs) as writer:
            workbook = writer.book
            
            # --- STYLES ---
            COLOR_NAVY = '#1F4E78'
            COLOR_BORDER = '#D9D9D9'
            COLOR_BANDS = '#F9F9F9'
            COLOR_AVG_BG = '#E2EFDA'  # Light green for average rows
            
            header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': COLOR_NAVY, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_name': 'Segoe UI'})
            data_fmt = workbook.add_format({'border': 1, 'border_color': COLOR_BORDER, 'font_name': 'Segoe UI', 'font_size': 10})
            banded_fmt = workbook.add_format({'border': 1, 'border_color': COLOR_BORDER, 'bg_color': COLOR_BANDS, 'font_name': 'Segoe UI', 'font_size': 10})
            avg_fmt = workbook.add_format({'border': 1, 'bold': True, 'bg_color': COLOR_AVG_BG, 'font_name': 'Segoe UI', 'font_size': 10})
            link_fmt = workbook.add_format({'font_color': 'blue', 'underline': 1, 'font_name': 'Segoe UI'})
            green_bg = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1})
            red_bg = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1})

            sheet_list = []

            # --- WRITER HELPER FUNCTION ---
            def write_sheet(sheet_name, df):
                # Clean sheet name for Excel (max 31 chars, no invalid chars)
                safe_name = re.sub(r'[\[\]\:\*\?\/\\]', '', sheet_name)[:30]
                
                # Handle Duplicate Sheet Names (append _1, _2 if needed)
                original_safe_name = safe_name
                counter = 1
                while safe_name in sheet_list:
                     safe_name = f"{original_safe_name[:27]}_{counter}"
                     counter += 1

                df.to_excel(writer, sheet_name=safe_name, index=False)
                worksheet = writer.sheets[safe_name]
                worksheet.hide_gridlines(2)
                
                (max_row, max_col) = df.shape
                
                # Format Headers
                for c, val in enumerate(df.columns):
                    worksheet.write(0, c, val, header_fmt)
                    worksheet.set_column(c, c, max(len(str(val)), 15))
                
                # Format Data (Banded Rows)
                for r in range(max_row):
                    fmt = banded_fmt if r % 2 == 1 else data_fmt
                    for c in range(max_col):
                        val = df.iloc[r, c]
                        if pd.isna(val):
                            worksheet.write_blank(r+1, c, None, fmt)
                        else:
                            try:
                                worksheet.write_number(r+1, c, float(val), fmt)
                            except:
                                worksheet.write(r+1, c, val, fmt)
                
                # --- SPEEDUP COLORING LOGIC ---
                # 1. Check Column Headers
                for c, name in enumerate(df.columns):
                    if "speedup" in str(name).lower():
                        worksheet.conditional_format(1, c, max_row, c, {'type': 'cell', 'criteria': '>', 'value': 1, 'format': green_bg})
                        worksheet.conditional_format(1, c, max_row, c, {'type': 'cell', 'criteria': '<', 'value': 1, 'format': red_bg})
                
                # 2. Check Row Labels (First Column)
                if max_col > 1:
                    first_col = df.iloc[:, 0].astype(str).tolist()
                    for r, label in enumerate(first_col):
                        if "speedup" in label.lower():
                             worksheet.conditional_format(r+1, 1, r+1, 1, {'type': 'cell', 'criteria': '>', 'value': 1, 'format': green_bg})
                             worksheet.conditional_format(r+1, 1, r+1, 1, {'type': 'cell', 'criteria': '<', 'value': 1, 'format': red_bg})

                sheet_list.append(safe_name)
                return safe_name

            # --- COMPUTE TRIAL AVERAGES FOR SWEEP RESULTS ---
            def compute_trial_averages(df, sweep_param_col):
                """
                Given a DataFrame with trial data, compute averages grouped by the sweep parameter.
                Returns a new DataFrame with one row per unique sweep parameter value.
                """
                if sweep_param_col not in df.columns:
                    return None
                
                # Identify numeric columns to average
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Exclude trial_id and total_trials from averaging (they're identifiers)
                exclude_cols = ['trial_id', 'total_trials']
                avg_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                if not avg_cols:
                    return None
                
                # Group by sweep parameter and compute mean and std
                grouped = df.groupby(sweep_param_col)
                
                result_rows = []
                for param_val, group in grouped:
                    row = {sweep_param_col: param_val}
                    row['num_trials'] = len(group)
                    
                    for col in avg_cols:
                        col_data = pd.to_numeric(group[col], errors='coerce')
                        row[f'{col}_mean'] = col_data.mean()
                        row[f'{col}_std'] = col_data.std()
                    
                    result_rows.append(row)
                
                return pd.DataFrame(result_rows)

            # 1. WRITE NORMAL SECTIONS + AVERAGES
            for name, list_of_chunks in final_sections_map.items():
                dfs = []
                for chunk in list_of_chunks:
                    try:
                        dfs.append(pd.read_csv(io.StringIO("\n".join(chunk))))
                    except:
                        dfs.append(pd.read_csv(io.StringIO("\n".join(chunk)), header=None))
                
                if dfs:
                    full_df = pd.concat(dfs, ignore_index=True)
                    write_sheet(name, full_df)
                    
                    # Check if this is a SWEEP_RESULTS section - compute averages
                    if name.startswith("SWEEP_RESULTS_"):
                        # Determine the sweep parameter from section name
                        sweep_type = name.replace("SWEEP_RESULTS_", "")
                        
                        # Map sweep type to column name
                        param_col_map = {
                            'EPSILON': 'epsilon',
                            'NOISE_PROB': 'noise_prob',
                            'QUBITS': 'n',
                            'DEPTH': 'd',
                            'INITIAL_RANK': 'initial_rank',
                            'CROSSOVER': 'n'
                        }
                        
                        # Try to find the sweep parameter column
                        param_col = param_col_map.get(sweep_type)
                        if param_col is None:
                            # Fallback: use first column
                            param_col = full_df.columns[0]
                        
                        # Check if we have trial data (total_trials > 1 or trial_id column exists)
                        if 'trial_id' in full_df.columns:
                            avg_df = compute_trial_averages(full_df, param_col)
                            if avg_df is not None and len(avg_df) > 0:
                                avg_sheet_name = f"AVG_{sweep_type}"
                                write_sheet(avg_sheet_name, avg_df)

            # 2. MERGE ALL MODE_COMPARISON SECTIONS INTO ONE SHEET
            mode_comparison_dfs = []
            mode_sections = [k for k in final_sections_map.keys() if k.startswith("MODE_COMPARISON_")]
            
            for name in mode_sections:
                for chunk in final_sections_map[name]:
                    try:
                        chunk_df = pd.read_csv(io.StringIO("\n".join(chunk)))
                        # Extract sweep param and value from section name
                        # Format: MODE_COMPARISON_{param}_{value}_trial{trial_id}
                        parts = name.replace("MODE_COMPARISON_", "").split("_trial")
                        if len(parts) >= 1:
                            param_parts = parts[0].rsplit("_", 1)
                            if len(param_parts) == 2:
                                chunk_df['sweep_param'] = param_parts[0]
                                chunk_df['sweep_value'] = param_parts[1]
                        mode_comparison_dfs.append(chunk_df)
                    except:
                        pass
            
            if mode_comparison_dfs:
                all_modes_df = pd.concat(mode_comparison_dfs, ignore_index=True)
                write_sheet("ALL_MODE_COMPARISONS", all_modes_df)
                
                # Compute mode averages per sweep parameter
                if 'sweep_value' in all_modes_df.columns and 'mode' in all_modes_df.columns:
                    # Group by sweep_value and mode, compute averages
                    mode_avg_rows = []
                    for (sv, mode), group in all_modes_df.groupby(['sweep_value', 'mode']):
                        row = {'sweep_value': sv, 'mode': mode, 'num_trials': len(group)}
                        for col in ['time_s', 'final_rank', 'purity', 'entropy', 'speedup_vs_seq', 'fidelity_vs_fdm']:
                            if col in group.columns:
                                col_data = pd.to_numeric(group[col], errors='coerce')
                                row[f'{col}_mean'] = col_data.mean()
                                row[f'{col}_std'] = col_data.std()
                        mode_avg_rows.append(row)
                    
                    if mode_avg_rows:
                        mode_avg_df = pd.DataFrame(mode_avg_rows)
                        write_sheet("AVG_MODE_COMPARISON", mode_avg_df)

            # 3. WRITE COMBINED PROGRESS LOGS (With Normalization Fix)
            if combined_progress_logs:
                # Fix: Scan for max columns to handle varying log formats
                max_cols = 0
                for line in combined_progress_logs:
                    max_cols = max(max_cols, len(line.split(',')))
                
                # Fix: Pad shorter lines with empty commas
                normalized_logs = []
                for line in combined_progress_logs:
                    line = line.strip()
                    curr_cols = len(line.split(','))
                    if curr_cols < max_cols:
                        line += ',' * (max_cols - curr_cols)
                    normalized_logs.append(line)

                df_log = pd.read_csv(io.StringIO("\n".join(normalized_logs)), header=None)
                
                # Assign Headers if count matches, else auto-generate
                if len(df_log.columns) == len(progress_headers):
                    df_log.columns = progress_headers
                else:
                    df_log.columns = [f"Col_{i+1}" for i in range(len(df_log.columns))]
                
                write_sheet("PROGRESS_LOGS", df_log)

            # 4. GENERATE DASHBOARD
            dash = workbook.add_worksheet('DASHBOARD')
            dash.hide_gridlines(2)
            dash.set_column('B:B', 40)
            dash.write('B2', 'QUANTUM REPORT DASHBOARD', workbook.add_format({'bold': True, 'font_size': 18, 'font_color': COLOR_NAVY}))
            
            # Summary Extraction
            row_idx = 4
            if "SUMMARY" in final_sections_map:
                try:
                    summ_lines = final_sections_map["SUMMARY"][0]
                    summ_df = pd.read_csv(io.StringIO("\n".join(summ_lines)))
                    if 'metric' in summ_df.columns and 'value' in summ_df.columns:
                        dash.write('B4', 'Executive Summary', workbook.add_format({'bold': True, 'bottom': 1}))
                        for _, row in summ_df.iterrows():
                            dash.write(row_idx, 1, str(row['metric']), data_fmt)
                            dash.write(row_idx, 2, str(row['value']), workbook.add_format({'bold':True}))
                            row_idx += 1
                        row_idx += 2
                except: pass
            
            # Navigation
            dash.write(row_idx, 1, 'Sheet Navigation:', workbook.add_format({'bold': True, 'bottom': 1}))
            row_idx += 1
            for sname in sheet_list:
                dash.write_url(row_idx, 1, f"internal:'{sname}'!A1", string=f"Go to {sname}", cell_format=link_fmt)
                row_idx += 1
            
            # Add notes about averages
            row_idx += 2
            dash.write(row_idx, 1, 'Notes:', workbook.add_format({'bold': True, 'bottom': 1}))
            row_idx += 1
            dash.write(row_idx, 1, '• AVG_* sheets contain trial averages (mean ± std)', data_fmt)
            row_idx += 1
            dash.write(row_idx, 1, '• fdm_executed=true means FDM ran this trial (only trial 0)', data_fmt)
            row_idx += 1
            dash.write(row_idx, 1, '• fdm_run=true means FDM data available (may be copied)', data_fmt)

        print(f"DONE: Optimized report '{output_xlsx}' created successfully.")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input.csv> <output.xlsx>")
    else:
        create_ultimate_report(sys.argv[1], sys.argv[2])
