#!/usr/bin/env python3
"""
Compare performance of multiple quantum circuits
Usage: python3 compare_circuits.py <circuit1.json> <circuit2.json> ... [--output results.csv]
"""
import json
import subprocess
import time
import sys
import os
from pathlib import Path

def run_circuit(circuit_file, simulator_path="../build/quantum_sim"):
    """Run a circuit and collect timing information"""
    if not Path(simulator_path).exists():
        raise FileNotFoundError(f"Simulator not found: {simulator_path}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [simulator_path, circuit_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        elapsed_time = time.time() - start_time
        
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        
        # Parse circuit details
        with open(circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        num_qubits = circuit_data['circuit']['num_qubits']
        num_gates = len(circuit_data['circuit']['operations'])
        
        # Parse final fidelity from output
        fidelity = None
        for line in output.split('\n'):
            if 'fidelity' in line.lower():
                try:
                    fidelity = float(line.split(':')[-1].strip())
                except:
                    pass
        
        return {
            'circuit': os.path.basename(circuit_file),
            'num_qubits': num_qubits,
            'num_gates': num_gates,
            'time_seconds': elapsed_time,
            'success': success,
            'fidelity': fidelity
        }
    except subprocess.TimeoutExpired:
        return {
            'circuit': os.path.basename(circuit_file),
            'success': False,
            'error': 'Timeout (>300s)'
        }
    except Exception as e:
        return {
            'circuit': os.path.basename(circuit_file),
            'success': False,
            'error': str(e)
        }

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 compare_circuits.py <circuit1.json> <circuit2.json> ... [--output results.csv]")
        print("\nExample:")
        print("  python3 compare_circuits.py circuits/*.json --output comparison.csv")
        sys.exit(0 if len(sys.argv) < 2 else 1)
    
    # Parse arguments
    circuits = []
    output_file = "circuit_comparison.csv"
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--output':
            if i + 2 < len(sys.argv):
                output_file = sys.argv[i + 2]
        elif not arg.endswith('.csv'):
            circuits.append(arg)
    
    if not circuits:
        print("❌ No circuit files specified")
        sys.exit(1)
    
    print(f"🚀 Comparing {len(circuits)} circuits...")
    print(f"   Output: {output_file}\n")
    
    results = []
    for i, circuit in enumerate(circuits, 1):
        print(f"⏳ [{i}/{len(circuits)}] Running {os.path.basename(circuit)}...", end=' ')
        result = run_circuit(circuit)
        results.append(result)
        
        if result['success']:
            print(f"✅ {result['time_seconds']:.2f}s")
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ {error}")
    
    # Write CSV
    with open(output_file, 'w') as f:
        # Header
        f.write("Circuit,Qubits,Gates,Time(s),Success,Fidelity\n")
        
        # Data
        for r in results:
            circuit = r.get('circuit', 'unknown')
            qubits = r.get('num_qubits', 'N/A')
            gates = r.get('num_gates', 'N/A')
            time_s = f"{r.get('time_seconds', 0):.3f}" if r.get('success') else 'N/A'
            success = 'Yes' if r.get('success') else 'No'
            fidelity = f"{r.get('fidelity', 0):.6f}" if r.get('fidelity') else 'N/A'
            
            f.write(f"{circuit},{qubits},{gates},{time_s},{success},{fidelity}\n")
    
    print(f"\n✅ Comparison complete: {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r.get('success'))
    print(f"\n📊 Summary:")
    print(f"   Total: {len(results)}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_time = sum(r['time_seconds'] for r in results if r.get('success')) / successful
        print(f"   ⏱️  Average time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()
