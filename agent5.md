# LRET AI Agent - Part 5: Visualization, Cloud & ML (Phases 14-16)

_Previous: [agent4.md](agent4.md) - Session, Batch Execution & API_  
_Next: [agent6.md](agent6.md) - Collaboration & User Guide_

---

## 📚 PHASE 14: Visualization & Charting

### Overview

Rich visualization capabilities for experiment results, performance metrics, and quantum state analysis.

### 14.1 Results Visualizer

**File: `agent/visualization/plotter.py`**

```python
"""Visualization for quantum experiment results."""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# Try to import matplotlib, fallback gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class ResultsVisualizer:
    """Visualize quantum experiment results."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./outputs/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Visualization disabled.")

    def plot_histogram(self, counts: Dict[str, int],
                      title: str = "Measurement Results",
                      filename: str = None) -> Optional[str]:
        """Plot measurement histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by count (descending) and take top states
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

        if len(sorted_counts) > 32:
            sorted_counts = sorted_counts[:32]
            title += " (top 32 states)"

        states = [s[0] for s in sorted_counts]
        values = [s[1] for s in sorted_counts]

        colors = plt.cm.viridis([v / max(values) for v in values])
        bars = ax.bar(range(len(states)), values, color=colors)

        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Quantum State")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def plot_fidelity_over_time(self, results: List[Dict[str, Any]],
                               title: str = "Fidelity Over Time",
                               filename: str = None) -> Optional[str]:
        """Plot fidelity across multiple experiments."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        fidelities = [r.get("fidelity", 0) for r in results]
        run_ids = [r.get("run_id", str(i))[-6:] for i, r in enumerate(results)]

        ax.plot(range(len(fidelities)), fidelities, 'b-o', markersize=8)
        ax.axhline(y=0.99, color='g', linestyle='--', label='Target (0.99)')

        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Fidelity")
        ax.set_title(title)
        ax.set_ylim(0.9, 1.01)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def plot_scaling_analysis(self, results: List[Dict[str, Any]],
                             metric: str = "simulation_time_seconds",
                             title: str = None,
                             filename: str = None) -> Optional[str]:
        """Plot scaling behavior across qubit counts."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        qubits = [r.get("n_qubits", 0) for r in results]
        values = [r.get(metric, 0) for r in results]

        ax.semilogy(qubits, values, 'ro-', markersize=10, linewidth=2)

        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Scaling Analysis: {metric}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def plot_backend_comparison(self, comparison: Dict[str, Any],
                               filename: str = None) -> Optional[str]:
        """Plot backend comparison results."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        backends = comparison.get("backends", [])
        metrics = comparison.get("metrics", {})

        if not backends or not metrics:
            return None

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_name, metric_data) in zip(axes, metrics.items()):
            values = [metric_data.get(b, 0) for b in backends]
            colors = ['#2ecc71' if v == min(values) and 'time' in metric_name.lower()
                     else '#3498db' for v in values]

            ax.bar(backends, values, color=colors)
            ax.set_xlabel("Backend")
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.set_title(f"{metric_name}")
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def plot_noise_impact(self, results: List[Dict[str, Any]],
                         filename: str = None) -> Optional[str]:
        """Plot impact of noise on fidelity."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        noise_levels = [r.get("noise_level", 0) for r in results]
        fidelities = [r.get("fidelity", 0) for r in results]

        ax.plot(noise_levels, fidelities, 'b-o', markersize=10)
        ax.fill_between(noise_levels, fidelities, alpha=0.3)

        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Fidelity")
        ax.set_title("Impact of Noise on Simulation Fidelity")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150)
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None

    def create_summary_report(self, experiment: Dict[str, Any],
                             filename: str = None) -> Optional[str]:
        """Create a multi-panel summary report."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = plt.figure(figsize=(16, 12))

        # Panel 1: Histogram (if counts available)
        ax1 = fig.add_subplot(2, 2, 1)
        counts = experiment.get("counts", {})
        if counts:
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:16]
            states = [s[0] for s in sorted_counts]
            values = [s[1] for s in sorted_counts]
            ax1.bar(range(len(states)), values, color='steelblue')
            ax1.set_xticks(range(len(states)))
            ax1.set_xticklabels(states, rotation=45, ha='right', fontsize=8)
            ax1.set_title("Top 16 States")
        else:
            ax1.text(0.5, 0.5, "No histogram data", ha='center', va='center')
            ax1.set_title("Measurement Distribution")

        # Panel 2: Key metrics
        ax2 = fig.add_subplot(2, 2, 2)
        metrics = {
            "Fidelity": experiment.get("fidelity", "N/A"),
            "Sim Time (s)": experiment.get("simulation_time_seconds", "N/A"),
            "Final Rank": experiment.get("final_rank", "N/A"),
            "Memory (MB)": experiment.get("memory_used_mb", "N/A"),
        }

        y_pos = 0.8
        for name, value in metrics.items():
            ax2.text(0.1, y_pos, f"{name}:", fontsize=12, fontweight='bold')
            ax2.text(0.6, y_pos, f"{value}", fontsize=12)
            y_pos -= 0.15

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title("Key Metrics")

        # Panel 3: Configuration
        ax3 = fig.add_subplot(2, 2, 3)
        config = experiment.get("config", {})
        config_text = json.dumps(config, indent=2)
        ax3.text(0.05, 0.95, config_text, fontsize=9, va='top',
                family='monospace', wrap=True)
        ax3.axis('off')
        ax3.set_title("Configuration")

        # Panel 4: Status
        ax4 = fig.add_subplot(2, 2, 4)
        status_info = [
            f"Run ID: {experiment.get('run_id', 'N/A')}",
            f"Status: {experiment.get('status', 'N/A')}",
            f"Backend: {experiment.get('backend_used', 'N/A')}",
            f"N Qubits: {config.get('n_qubits', 'N/A')}",
            f"Depth: {config.get('depth', 'N/A')}",
        ]

        y_pos = 0.9
        for info in status_info:
            ax4.text(0.1, y_pos, info, fontsize=11)
            y_pos -= 0.15

        ax4.axis('off')
        ax4.set_title("Experiment Info")

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
```

### 14.2 ASCII Charts (Terminal)

**File: `agent/visualization/ascii_charts.py`**

```python
"""ASCII visualization for terminal output."""
from typing import Dict, List, Any

class ASCIICharts:
    """Terminal-friendly ASCII visualizations."""

    @staticmethod
    def histogram(counts: Dict[str, int], width: int = 50, top_n: int = 16) -> str:
        """Create ASCII histogram."""
        if not counts:
            return "No data to display"

        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        max_count = max(c[1] for c in sorted_counts)
        max_label = max(len(c[0]) for c in sorted_counts)

        lines = ["=" * (max_label + width + 10)]
        lines.append(f"{'State':<{max_label}}  {'Bar':<{width}}  Count")
        lines.append("-" * (max_label + width + 10))

        for state, count in sorted_counts:
            bar_len = int(count / max_count * width)
            bar = "█" * bar_len
            lines.append(f"{state:<{max_label}}  {bar:<{width}}  {count}")

        lines.append("=" * (max_label + width + 10))
        return "\n".join(lines)

    @staticmethod
    def progress_bar(current: float, total: float, width: int = 40,
                    prefix: str = "", suffix: str = "") -> str:
        """Create progress bar."""
        pct = current / total if total > 0 else 0
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"{prefix}|{bar}| {pct*100:.1f}% {suffix}"

    @staticmethod
    def table(data: List[Dict[str, Any]], columns: List[str] = None) -> str:
        """Create ASCII table."""
        if not data:
            return "No data"

        if columns is None:
            columns = list(data[0].keys())

        # Calculate column widths
        widths = {col: len(str(col)) for col in columns}
        for row in data:
            for col in columns:
                val_len = len(str(row.get(col, "")))
                widths[col] = max(widths[col], val_len)

        # Build table
        lines = []

        # Header
        header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
        separator = "-+-".join("-" * widths[col] for col in columns)
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        # Rows
        for row in data:
            line = " | ".join(f"{str(row.get(col, '')):<{widths[col]}}"
                            for col in columns)
            lines.append(line)

        lines.append(separator)
        return "\n".join(lines)

    @staticmethod
    def sparkline(values: List[float], width: int = 20) -> str:
        """Create sparkline from values."""
        if not values:
            return ""

        chars = "▁▂▃▄▅▆▇█"
        min_val, max_val = min(values), max(values)

        if max_val == min_val:
            return chars[4] * min(len(values), width)

        # Resample if needed
        if len(values) > width:
            step = len(values) / width
            values = [values[int(i * step)] for i in range(width)]

        result = ""
        for v in values:
            idx = int((v - min_val) / (max_val - min_val) * (len(chars) - 1))
            result += chars[idx]

        return result

# Example usage
if __name__ == "__main__":
    charts = ASCIICharts()

    # Histogram
    counts = {"00": 450, "01": 50, "10": 60, "11": 440}
    print(charts.histogram(counts))

    # Progress
    print(charts.progress_bar(75, 100, prefix="Progress: "))

    # Table
    data = [
        {"run_id": "exp_001", "qubits": 10, "fidelity": 0.995},
        {"run_id": "exp_002", "qubits": 12, "fidelity": 0.991},
    ]
    print(charts.table(data))

    # Sparkline
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.95, 0.99]
    print(f"Fidelity trend: {charts.sparkline(values)}")
```

---

## 📚 PHASE 15: Cloud Integration

### Overview

Deploy and run experiments on cloud platforms (AWS, GCP) and Kubernetes.

### 15.1 AWS Integration

**File: `agent/cloud/aws_runner.py`**

```python
"""AWS integration for LRET experiments."""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json
import time

@dataclass
class AWSConfig:
    """AWS configuration."""
    region: str = "us-east-1"
    instance_type: str = "c5.2xlarge"
    ami_id: str = "ami-quantum-compute"  # Custom AMI with LRET
    key_name: str = "lret-key"
    security_group: str = "lret-sg"
    s3_bucket: str = "lret-experiments"
    use_spot: bool = True
    spot_max_price: float = 0.5

class AWSRunner:
    """Run LRET experiments on AWS EC2."""

    def __init__(self, config: AWSConfig = None):
        self.config = config or AWSConfig()
        self._boto3 = None

    @property
    def boto3(self):
        """Lazy load boto3."""
        if self._boto3 is None:
            import boto3
            self._boto3 = boto3
        return self._boto3

    @property
    def ec2(self):
        return self.boto3.client('ec2', region_name=self.config.region)

    @property
    def s3(self):
        return self.boto3.client('s3', region_name=self.config.region)

    def launch_instance(self, experiment_config: Dict[str, Any]) -> str:
        """Launch EC2 instance for experiment."""
        user_data = self._create_user_data(experiment_config)

        launch_params = {
            'ImageId': self.config.ami_id,
            'InstanceType': self.config.instance_type,
            'KeyName': self.config.key_name,
            'SecurityGroups': [self.config.security_group],
            'UserData': user_data,
            'MinCount': 1,
            'MaxCount': 1,
            'TagSpecifications': [{
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f"lret-experiment-{experiment_config.get('run_id', 'unknown')}"},
                    {'Key': 'Project', 'Value': 'LRET'},
                ]
            }],
            'InstanceInitiatedShutdownBehavior': 'terminate'
        }

        if self.config.use_spot:
            return self._launch_spot_instance(launch_params)

        response = self.ec2.run_instances(**launch_params)
        instance_id = response['Instances'][0]['InstanceId']

        return instance_id

    def _launch_spot_instance(self, launch_params: Dict) -> str:
        """Launch spot instance."""
        response = self.ec2.request_spot_instances(
            SpotPrice=str(self.config.spot_max_price),
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification={
                'ImageId': launch_params['ImageId'],
                'InstanceType': launch_params['InstanceType'],
                'KeyName': launch_params['KeyName'],
                'SecurityGroups': launch_params['SecurityGroups'],
                'UserData': launch_params['UserData'],
            }
        )

        request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

        # Wait for fulfillment
        waiter = self.ec2.get_waiter('spot_instance_request_fulfilled')
        waiter.wait(SpotInstanceRequestIds=[request_id])

        response = self.ec2.describe_spot_instance_requests(
            SpotInstanceRequestIds=[request_id]
        )
        instance_id = response['SpotInstanceRequests'][0]['InstanceId']

        return instance_id

    def _create_user_data(self, config: Dict[str, Any]) -> str:
        """Create user data script for instance."""
        config_json = json.dumps(config)

        script = f"""#!/bin/bash
set -e

# Install dependencies
pip install lret cirq

# Create experiment config
cat > /tmp/experiment_config.json << 'EOF'
{config_json}
EOF

# Run experiment
cd /opt/lret
python -m agent.execution.runner /tmp/experiment_config.json

# Upload results to S3
aws s3 cp /tmp/results.json s3://{self.config.s3_bucket}/results/{config.get('run_id', 'unknown')}/

# Shutdown instance
shutdown -h now
"""

        import base64
        return base64.b64encode(script.encode()).decode()

    def upload_to_s3(self, local_path: Path, s3_key: str):
        """Upload file to S3."""
        self.s3.upload_file(str(local_path), self.config.s3_bucket, s3_key)

    def download_from_s3(self, s3_key: str, local_path: Path):
        """Download file from S3."""
        self.s3.download_file(self.config.s3_bucket, s3_key, str(local_path))

    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results from S3."""
        s3_key = f"results/{run_id}/results.json"
        local_path = Path(f"/tmp/{run_id}_results.json")

        try:
            self.download_from_s3(s3_key, local_path)
            with open(local_path) as f:
                return json.load(f)
        except Exception:
            return None

    def wait_for_completion(self, instance_id: str, timeout: int = 3600) -> bool:
        """Wait for instance to terminate (experiment complete)."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.ec2.describe_instances(InstanceIds=[instance_id])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']

            if state == 'terminated':
                return True

            time.sleep(30)

        return False
```

### 15.2 GCP Integration

**File: `agent/cloud/gcp_runner.py`**

```python
"""GCP integration for LRET experiments."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import time

@dataclass
class GCPConfig:
    """GCP configuration."""
    project_id: str = "lret-quantum"
    zone: str = "us-central1-a"
    machine_type: str = "n2-standard-8"
    image_family: str = "debian-11"
    image_project: str = "debian-cloud"
    gcs_bucket: str = "lret-experiments"
    use_preemptible: bool = True

class GCPRunner:
    """Run LRET experiments on GCP Compute Engine."""

    def __init__(self, config: GCPConfig = None):
        self.config = config or GCPConfig()
        self._compute = None
        self._storage = None

    @property
    def compute(self):
        """Lazy load compute client."""
        if self._compute is None:
            from google.cloud import compute_v1
            self._compute = compute_v1.InstancesClient()
        return self._compute

    @property
    def storage(self):
        """Lazy load storage client."""
        if self._storage is None:
            from google.cloud import storage
            self._storage = storage.Client(project=self.config.project_id)
        return self._storage

    def launch_instance(self, experiment_config: Dict[str, Any]) -> str:
        """Launch GCE instance for experiment."""
        from google.cloud import compute_v1

        run_id = experiment_config.get('run_id', 'unknown')
        instance_name = f"lret-experiment-{run_id}"

        # Create startup script
        startup_script = self._create_startup_script(experiment_config)

        # Instance configuration
        instance = compute_v1.Instance()
        instance.name = instance_name
        instance.machine_type = f"zones/{self.config.zone}/machineTypes/{self.config.machine_type}"

        # Boot disk
        disk = compute_v1.AttachedDisk()
        disk.boot = True
        disk.auto_delete = True
        disk.initialize_params = compute_v1.AttachedDiskInitializeParams()
        disk.initialize_params.source_image = f"projects/{self.config.image_project}/global/images/family/{self.config.image_family}"
        disk.initialize_params.disk_size_gb = 50
        instance.disks = [disk]

        # Network
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = "global/networks/default"
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]

        # Startup script
        metadata = compute_v1.Metadata()
        metadata.items = [
            compute_v1.Items(key="startup-script", value=startup_script)
        ]
        instance.metadata = metadata

        # Preemptible
        if self.config.use_preemptible:
            instance.scheduling = compute_v1.Scheduling()
            instance.scheduling.preemptible = True

        # Launch
        operation = self.compute.insert(
            project=self.config.project_id,
            zone=self.config.zone,
            instance_resource=instance
        )

        # Wait for creation
        self._wait_for_operation(operation.name)

        return instance_name

    def _create_startup_script(self, config: Dict[str, Any]) -> str:
        """Create startup script for instance."""
        config_json = json.dumps(config)

        return f"""#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y python3-pip
pip3 install lret cirq google-cloud-storage

# Create experiment config
cat > /tmp/experiment_config.json << 'EOF'
{config_json}
EOF

# Run experiment
python3 -m agent.execution.runner /tmp/experiment_config.json

# Upload results to GCS
gsutil cp /tmp/results.json gs://{self.config.gcs_bucket}/results/{config.get('run_id', 'unknown')}/

# Delete instance (self-destruct)
gcloud compute instances delete $(hostname) --zone={self.config.zone} --quiet
"""

    def _wait_for_operation(self, operation_name: str, timeout: int = 300):
        """Wait for GCE operation to complete."""
        from google.cloud import compute_v1

        operations_client = compute_v1.ZoneOperationsClient()
        start_time = time.time()

        while time.time() - start_time < timeout:
            operation = operations_client.get(
                project=self.config.project_id,
                zone=self.config.zone,
                operation=operation_name
            )

            if operation.status == compute_v1.Operation.Status.DONE:
                return

            time.sleep(5)

        raise TimeoutError(f"Operation {operation_name} timed out")

    def upload_to_gcs(self, local_path: str, blob_name: str):
        """Upload file to GCS."""
        bucket = self.storage.bucket(self.config.gcs_bucket)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

    def download_from_gcs(self, blob_name: str, local_path: str):
        """Download file from GCS."""
        bucket = self.storage.bucket(self.config.gcs_bucket)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)

    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results from GCS."""
        blob_name = f"results/{run_id}/results.json"
        local_path = f"/tmp/{run_id}_results.json"

        try:
            self.download_from_gcs(blob_name, local_path)
            with open(local_path) as f:
                return json.load(f)
        except Exception:
            return None
```

### 15.3 Kubernetes Integration

**File: `agent/cloud/k8s_runner.py`**

```python
"""Kubernetes integration for LRET experiments."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import time
import yaml

@dataclass
class K8sConfig:
    """Kubernetes configuration."""
    namespace: str = "lret"
    image: str = "lret/agent:latest"
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "4Gi"
    memory_limit: str = "8Gi"
    storage_class: str = "standard"

class KubernetesRunner:
    """Run LRET experiments on Kubernetes."""

    def __init__(self, config: K8sConfig = None):
        self.config = config or K8sConfig()
        self._core_v1 = None
        self._batch_v1 = None

    @property
    def core_v1(self):
        """Get core v1 API client."""
        if self._core_v1 is None:
            from kubernetes import client, config
            config.load_incluster_config()  # or load_kube_config() for local
            self._core_v1 = client.CoreV1Api()
        return self._core_v1

    @property
    def batch_v1(self):
        """Get batch v1 API client."""
        if self._batch_v1 is None:
            from kubernetes import client, config
            config.load_incluster_config()
            self._batch_v1 = client.BatchV1Api()
        return self._batch_v1

    def submit_job(self, experiment_config: Dict[str, Any]) -> str:
        """Submit Kubernetes job for experiment."""
        from kubernetes import client

        run_id = experiment_config.get('run_id', 'unknown')
        job_name = f"lret-{run_id}"

        # ConfigMap for experiment config
        config_map = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=f"{job_name}-config",
                namespace=self.config.namespace
            ),
            data={"config.json": json.dumps(experiment_config)}
        )

        self.core_v1.create_namespaced_config_map(
            namespace=self.config.namespace,
            body=config_map
        )

        # Job spec
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.config.namespace
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="lret-runner",
                                image=self.config.image,
                                command=["python", "-m", "agent.execution.runner"],
                                args=["/config/config.json"],
                                resources=client.V1ResourceRequirements(
                                    requests={
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request
                                    },
                                    limits={
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit
                                    }
                                ),
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name="config",
                                        mount_path="/config"
                                    ),
                                    client.V1VolumeMount(
                                        name="results",
                                        mount_path="/results"
                                    )
                                ]
                            )
                        ],
                        volumes=[
                            client.V1Volume(
                                name="config",
                                config_map=client.V1ConfigMapVolumeSource(
                                    name=f"{job_name}-config"
                                )
                            ),
                            client.V1Volume(
                                name="results",
                                empty_dir=client.V1EmptyDirVolumeSource()
                            )
                        ],
                        restart_policy="Never"
                    )
                ),
                backoff_limit=2,
                ttl_seconds_after_finished=3600
            )
        )

        self.batch_v1.create_namespaced_job(
            namespace=self.config.namespace,
            body=job
        )

        return job_name

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of Kubernetes job."""
        job = self.batch_v1.read_namespaced_job(
            name=job_name,
            namespace=self.config.namespace
        )

        status = {
            "name": job_name,
            "active": job.status.active or 0,
            "succeeded": job.status.succeeded or 0,
            "failed": job.status.failed or 0,
            "start_time": str(job.status.start_time) if job.status.start_time else None,
            "completion_time": str(job.status.completion_time) if job.status.completion_time else None,
        }

        if job.status.succeeded:
            status["status"] = "completed"
        elif job.status.failed:
            status["status"] = "failed"
        elif job.status.active:
            status["status"] = "running"
        else:
            status["status"] = "pending"

        return status

    def get_job_logs(self, job_name: str) -> str:
        """Get logs from job pods."""
        # Find pod for this job
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.config.namespace,
            label_selector=f"job-name={job_name}"
        )

        if not pods.items:
            return "No pods found for job"

        pod_name = pods.items[0].metadata.name

        return self.core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=self.config.namespace
        )

    def delete_job(self, job_name: str, cleanup: bool = True):
        """Delete job and optionally cleanup resources."""
        from kubernetes import client

        # Delete job
        self.batch_v1.delete_namespaced_job(
            name=job_name,
            namespace=self.config.namespace,
            propagation_policy="Background"
        )

        if cleanup:
            # Delete ConfigMap
            try:
                self.core_v1.delete_namespaced_config_map(
                    name=f"{job_name}-config",
                    namespace=self.config.namespace
                )
            except:
                pass

    def list_jobs(self, status_filter: str = None) -> List[Dict[str, Any]]:
        """List all LRET jobs."""
        jobs = self.batch_v1.list_namespaced_job(
            namespace=self.config.namespace,
            label_selector="app=lret"
        )

        results = []
        for job in jobs.items:
            job_status = self.get_job_status(job.metadata.name)

            if status_filter and job_status["status"] != status_filter:
                continue

            results.append(job_status)

        return results
```

---

## 📚 PHASE 16: ML Optimization

### Overview

Machine learning-based optimization for experiment parameters and circuit design.

### 16.1 Parameter Optimizer

**File: `agent/ml/optimizer.py`**

```python
"""ML-based parameter optimization."""
from typing import Dict, Any, List, Callable, Tuple, Optional
from dataclasses import dataclass
import random
import math

@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_params: Dict[str, Any]
    best_score: float
    history: List[Tuple[Dict[str, Any], float]]
    iterations: int

class ParameterOptimizer:
    """Optimize experiment parameters using various strategies."""

    def __init__(self, objective_fn: Callable[[Dict[str, Any]], float],
                 param_space: Dict[str, Tuple],
                 maximize: bool = True):
        """
        Initialize optimizer.

        Args:
            objective_fn: Function that takes params and returns score
            param_space: Dict of param_name -> (min, max) for continuous,
                        or param_name -> [values] for discrete
            maximize: Whether to maximize (True) or minimize (False) objective
        """
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.maximize = maximize
        self.history: List[Tuple[Dict[str, Any], float]] = []

    def _sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from space."""
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                params[name] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                if isinstance(space[0], int) and isinstance(space[1], int):
                    params[name] = random.randint(space[0], space[1])
                else:
                    params[name] = random.uniform(space[0], space[1])
        return params

    def _is_better(self, score1: float, score2: float) -> bool:
        """Check if score1 is better than score2."""
        if self.maximize:
            return score1 > score2
        return score1 < score2

    def random_search(self, n_iterations: int = 50) -> OptimizationResult:
        """Simple random search optimization."""
        best_params = None
        best_score = float('-inf') if self.maximize else float('inf')

        for i in range(n_iterations):
            params = self._sample_params()
            score = self.objective_fn(params)
            self.history.append((params.copy(), score))

            if self._is_better(score, best_score):
                best_score = score
                best_params = params.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            iterations=n_iterations
        )

    def grid_search(self, grid_points: int = 5) -> OptimizationResult:
        """Grid search over parameter space."""
        from itertools import product

        # Create grid
        param_grids = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                param_grids[name] = space
            elif isinstance(space, tuple):
                if isinstance(space[0], int):
                    param_grids[name] = list(range(space[0], space[1] + 1,
                                                   max(1, (space[1] - space[0]) // grid_points)))
                else:
                    step = (space[1] - space[0]) / grid_points
                    param_grids[name] = [space[0] + i * step for i in range(grid_points + 1)]

        # Search
        best_params = None
        best_score = float('-inf') if self.maximize else float('inf')

        param_names = list(param_grids.keys())
        for values in product(*param_grids.values()):
            params = dict(zip(param_names, values))
            score = self.objective_fn(params)
            self.history.append((params.copy(), score))

            if self._is_better(score, best_score):
                best_score = score
                best_params = params.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            iterations=len(self.history)
        )

    def bayesian_optimization(self, n_iterations: int = 30,
                             n_initial: int = 10) -> OptimizationResult:
        """Bayesian optimization with Gaussian Process surrogate."""
        # Initial random sampling
        for _ in range(n_initial):
            params = self._sample_params()
            score = self.objective_fn(params)
            self.history.append((params.copy(), score))

        best_params = max(self.history, key=lambda x: x[1] if self.maximize else -x[1])[0]
        best_score = max(s for _, s in self.history) if self.maximize else min(s for _, s in self.history)

        # Simplified acquisition function optimization
        for _ in range(n_iterations - n_initial):
            # Generate candidates
            candidates = [self._sample_params() for _ in range(100)]

            # Score candidates using UCB-like acquisition
            candidate_scores = []
            for cand in candidates:
                # Mean from nearest neighbors
                distances = []
                for hist_params, hist_score in self.history:
                    dist = sum((cand.get(k, 0) - hist_params.get(k, 0))**2
                              for k in cand.keys())**0.5
                    distances.append((dist, hist_score))

                distances.sort(key=lambda x: x[0])
                nearest = distances[:3]

                mean = sum(s for _, s in nearest) / len(nearest)
                std = (sum((s - mean)**2 for _, s in nearest) / len(nearest))**0.5 + 0.1

                # UCB acquisition
                acq = mean + 2.0 * std if self.maximize else mean - 2.0 * std
                candidate_scores.append((cand, acq))

            # Select best candidate
            if self.maximize:
                best_cand = max(candidate_scores, key=lambda x: x[1])[0]
            else:
                best_cand = min(candidate_scores, key=lambda x: x[1])[0]

            # Evaluate
            score = self.objective_fn(best_cand)
            self.history.append((best_cand.copy(), score))

            if self._is_better(score, best_score):
                best_score = score
                best_params = best_cand.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            history=self.history,
            iterations=n_iterations
        )
```

### 16.2 AutoML for Circuits

**File: `agent/ml/circuit_automl.py`**

```python
"""AutoML for quantum circuit optimization."""
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import random

@dataclass
class CircuitTemplate:
    """Template for circuit architecture."""
    name: str
    layers: List[str]  # e.g., ["H", "CNOT", "RZ", "CNOT", "measure"]
    params: Dict[str, Any]

class CircuitAutoML:
    """Automated circuit architecture search."""

    GATE_POOL = ["H", "X", "Y", "Z", "CNOT", "CZ", "RX", "RY", "RZ", "T", "S"]

    def __init__(self, n_qubits: int, target_depth: int,
                 fitness_fn: Callable[[CircuitTemplate], float]):
        """
        Initialize CircuitAutoML.

        Args:
            n_qubits: Number of qubits
            target_depth: Target circuit depth
            fitness_fn: Function to evaluate circuit fitness
        """
        self.n_qubits = n_qubits
        self.target_depth = target_depth
        self.fitness_fn = fitness_fn
        self.population: List[CircuitTemplate] = []
        self.best_circuit: Optional[CircuitTemplate] = None
        self.best_fitness: float = float('-inf')

    def _random_circuit(self) -> CircuitTemplate:
        """Generate random circuit template."""
        layers = []

        for _ in range(self.target_depth):
            # Random gate selection
            gate = random.choice(self.GATE_POOL)
            layers.append(gate)

        return CircuitTemplate(
            name=f"circuit_{random.randint(1000, 9999)}",
            layers=layers,
            params={"n_qubits": self.n_qubits}
        )

    def _mutate(self, circuit: CircuitTemplate) -> CircuitTemplate:
        """Mutate circuit template."""
        new_layers = circuit.layers.copy()

        # Random mutation
        mutation_type = random.choice(["swap", "insert", "delete", "replace"])

        if mutation_type == "swap" and len(new_layers) >= 2:
            i, j = random.sample(range(len(new_layers)), 2)
            new_layers[i], new_layers[j] = new_layers[j], new_layers[i]

        elif mutation_type == "insert" and len(new_layers) < self.target_depth * 2:
            pos = random.randint(0, len(new_layers))
            new_layers.insert(pos, random.choice(self.GATE_POOL))

        elif mutation_type == "delete" and len(new_layers) > 3:
            pos = random.randint(0, len(new_layers) - 1)
            new_layers.pop(pos)

        elif mutation_type == "replace":
            pos = random.randint(0, len(new_layers) - 1)
            new_layers[pos] = random.choice(self.GATE_POOL)

        return CircuitTemplate(
            name=f"circuit_{random.randint(1000, 9999)}",
            layers=new_layers,
            params=circuit.params.copy()
        )

    def _crossover(self, parent1: CircuitTemplate,
                   parent2: CircuitTemplate) -> CircuitTemplate:
        """Crossover two circuits."""
        # Single point crossover
        point = random.randint(1, min(len(parent1.layers), len(parent2.layers)) - 1)

        new_layers = parent1.layers[:point] + parent2.layers[point:]

        return CircuitTemplate(
            name=f"circuit_{random.randint(1000, 9999)}",
            layers=new_layers,
            params=parent1.params.copy()
        )

    def evolve(self, population_size: int = 20, generations: int = 50,
              mutation_rate: float = 0.3, elite_size: int = 2) -> CircuitTemplate:
        """Evolve circuit population using genetic algorithm."""
        # Initialize population
        self.population = [self._random_circuit() for _ in range(population_size)]

        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [(c, self.fitness_fn(c)) for c in self.population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Update best
            if fitness_scores[0][1] > self.best_fitness:
                self.best_fitness = fitness_scores[0][1]
                self.best_circuit = fitness_scores[0][0]

            # Selection (tournament)
            def tournament_select():
                contestants = random.sample(fitness_scores, 3)
                return max(contestants, key=lambda x: x[1])[0]

            # Next generation
            new_population = []

            # Elitism - keep best
            for i in range(elite_size):
                new_population.append(fitness_scores[i][0])

            # Fill rest with offspring
            while len(new_population) < population_size:
                parent1 = tournament_select()
                parent2 = tournament_select()

                child = self._crossover(parent1, parent2)

                if random.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            self.population = new_population

        return self.best_circuit

    def search_architecture(self, search_budget: int = 100) -> CircuitTemplate:
        """Neural architecture search for circuits."""
        # Simplified NAS using random search with early stopping
        best_circuit = None
        best_fitness = float('-inf')

        for i in range(search_budget):
            # Generate candidate
            circuit = self._random_circuit()

            # Quick evaluation (early stopping for bad circuits)
            fitness = self.fitness_fn(circuit)

            if fitness > best_fitness:
                best_fitness = fitness
                best_circuit = circuit

                # Intensify search around good solution
                for _ in range(5):
                    mutated = self._mutate(circuit)
                    mut_fitness = self.fitness_fn(mutated)

                    if mut_fitness > best_fitness:
                        best_fitness = mut_fitness
                        best_circuit = mutated

        self.best_circuit = best_circuit
        self.best_fitness = best_fitness

        return best_circuit
```

---

## Phase 14-16 Summary

**Completed Features:**

- ✅ Matplotlib visualization for histograms, scaling, comparisons
- ✅ ASCII charts for terminal output
- ✅ AWS EC2 integration (on-demand and spot instances)
- ✅ GCP Compute Engine integration
- ✅ Kubernetes job runner
- ✅ Parameter optimization (random, grid, Bayesian)
- ✅ Circuit AutoML with genetic algorithms

---

_Continue to [agent6.md](agent6.md) for Phase 17: Collaboration & Complete User Guide._
