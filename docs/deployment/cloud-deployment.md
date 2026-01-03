# Cloud Deployment Guide

Deploy LRET on major cloud platforms for scalable quantum simulations.

## Overview

LRET can be deployed on cloud platforms for:
- **Large-scale simulations** (20+ qubits)
- **Distributed computing** with MPI
- **GPU acceleration** for faster simulations
- **API services** for remote access
- **Batch processing** of multiple circuits

## Amazon Web Services (AWS)

### EC2 Instance Deployment

#### 1. Choose Instance Type

| Instance Type | vCPUs | Memory | GPU | Use Case | Cost/hour |
|---------------|-------|--------|-----|----------|-----------|
| `t3.2xlarge` | 8 | 32 GB | No | Small simulations | $0.33 |
| `c6i.8xlarge` | 32 | 64 GB | No | CPU-intensive | $1.36 |
| `r6i.8xlarge` | 32 | 256 GB | No | Memory-intensive | $2.02 |
| `p3.2xlarge` | 8 | 61 GB | V100 | GPU acceleration | $3.06 |
| `p4d.24xlarge` | 96 | 1152 GB | 8x A100 | Large-scale GPU | $32.77 |

#### 2. Launch Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Ubuntu 22.04
    --instance-type c6i.8xlarge \
    --key-name my-key \
    --security-group-ids sg-xxxxx \
    --subnet-id subnet-xxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=LRET-Simulator}]'
```

#### 3. Connect and Install

```bash
# SSH to instance
ssh -i my-key.pem ubuntu@<instance-ip>

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu

# Pull LRET image
docker pull kunal5556/lret:latest

# Or build from source
git clone https://github.com/kunal5556/LRET.git
cd LRET
docker build -t lret:local .
```

#### 4. Run Simulation

```bash
docker run -v $(pwd)/results:/output lret:latest \
    lret --qubits 20 --gates "..." --output /output/results.csv
```

### ECS (Elastic Container Service)

Deploy LRET as a containerized service:

#### 1. Create Task Definition

`lret-task-definition.json`:
```json
{
  "family": "lret-simulator",
  "containerDefinitions": [{
    "name": "lret",
    "image": "kunal5556/lret:latest",
    "memory": 16384,
    "cpu": 4096,
    "essential": true,
    "environment": [
      {"name": "LRET_THREADS", "value": "16"},
      {"name": "LRET_MEMORY_LIMIT", "value": "16GB"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/lret",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

Register:
```bash
aws ecs register-task-definition --cli-input-json file://lret-task-definition.json
```

#### 2. Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name lret-cluster
```

#### 3. Run Task

```bash
aws ecs run-task \
    --cluster lret-cluster \
    --task-definition lret-simulator \
    --count 1 \
    --launch-type FARGATE
```

### AWS Batch

For batch processing multiple simulations:

#### 1. Create Compute Environment

```bash
aws batch create-compute-environment \
    --compute-environment-name lret-compute-env \
    --type MANAGED \
    --compute-resources type=EC2,minvCpus=0,maxvCpus=256,instanceTypes=c6i.8xlarge
```

#### 2. Create Job Queue

```bash
aws batch create-job-queue \
    --job-queue-name lret-job-queue \
    --compute-environment-order order=1,computeEnvironment=lret-compute-env
```

#### 3. Register Job Definition

```bash
aws batch register-job-definition \
    --job-definition-name lret-simulation \
    --type container \
    --container-properties '{
        "image": "kunal5556/lret:latest",
        "vcpus": 16,
        "memory": 32768,
        "command": ["lret", "--qubits", "Ref::qubits", "--gates", "Ref::gates"]
    }'
```

#### 4. Submit Jobs

```bash
aws batch submit-job \
    --job-name lret-job-1 \
    --job-queue lret-job-queue \
    --job-definition lret-simulation \
    --parameters qubits=20,gates="H 0; CNOT 0 1"
```

### S3 Integration

Store inputs and outputs in S3:

```bash
# Upload circuit definition
aws s3 cp circuit.json s3://my-lret-bucket/inputs/

# Run simulation with S3 I/O
docker run \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    lret:latest \
    lret --input s3://my-lret-bucket/inputs/circuit.json \
         --output s3://my-lret-bucket/outputs/results.csv
```

---

## Google Cloud Platform (GCP)

### Compute Engine Deployment

#### 1. Choose Machine Type

```bash
gcloud compute machine-types list --filter="zone:us-central1-a"
```

Recommended types:
- `n2-highmem-32`: 32 vCPUs, 256 GB RAM
- `c2-standard-60`: 60 vCPUs, 240 GB RAM (CPU-optimized)
- `a2-highgpu-1g`: 12 vCPUs, 85 GB RAM, 1x A100 GPU

#### 2. Create Instance

```bash
gcloud compute instances create lret-vm \
    --machine-type=n2-highmem-32 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --zone=us-central1-a
```

#### 3. SSH and Install

```bash
gcloud compute ssh lret-vm --zone=us-central1-a

# Install Docker and LRET (same as AWS)
```

### Google Kubernetes Engine (GKE)

Deploy LRET on Kubernetes:

#### 1. Create GKE Cluster

```bash
gcloud container clusters create lret-cluster \
    --num-nodes=3 \
    --machine-type=n2-highmem-16 \
    --zone=us-central1-a
```

#### 2. Create Deployment

`lret-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lret-simulator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lret
  template:
    metadata:
      labels:
        app: lret
    spec:
      containers:
      - name: lret
        image: kunal5556/lret:latest
        resources:
          requests:
            memory: "32Gi"
            cpu: "16"
          limits:
            memory: "64Gi"
            cpu: "32"
        env:
        - name: LRET_THREADS
          value: "16"
```

Apply:
```bash
kubectl apply -f lret-deployment.yaml
```

#### 3. Create Job

`lret-job.yaml`:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: lret-simulation
spec:
  template:
    spec:
      containers:
      - name: lret
        image: kunal5556/lret:latest
        command: ["lret", "--qubits", "20", "--gates", "H 0; CNOT 0 1"]
      restartPolicy: Never
  backoffLimit: 3
```

### Cloud Storage Integration

```bash
# Upload to GCS
gsutil cp circuit.json gs://my-lret-bucket/inputs/

# Run with GCS I/O
docker run lret:latest \
    lret --input gs://my-lret-bucket/inputs/circuit.json \
         --output gs://my-lret-bucket/outputs/results.csv
```

---

## Microsoft Azure

### Azure Virtual Machines

#### 1. Choose VM Size

```bash
az vm list-sizes --location eastus
```

Recommended:
- `Standard_D32s_v3`: 32 vCPUs, 128 GB RAM
- `Standard_E32s_v3`: 32 vCPUs, 256 GB RAM (memory-optimized)
- `Standard_NC24s_v3`: 24 vCPUs, 448 GB RAM, 4x V100 GPUs

#### 2. Create VM

```bash
az vm create \
    --resource-group lret-rg \
    --name lret-vm \
    --image UbuntuLTS \
    --size Standard_D32s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys
```

#### 3. Install and Run

```bash
az vm run-command invoke \
    --resource-group lret-rg \
    --name lret-vm \
    --command-id RunShellScript \
    --scripts "curl -fsSL https://get.docker.com | sudo sh"
```

### Azure Container Instances

Quick container deployment:

```bash
az container create \
    --resource-group lret-rg \
    --name lret-container \
    --image kunal5556/lret:latest \
    --cpu 16 \
    --memory 32 \
    --command-line "lret --qubits 20 --gates 'H 0; CNOT 0 1'"
```

### Azure Kubernetes Service (AKS)

```bash
# Create cluster
az aks create \
    --resource-group lret-rg \
    --name lret-aks \
    --node-count 3 \
    --node-vm-size Standard_D32s_v3 \
    --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group lret-rg --name lret-aks

# Deploy (same as GKE)
kubectl apply -f lret-deployment.yaml
```

### Azure Blob Storage

```bash
# Upload
az storage blob upload \
    --account-name lretstorage \
    --container-name inputs \
    --name circuit.json \
    --file circuit.json

# Run with Azure Storage
docker run lret:latest \
    lret --input azure://inputs/circuit.json \
         --output azure://outputs/results.csv
```

---

## Cost Optimization

### Spot/Preemptible Instances

**AWS Spot Instances:**
```bash
aws ec2 request-spot-instances \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://spot-config.json
```

**GCP Preemptible VMs:**
```bash
gcloud compute instances create lret-preemptible \
    --preemptible \
    --machine-type=n2-highmem-32
```

**Azure Spot VMs:**
```bash
az vm create \
    --resource-group lret-rg \
    --name lret-spot \
    --priority Spot \
    --max-price -1 \
    --size Standard_D32s_v3
```

### Auto-Scaling

Set up auto-scaling based on workload:

**AWS Auto Scaling:**
```bash
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name lret-asg \
    --min-size 0 \
    --max-size 10 \
    --desired-capacity 1
```

**GKE Autoscaling:**
```bash
kubectl autoscale deployment lret-simulator --min=1 --max=10 --cpu-percent=80
```

### Reserved Instances

Save up to 70% with 1-3 year commitments:
- **AWS**: Reserved Instances or Savings Plans
- **GCP**: Committed Use Discounts
- **Azure**: Reserved VM Instances

---

## Monitoring and Logging

### AWS CloudWatch

```bash
# Enable detailed monitoring
aws ec2 monitor-instances --instance-ids i-xxxxx

# Create custom metrics
aws cloudwatch put-metric-data \
    --namespace LRET \
    --metric-name SimulationTime \
    --value 42.5
```

### GCP Stackdriver

```yaml
# In kubernetes deployment
containers:
- name: lret
  image: kunal5556/lret:latest
  env:
  - name: GOOGLE_APPLICATION_CREDENTIALS
    value: /secrets/gcp-key.json
```

### Azure Monitor

```bash
az monitor metrics alert create \
    --name lret-cpu-alert \
    --resource-group lret-rg \
    --scopes <vm-id> \
    --condition "avg Percentage CPU > 90"
```

---

## Security Best Practices

1. **Use IAM Roles**: Don't hardcode credentials
2. **Private Networks**: Deploy in VPC/VNet
3. **Encryption**: Enable at-rest and in-transit encryption
4. **Firewall Rules**: Restrict access to necessary ports
5. **Regular Updates**: Keep base images updated
6. **Secrets Management**: Use cloud-native secret stores

---

## Example Workflows

### Workflow 1: Large-Scale Batch Processing

```bash
# 1. Upload 100 circuits to S3
for i in {1..100}; do
    aws s3 cp circuit_$i.json s3://lret-bucket/inputs/
done

# 2. Submit batch jobs
for i in {1..100}; do
    aws batch submit-job \
        --job-name lret-job-$i \
        --job-definition lret-simulation \
        --parameters input=s3://lret-bucket/inputs/circuit_$i.json
done

# 3. Monitor progress
aws batch list-jobs --job-queue lret-job-queue --job-status RUNNING
```

### Workflow 2: CI/CD Pipeline

```yaml
# GitHub Actions
name: LRET Cloud Simulation
on: [push]
jobs:
  simulate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to AWS
        run: |
          aws ecs run-task \
            --cluster lret-cluster \
            --task-definition lret-simulator
```

---

## Further Reading

- [Docker Deployment Guide](docker-guide.md)
- [HPC Deployment Guide](hpc-deployment.md)
- [Performance Guide](../developer-guide/06-performance.md)
- [API Reference](../api-reference/)
