# LRET AI Agent - Part 2: Core Implementation (Phases 1-4)

_Previous: [agent1.md](agent1.md) - Quick Reference & Getting Started_  
_Next: [agent3.md](agent3.md) - Backend & Execution Systems_

---

## 📚 PHASE 1: Foundation & Core Infrastructure

### Overview

Build the foundational components: configuration management, LLM abstraction layer, logging, and CLI interface.

### Project Structure

```
agent/
├── __init__.py
├── main.py              # Entry point
├── config.py            # Configuration management
├── cli/
│   └── parser.py        # Command-line argument parsing
├── llm/
│   ├── __init__.py
│   ├── interface.py     # Abstract LLM interface
│   ├── claude_adapter.py
│   ├── gpt_adapter.py
│   └── gemini_adapter.py
├── io/
│   ├── logger.py        # Structured logging
│   └── output.py        # Human-readable output
└── utils/
    ├── constants.py     # Enums and constants
    └── backend_router.py # Backend routing logic
```

### 1.1 Constants and Enums

**File: `agent/utils/constants.py`**

```python
"""Constants and enumerations for LRET AI Agent."""
from enum import Enum

class IntentType(Enum):
    """Types of user intents."""
    RUN_EXPERIMENT = "run_experiment"
    RUN_BENCHMARK = "run_benchmark"
    COMPARE_EXPERIMENTS = "compare_experiments"
    EXPLAIN_RESULTS = "explain_results"
    INSPECT_CODE = "inspect_code"
    PROPOSE_CHANGES = "propose_changes"
    UNKNOWN = "unknown"

class ActionCategory(Enum):
    """Risk categories for actions."""
    READ = "read"    # Safe: reading files, explaining
    RUN = "run"      # Medium: executing simulations
    WRITE = "write"  # High: modifying code

class Backend(Enum):
    """Simulation backends."""
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"
    LRET = "lret"
    FDM = "fdm"
    AUTO = "auto"
    CIRQ = "cirq"

class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"

# Keywords for backend routing
LRET_KEYWORDS = [
    "lret", "rank", "truncation", "tensor", "low-rank",
    "compression", "rank threshold", "svd", "fdm", "density matrix"
]

CIRQ_KEYWORDS = [
    "cirq", "google", "sycamore", "weber", "rainbow", "qsim",
    "hardware", "vqe", "qaoa", "variational", "optimization"
]

# Backend capabilities
BACKEND_CAPABILITIES = {
    "lret": {
        "max_qubits": 50,
        "supports_noise": True,
        "supports_gpu": True,
        "supports_hardware": False,
        "supports_density_matrix": True,
        "rank_tracking": True,
        "best_for": ["large_systems", "noise_simulation", "tensor_network"]
    },
    "cirq": {
        "max_qubits": 30,
        "supports_noise": True,
        "supports_gpu": True,
        "supports_hardware": True,
        "supports_density_matrix": True,
        "rank_tracking": False,
        "best_for": ["circuits", "hardware_sim", "vqe", "qaoa"]
    }
}

# Configuration defaults
DEFAULT_LLM_PROVIDER = LLMProvider.CLAUDE
DEFAULT_BACKEND = Backend.AUTO
DEFAULT_LOG_LEVEL = "INFO"
MAX_MEMORY_SIZE_MB = 100
SESSION_TIMEOUT_MINUTES = 120
```

### 1.2 Configuration Management

**File: `agent/config.py`**

```python
"""Configuration management for LRET AI Agent."""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from agent.utils.constants import (
    LLMProvider, Backend, DEFAULT_LLM_PROVIDER,
    DEFAULT_BACKEND, DEFAULT_LOG_LEVEL
)

class AgentConfig:
    """Centralized configuration for the agent."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv(
            "LRET_AGENT_CONFIG",
            str(Path.home() / ".lret" / "agent_config.json")
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "provider": DEFAULT_LLM_PROVIDER.value,
                "model": "claude-sonnet-4.5",
                "temperature": 0.2,
                "max_tokens": 4096
            },
            "backend": {"default": DEFAULT_BACKEND.value, "auto_select": True},
            "safety": {
                "require_approval_for_write": True,
                "require_approval_for_run": False,
                "dry_run_mode": False
            },
            "memory": {
                "persist_across_sessions": True,
                "max_size_mb": 100,
                "session_timeout_minutes": 120
            },
            "logging": {"level": DEFAULT_LOG_LEVEL, "structured": True, "audit_trail": True},
            "repository": {"root": os.getcwd(), "build_dir": "build"}
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    @property
    def llm_provider(self) -> LLMProvider:
        return LLMProvider(self.get("llm.provider"))

    @property
    def llm_api_key(self) -> Optional[str]:
        provider = self.llm_provider
        key_map = {
            LLMProvider.CLAUDE: "ANTHROPIC_API_KEY",
            LLMProvider.GPT: "OPENAI_API_KEY",
            LLMProvider.GEMINI: "GOOGLE_API_KEY"
        }
        return os.getenv(key_map.get(provider, ""))
```

### 1.3 LLM Interface

**File: `agent/llm/interface.py`**

```python
"""Abstract interface for LLM providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Optional[Any] = None

class LLMInterface(ABC):
    def __init__(self, api_key: str, model: str, temperature: float = 0.2,
                 max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        pass
```

### 1.4 Claude Adapter

**File: `agent/llm/claude_adapter.py`**

```python
"""Claude (Anthropic) LLM adapter."""
import anthropic
from typing import List
from agent.llm.interface import LLMInterface, Message, LLMResponse

class ClaudeAdapter(LLMInterface):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.2, max_tokens: int = 4096):
        super().__init__(api_key, model, temperature, max_tokens)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages: List[Message], **kwargs) -> LLMResponse:
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system_message,
            messages=conversation
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens},
            finish_reason=response.stop_reason,
            raw_response=response
        )

    def validate_api_key(self) -> bool:
        try:
            self.client.messages.create(
                model=self.model, max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except:
            return False
```

### 1.5 Structured Logging

**File: `agent/io/logger.py`**

```python
"""Structured logging for LRET AI Agent."""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

class StructuredLogger:
    def __init__(self, name: str, log_dir: Optional[Path] = None,
                 log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.session_id = str(uuid.uuid4())

        self.log_dir = log_dir or Path.home() / ".lret" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        log_file = self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

    def _format_log(self, level: str, message: str,
                    extra: Optional[Dict[str, Any]] = None) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level, "session_id": self.session_id,
            "logger": self.name, "message": message
        }
        if extra:
            log_entry["extra"] = extra
        return json.dumps(log_entry)

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_log("INFO", message, kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_log("ERROR", message, kwargs))
```

### 1.6 CLI Parser

**File: `agent/cli/parser.py`**

```python
"""Command-line argument parser for LRET AI Agent."""
import argparse
from typing import List, Optional
from agent.utils.constants import Backend, LLMProvider

class CLIParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="LRET AI Agent - Intelligent quantum simulation assistant"
        )
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument('query', type=str, nargs='?',
                                 help='Natural language query')
        self.parser.add_argument('--llm', type=str,
                                 choices=[p.value for p in LLMProvider])
        self.parser.add_argument('--backend', type=str,
                                 choices=[b.value for b in Backend])
        self.parser.add_argument('--dry-run', action='store_true')
        self.parser.add_argument('--interactive', '-i', action='store_true')
        self.parser.add_argument('--verbose', '-v', action='store_true')

    def parse(self, args: Optional[List[str]] = None):
        return self.parser.parse_args(args)
```

---

## 📚 PHASE 2: Intent Understanding & Planning

### Overview

Implement natural language understanding to parse user queries, classify intent, extract parameters, and generate execution plans.

### 2.1 Intent Classifier

**File: `agent/intent/classifier.py`**

```python
"""Intent classification for user queries."""
from dataclasses import dataclass
from typing import List, Dict, Any
from agent.utils.constants import IntentType
from agent.llm.interface import LLMInterface, Message

@dataclass
class Intent:
    type: IntentType
    confidence: float
    parameters: Dict[str, Any]
    raw_query: str

class IntentClassifier:
    INTENT_EXAMPLES = {
        IntentType.RUN_EXPERIMENT: [
            "run experiment with 10 qubits",
            "simulate 12 qubit circuit with depth 20",
            "run 10 qubits with 1% noise in hybrid mode",
            "run on Cirq with 1000 shots",
        ],
        IntentType.RUN_BENCHMARK: [
            "benchmark qubits 8,10,12 with depths 20,30,40",
            "run performance test with 5 trials",
        ],
        IntentType.COMPARE_EXPERIMENTS: [
            "compare run_001 and run_002",
            "compare LRET vs Cirq results",
        ],
        IntentType.EXPLAIN_RESULTS: [
            "explain the results",
            "what do these metrics mean",
        ],
    }

    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def classify(self, query: str) -> Intent:
        messages = self._build_classification_prompt(query)
        response = self.llm.generate(messages, temperature=0.1)

        import json
        try:
            result = json.loads(response.content)
            intent_type = IntentType(result["intent"].lower())
            confidence = float(result["confidence"])
            parameters = result.get("parameters", {})
        except (json.JSONDecodeError, KeyError, ValueError):
            intent_type = IntentType.UNKNOWN
            confidence = 0.0
            parameters = {}

        return Intent(type=intent_type, confidence=confidence,
                      parameters=parameters, raw_query=query)

    def _build_classification_prompt(self, query: str) -> List[Message]:
        system_prompt = """You are an intent classifier for a quantum simulation AI agent.

Classify the query into one of these intents:
- RUN_EXPERIMENT: Run a quantum simulation
- RUN_BENCHMARK: Run performance benchmarks
- COMPARE_EXPERIMENTS: Compare experiments
- EXPLAIN_RESULTS: Explain results or metrics
- INSPECT_CODE: Read/understand code
- PROPOSE_CHANGES: Modify or improve code
- UNKNOWN: Cannot determine intent

Extract parameters like n_qubits, depth, noise_level, mode, backend, etc.

Respond in JSON: {"intent": "INTENT_TYPE", "confidence": 0.0-1.0, "parameters": {...}}"""

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Classify: {query}")
        ]
```

### 2.2 Plan Generator

**File: `agent/planning/planner.py`**

```python
"""Execution plan generation."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from agent.intent.classifier import Intent
from agent.utils.constants import ActionCategory, IntentType
from agent.llm.interface import LLMInterface, Message

@dataclass
class PlanStep:
    id: int
    action: str
    category: ActionCategory
    description: str
    parameters: Dict[str, Any]
    dependencies: List[int]
    backend: Optional[str] = None

@dataclass
class ExecutionPlan:
    steps: List[PlanStep]
    estimated_duration: Optional[str]
    requires_approval: bool
    risk_level: str
    backend: Optional[str] = None

class Planner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def generate_plan(self, intent: Intent) -> ExecutionPlan:
        messages = self._build_planning_prompt(intent)
        response = self.llm.generate(messages, temperature=0.2)

        import json
        try:
            result = json.loads(response.content)
            steps = [
                PlanStep(
                    id=s["id"], action=s["action"],
                    category=ActionCategory(s["category"]),
                    description=s["description"],
                    parameters=s.get("parameters", {}),
                    dependencies=s.get("dependencies", []),
                    backend=s.get("backend")
                )
                for s in result["steps"]
            ]
            requires_approval = any(s.category == ActionCategory.WRITE for s in steps)

            return ExecutionPlan(
                steps=steps,
                estimated_duration=result.get("estimated_duration"),
                requires_approval=requires_approval,
                risk_level=result.get("risk_level", "medium"),
                backend=result.get("backend_used")
            )
        except (json.JSONDecodeError, KeyError):
            return ExecutionPlan(steps=[], estimated_duration=None,
                                requires_approval=False, risk_level="unknown")

    def _build_planning_prompt(self, intent: Intent) -> List[Message]:
        system_prompt = """You are a plan generator for a quantum simulation AI agent.

Given a user intent, create a step-by-step execution plan.

Available actions:
- READ category: read_file, search_code, explain_code, read_results
- RUN category: build_project, run_experiment, run_benchmark, run_cirq_simulation
- WRITE category: propose_changes, apply_changes

Respond in JSON:
{
  "steps": [{"id": 1, "action": "...", "category": "read|run|write",
             "description": "...", "parameters": {}, "dependencies": []}],
  "estimated_duration": "X minutes",
  "risk_level": "low|medium|high"
}"""

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Intent: {intent.type.value}\n"
                                         f"Parameters: {intent.parameters}\n"
                                         f"Query: {intent.raw_query}")
        ]

    def validate_plan(self, plan: ExecutionPlan) -> tuple[bool, Optional[str]]:
        if not plan.steps:
            return False, "Plan has no steps"

        visited = set()
        for step in plan.steps:
            if step.id in visited:
                return False, f"Duplicate step ID: {step.id}"
            visited.add(step.id)

        return True, None
```

---

## 📚 PHASE 3: Experiment Execution

### Overview

Implement experiment execution with proper result handling, timeout management, and output parsing.

### 3.1 Experiment Configuration

**File: `agent/execution/runner.py`**

```python
"""Experiment execution for LRET simulations."""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import subprocess
import uuid

LRET_DOCKER_IMAGE = "ajs911/lret777:latest"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    n_qubits: int
    depth: int
    backend: str = "auto"
    noise_level: float = 0.001
    noise_type: str = "depolarizing"
    mode: str = "hybrid"
    rank_threshold: float = 1e-4
    threads: Optional[int] = None
    output_file: Optional[str] = None
    output_format: str = "json"
    compare_to_exact: bool = False
    use_gpu: bool = False
    timeout_seconds: int = 3600
    use_docker: bool = False

@dataclass
class ExperimentResult:
    """Canonical result schema for LRET experiments."""
    run_id: str
    config: ExperimentConfig
    status: str  # "success", "failed", "timeout"

    # Metrics
    simulation_time_seconds: Optional[float] = None
    final_rank: Optional[int] = None
    fidelity: Optional[float] = None
    speedup: Optional[float] = None
    memory_used_mb: Optional[float] = None

    # Raw output
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None

    # Metadata
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class ExperimentRunner:
    def __init__(self, repo_root: Path, build_dir: str = "build"):
        self.repo_root = repo_root
        self.build_dir = repo_root / build_dir
        self.current_process = None

    def build_project(self) -> tuple[bool, str]:
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)

            cmake_result = subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                cwd=self.build_dir, capture_output=True, text=True, timeout=120
            )
            if cmake_result.returncode != 0:
                return False, f"CMake failed: {cmake_result.stderr}"

            make_result = subprocess.run(
                ["cmake", "--build", ".", "-j"],
                cwd=self.build_dir, capture_output=True, text=True, timeout=600
            )
            if make_result.returncode != 0:
                return False, f"Build failed: {make_result.stderr}"

            return True, "Build successful"
        except Exception as e:
            return False, str(e)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.utcnow().isoformat()

        cmd = self._build_command(config)

        try:
            self.current_process = subprocess.Popen(
                cmd, cwd=self.repo_root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = self.current_process.communicate(timeout=config.timeout_seconds)
            exit_code = self.current_process.returncode

            result = self._parse_output(stdout, stderr, exit_code)
            result.run_id = run_id
            result.config = config
            result.started_at = started_at
            result.completed_at = datetime.utcnow().isoformat()
            result.status = "success" if exit_code == 0 else "failed"

            return result

        except subprocess.TimeoutExpired:
            self.current_process.kill()
            return ExperimentResult(
                run_id=run_id, config=config, status="timeout",
                started_at=started_at, completed_at=datetime.utcnow().isoformat()
            )

    def _build_command(self, config: ExperimentConfig) -> List[str]:
        executable = self.build_dir / "quantum_sim"
        cmd = [
            str(executable),
            "-n", str(config.n_qubits),
            "-d", str(config.depth),
            "--mode", config.mode,
            "--noise", str(config.noise_level),
            "--threshold", str(config.rank_threshold),
            "--format", config.output_format,
        ]

        if config.backend != "auto":
            cmd.extend(["--backend", config.backend])
        if config.compare_to_exact:
            cmd.append("--compare-fdm")
        if config.use_gpu:
            cmd.append("--gpu")
        if config.output_file:
            cmd.extend(["-o", config.output_file])

        return cmd

    def _parse_output(self, stdout: str, stderr: str, exit_code: int) -> ExperimentResult:
        import re
        result = ExperimentResult(run_id="", config=None, status="unknown")

        patterns = {
            "simulation_time_seconds": r"(?:Simulation Time|time):\s*([\d.]+)",
            "final_rank": r"(?:Final Rank|final_rank):\s*(\d+)",
            "fidelity": r"[Ff]idelity:\s*([\d.]+)",
            "memory_used_mb": r"[Mm]emory.*?:\s*([\d.]+)\s*MB",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                value = match.group(1)
                if field == "final_rank":
                    setattr(result, field, int(value))
                else:
                    setattr(result, field, float(value))

        result.stdout = stdout
        result.stderr = stderr
        result.exit_code = exit_code
        return result
```

---

## 📚 PHASE 4: Backend Management

### Overview

Implement intelligent backend routing between LRET and Cirq based on query analysis.

### 4.1 Backend Router

**File: `agent/utils/backend_router.py`**

```python
"""Intelligent routing between LRET and Cirq backends."""
from typing import Dict, Any, Optional
from agent.utils.constants import LRET_KEYWORDS, CIRQ_KEYWORDS, BACKEND_CAPABILITIES

class BackendRouter:
    """Routes queries to appropriate backend based on content analysis."""

    def __init__(self):
        self.lret_keywords = LRET_KEYWORDS
        self.cirq_keywords = CIRQ_KEYWORDS
        self.capabilities = BACKEND_CAPABILITIES

    def detect_backend(self, query: str, parameters: Dict[str, Any]) -> str:
        query_lower = query.lower()

        # Priority 1: Explicit backend specification
        explicit = self._check_explicit_backend(query_lower)
        if explicit:
            return explicit

        # Priority 2: Comparison request
        if self._is_comparison_request(query_lower):
            return "both"

        # Priority 3: Hardware keywords → Cirq
        if self._has_hardware_keywords(query_lower):
            return "cirq"

        # Priority 4: LRET-specific keywords
        if self._has_lret_keywords(query_lower):
            return "lret"

        # Priority 5: Qubit count heuristics
        n_qubits = parameters.get("n_qubits", 0)
        if n_qubits > 18:
            return "lret"  # LRET handles large systems better

        return "lret"  # Default

    def _check_explicit_backend(self, query: str) -> Optional[str]:
        cirq_patterns = ["on cirq", "using cirq", "with cirq", "cirq backend"]
        lret_patterns = ["on lret", "using lret", "with lret", "lret backend"]

        if any(p in query for p in cirq_patterns):
            return "cirq"
        if any(p in query for p in lret_patterns):
            return "lret"
        return None

    def _is_comparison_request(self, query: str) -> bool:
        patterns = ["compare lret", "lret vs cirq", "both backends", "compare backends"]
        return any(p in query for p in patterns)

    def _has_hardware_keywords(self, query: str) -> bool:
        keywords = ["sycamore", "weber", "rainbow", "real device", "hardware"]
        return any(kw in query for kw in keywords)

    def _has_lret_keywords(self, query: str) -> bool:
        return any(kw in query for kw in self.lret_keywords)

    def get_backend_recommendation(self, query: str, n_qubits: int = 10,
                                   circuit_type: str = None,
                                   **kwargs) -> Dict[str, Any]:
        if kwargs.get("require_hardware"):
            return {"backend": "cirq", "reason": "Hardware execution requires Cirq"}

        if circuit_type in ["vqe", "qaoa"]:
            return {"backend": "cirq",
                    "reason": f"{circuit_type.upper()} has optimized Cirq implementation"}

        if n_qubits > 15:
            return {"backend": "lret",
                    "reason": f"LRET's rank truncation is efficient for {n_qubits} qubits"}

        return {"backend": "lret", "reason": "Default backend with rank tracking"}
```

### 4.2 Result Validation

**File: `agent/validation/validator.py`**

```python
"""Result validation and schema enforcement."""
from typing import Dict, Any, List, Optional
from agent.execution.runner import ExperimentResult

class ResultValidator:
    """Validate experiment results against schema."""

    REQUIRED_FIELDS = ["run_id", "status"]
    REQUIRED_FOR_SUCCESS = ["simulation_time_seconds"]

    def validate(self, result: ExperimentResult) -> tuple[bool, List[str]]:
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if not getattr(result, field, None):
                errors.append(f"Missing required field: {field}")

        # For successful runs, check additional fields
        if result.status == "success":
            for field in self.REQUIRED_FOR_SUCCESS:
                if getattr(result, field, None) is None:
                    errors.append(f"Missing field for successful run: {field}")

        # Validate ranges
        if result.fidelity is not None:
            if not 0 <= result.fidelity <= 1:
                errors.append(f"Fidelity out of range: {result.fidelity}")

        if result.simulation_time_seconds is not None:
            if result.simulation_time_seconds < 0:
                errors.append(f"Negative simulation time: {result.simulation_time_seconds}")

        return len(errors) == 0, errors

    def get_validation_summary(self, result: ExperimentResult) -> Dict[str, Any]:
        is_valid, errors = self.validate(result)

        return {
            "is_valid": is_valid,
            "errors": errors,
            "metrics_present": {
                "fidelity": result.fidelity is not None,
                "simulation_time": result.simulation_time_seconds is not None,
                "final_rank": result.final_rank is not None,
                "memory": result.memory_used_mb is not None,
            },
            "status": result.status
        }
```

---

## Phase 1-4 Summary

**Completed Features:**

- ✅ Configuration management system
- ✅ Multi-LLM abstraction (Claude, GPT, Gemini)
- ✅ Structured logging infrastructure
- ✅ Command-line interface
- ✅ Intent classification system
- ✅ Plan generation and validation
- ✅ Experiment execution with timeout handling
- ✅ Backend routing between LRET and Cirq
- ✅ Result validation and schema enforcement

---

_Continue to [agent3.md](agent3.md) for Phases 5-10: Backend execution, safety, memory, and testing._
