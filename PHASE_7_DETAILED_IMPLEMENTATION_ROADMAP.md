# Phase 7: Detailed Implementation Roadmap - All Tiers

**Document Purpose:** Step-by-step implementation guide for all ecosystem integrations  
**Created:** January 14, 2026  
**Branch:** phase-7  
**Reference:** PHASE_7_COMPREHENSIVE_INTEGRATION_ANALYSIS.md

---

## Document Structure

Each integration follows this template:
1. **Overview** - Strategic rationale and scope
2. **Prerequisites** - Dependencies and setup
3. **Architecture** - System design and interfaces
4. **Implementation Steps** - Day-by-day breakdown
5. **Testing Strategy** - Validation approach
6. **Documentation** - User guides and examples
7. **Success Metrics** - KPIs and validation criteria

---

# TIER 1: CRITICAL INTEGRATIONS

---

## 7.1.1: Qiskit (IBM Quantum) Integration

**Duration:** 5-7 days  
**Priority:** 🔴 CRITICAL  
**Complexity:** ⭐⭐⭐ Medium  
**Expected Users:** 100,000+

### **Overview**

Integrate LRET as a Qiskit Backend (BackendV2) to enable 500,000+ monthly Qiskit users to leverage LRET's low-rank simulation for noisy quantum circuits. This is the highest-impact integration due to Qiskit's dominant market position.

**Strategic Value:**
- Access to 60% of quantum software market
- Enterprise customers (IBM, JP Morgan, Daimler)
- Academic citations from Qiskit papers
- Gateway to IBM Quantum ecosystem

---

### **Prerequisites**

**Software Dependencies:**
```bash
# Required packages
pip install qiskit>=1.0.0
pip install qiskit-aer>=0.13.0
pip install qiskit-ibm-runtime>=0.15.0

# For noise models
pip install qiskit-experiments>=0.5.0

# LRET prerequisites
# Already in LRET: Eigen3, nlohmann/json, pybind11
```

**Knowledge Requirements:**
- Qiskit BackendV2 API specification
- OpenQASM 3.0 format
- Qiskit Result/Job API
- Noise model JSON structure (Qiskit Aer format)

**Reference Documentation:**
- https://qiskit.org/documentation/apidoc/providers.html
- https://qiskit.org/documentation/apidoc/providers_backend.html
- https://github.com/Qiskit/qiskit-aer (noise models)

---

### **Architecture Design**

```
┌─────────────────────────────────────────────────────────────┐
│                    Qiskit User Code                          │
│  from qiskit import QuantumCircuit                          │
│  from lret_qiskit import LRETProvider                       │
│                                                              │
│  provider = LRETProvider()                                  │
│  backend = provider.get_backend('lret_simulator')          │
│  job = backend.run(circuit, shots=1000)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LRETProvider (Python Layer)                     │
│  • Backend discovery                                        │
│  • Backend instantiation                                    │
│  • Configuration management                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LRETBackend (BackendV2)                        │
│  • Target definition (supported gates)                      │
│  • Options management (shots, epsilon, noise_model)         │
│  • run() method → creates LRETJob                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LRETJob (Job API)                              │
│  • Circuit translation (Qiskit → LRET)                     │
│  • Execution management                                     │
│  • Result formatting (LRET → Qiskit Result)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Circuit Translator                              │
│  • QuantumCircuit → LRET QuantumSequence                   │
│  • Gate mapping (30+ gate types)                           │
│  • Parameter binding                                        │
│  • Conditional operations                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LRET C++ Core (Existing)                       │
│  • Low-rank simulation                                      │
│  • Noise application                                        │
│  • Measurement sampling                                     │
└─────────────────────────────────────────────────────────────┘
```

---

### **Implementation Steps**

#### **Day 1: Project Setup & Basic Structure**

**Morning (4 hours): Repository Setup**

```bash
# Create directory structure
cd python
mkdir -p lret_qiskit/{backends,translators,tests}
touch lret_qiskit/__init__.py
touch lret_qiskit/backends/__init__.py
touch lret_qiskit/translators/__init__.py
touch lret_qiskit/tests/__init__.py
```

File structure:
```
python/lret_qiskit/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── lret_backend.py      # BackendV2 implementation
│   └── lret_job.py           # Job implementation
├── translators/
│   ├── __init__.py
│   ├── circuit_translator.py  # Qiskit → LRET
│   ├── gate_mapper.py         # Gate conversion logic
│   └── result_converter.py    # LRET → Qiskit Result
├── provider.py                # LRETProvider
├── version.py                 # Version info
└── tests/
    ├── test_backend.py
    ├── test_translator.py
    └── test_integration.py
```

**Afternoon (4 hours): Provider Implementation**

Create `python/lret_qiskit/provider.py`:

```python
"""
LRET Provider for Qiskit
========================

Enables Qiskit users to access LRET simulator backends.
"""

from qiskit.providers import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from .backends.lret_backend import LRETBackend
from .version import __version__

class LRETProvider(ProviderV1):
    """
    Provider for LRET quantum simulators.
    
    Usage:
        >>> from lret_qiskit import LRETProvider
        >>> provider = LRETProvider()
        >>> backend = provider.get_backend('lret_simulator')
        >>> job = backend.run(circuit, shots=1000)
    """
    
    def __init__(self):
        super().__init__()
        self._backends = self._initialize_backends()
    
    def _initialize_backends(self):
        """Create available LRET backends."""
        return {
            'lret_simulator': LRETBackend(
                name='lret_simulator',
                description='LRET Low-Rank Quantum Simulator',
                epsilon=1e-4
            ),
            'lret_simulator_accurate': LRETBackend(
                name='lret_simulator_accurate',
                description='LRET Simulator (High Accuracy)',
                epsilon=1e-6
            ),
            'lret_simulator_fast': LRETBackend(
                name='lret_simulator_fast',
                description='LRET Simulator (Fast Mode)',
                epsilon=1e-3
            ),
        }
    
    def backends(self, name=None, filters=None, **kwargs):
        """
        Return a list of backends matching the specified filtering.
        
        Args:
            name (str): Backend name to filter on
            filters (callable): Lambda for custom filtering
            
        Returns:
            list: List of Backend instances
        """
        backends = list(self._backends.values())
        
        if name:
            backends = [b for b in backends if b.name() == name]
        
        if filters:
            backends = [b for b in backends if filters(b)]
        
        return backends
    
    def get_backend(self, name=None, **kwargs):
        """
        Return a single backend matching the specified name.
        
        Args:
            name (str): Name of the backend
            
        Returns:
            Backend: The backend instance
            
        Raises:
            QiskitBackendNotFoundError: If backend not found
        """
        if name is None:
            name = 'lret_simulator'
        
        try:
            return self._backends[name]
        except KeyError:
            raise QiskitBackendNotFoundError(
                f"Backend '{name}' not found. "
                f"Available backends: {list(self._backends.keys())}"
            )
    
    def __str__(self):
        return f"<LRETProvider(version={__version__})>"
```

Create `python/lret_qiskit/version.py`:

```python
"""Version information for lret_qiskit package."""

__version__ = '1.0.0'
__author__ = 'LRET Development Team'
__license__ = 'Apache 2.0'
```

**Deliverables Day 1:**
- ✅ Directory structure created
- ✅ Provider class implemented
- ✅ Multiple backend variants (fast/accurate)
- ✅ Version management

---

#### **Day 2: Backend Implementation (Core Interface)**

**Morning (4 hours): BackendV2 Base Class**

Create `python/lret_qiskit/backends/lret_backend.py`:

```python
"""
LRET Backend for Qiskit
=======================

Implements BackendV2 interface for LRET simulator.
"""

from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit import Parameter, Measure
from qiskit.circuit.library import *
import numpy as np

class LRETBackend(BackendV2):
    """
    LRET simulator backend for Qiskit.
    
    This backend provides a low-rank quantum simulation engine
    optimized for noisy intermediate-scale quantum (NISQ) circuits.
    
    Attributes:
        epsilon (float): Truncation threshold for low-rank approximation
        max_qubits (int): Maximum number of qubits supported
        noise_model: Optional Qiskit noise model
    """
    
    def __init__(self, name='lret_simulator', description='', epsilon=1e-4):
        """
        Initialize LRET backend.
        
        Args:
            name (str): Backend name
            description (str): Backend description
            epsilon (float): SVD truncation threshold
        """
        super().__init__(
            name=name,
            description=description,
            backend_version='1.0.0'
        )
        
        self._epsilon = epsilon
        self._max_qubits = 28  # LRET theoretical limit
        self._target = self._build_target()
        self._options = self._default_options()
    
    @property
    def target(self):
        """Return the Target object for this backend."""
        return self._target
    
    @property
    def max_circuits(self):
        """Maximum number of circuits to run in single job."""
        return 300
    
    @classmethod
    def _default_options(cls):
        """Return default options."""
        return Options(
            shots=1024,
            epsilon=1e-4,
            seed_simulator=None,
            noise_model=None,
            initial_rank=1,
            max_rank=None,
            parallelization='auto'  # 'row', 'column', 'hybrid', 'auto'
        )
    
    def _build_target(self):
        """
        Build the Target object defining supported operations.
        
        Returns:
            Target: Qiskit Target with gate specifications
        """
        target = Target(
            description=f"LRET simulator target",
            num_qubits=self._max_qubits
        )
        
        # Single-qubit gates (apply to any qubit)
        single_qubit_gates = [
            (IGate(), None),
            (XGate(), None),
            (YGate(), None),
            (ZGate(), None),
            (HGate(), None),
            (SGate(), None),
            (SdgGate(), None),
            (TGate(), None),
            (TdgGate(), None),
            (SXGate(), None),
            (SXdgGate(), None),
        ]
        
        for gate, props in single_qubit_gates:
            for qubit in range(self._max_qubits):
                target.add_instruction(
                    gate,
                    {(qubit,): props}
                )
        
        # Parameterized single-qubit gates
        param_single_gates = [
            RXGate(Parameter('θ')),
            RYGate(Parameter('θ')),
            RZGate(Parameter('θ')),
            PhaseGate(Parameter('θ')),
            U1Gate(Parameter('λ')),
            U2Gate(Parameter('φ'), Parameter('λ')),
            U3Gate(Parameter('θ'), Parameter('φ'), Parameter('λ')),
        ]
        
        for gate in param_single_gates:
            for qubit in range(self._max_qubits):
                target.add_instruction(
                    gate,
                    {(qubit,): None}
                )
        
        # Two-qubit gates (fully connected)
        two_qubit_gates = [
            CXGate(),
            CYGate(),
            CZGate(),
            CHGate(),
            SwapGate(),
            iSwapGate(),
            ECRGate(),
        ]
        
        for gate in two_qubit_gates:
            for q1 in range(self._max_qubits):
                for q2 in range(self._max_qubits):
                    if q1 != q2:
                        target.add_instruction(
                            gate,
                            {(q1, q2): None}
                        )
        
        # Parameterized two-qubit gates
        param_two_gates = [
            CRXGate(Parameter('θ')),
            CRYGate(Parameter('θ')),
            CRZGate(Parameter('θ')),
            RXXGate(Parameter('θ')),
            RYYGate(Parameter('θ')),
            RZZGate(Parameter('θ')),
        ]
        
        for gate in param_two_gates:
            for q1 in range(self._max_qubits):
                for q2 in range(self._max_qubits):
                    if q1 != q2:
                        target.add_instruction(
                            gate,
                            {(q1, q2): None}
                        )
        
        # Measurement
        for qubit in range(self._max_qubits):
            target.add_instruction(
                Measure(),
                {(qubit,): None}
            )
        
        return target
    
    @property
    def options(self):
        """Return the options for this backend."""
        return self._options
    
    def run(self, run_input, **options):
        """
        Run circuits on the LRET backend.
        
        Args:
            run_input: QuantumCircuit or list of QuantumCircuit
            **options: Backend options (shots, epsilon, etc.)
            
        Returns:
            LRETJob: Job object for tracking execution
        """
        from .lret_job import LRETJob
        
        # Handle single circuit or list
        if not isinstance(run_input, list):
            run_input = [run_input]
        
        # Validate number of circuits
        if len(run_input) > self.max_circuits:
            raise ValueError(
                f"Number of circuits ({len(run_input)}) exceeds "
                f"max_circuits ({self.max_circuits})"
            )
        
        # Merge options
        job_options = self._options.copy()
        job_options.update(**options)
        
        # Create and submit job
        job = LRETJob(
            backend=self,
            job_id=self._generate_job_id(),
            circuits=run_input,
            options=job_options
        )
        
        job._submit()
        return job
    
    def _generate_job_id(self):
        """Generate unique job ID."""
        import uuid
        return f"lret-{uuid.uuid4().hex[:16]}"
```

**Afternoon (4 hours): Job Implementation**

Create `python/lret_qiskit/backends/lret_job.py`:

```python
"""
LRET Job Implementation
=======================

Manages circuit execution and result formatting.
"""

from qiskit.providers import JobV1, JobStatus
from qiskit.result import Result
from datetime import datetime
import uuid

class LRETJob(JobV1):
    """
    LRET job for managing circuit execution.
    
    Attributes:
        backend: LRETBackend instance
        circuits: List of QuantumCircuits to execute
        options: Execution options
    """
    
    def __init__(self, backend, job_id, circuits, options):
        """
        Initialize LRET job.
        
        Args:
            backend: LRETBackend instance
            job_id (str): Unique job identifier
            circuits (list): List of QuantumCircuits
            options (Options): Execution options
        """
        super().__init__(backend, job_id)
        self._circuits = circuits
        self._options = options
        self._result = None
        self._status = JobStatus.INITIALIZING
    
    def submit(self):
        """Submit job for execution."""
        self._submit()
    
    def _submit(self):
        """Internal method to run simulation."""
        self._status = JobStatus.RUNNING
        
        try:
            # Import translator
            from ..translators.circuit_translator import CircuitTranslator
            from ..translators.result_converter import ResultConverter
            
            translator = CircuitTranslator()
            converter = ResultConverter()
            
            # Execute each circuit
            experiment_results = []
            
            for circuit in self._circuits:
                # Translate circuit
                lret_circuit = translator.translate(circuit)
                
                # Run LRET simulation
                lret_result = self._run_lret_simulation(
                    lret_circuit,
                    circuit.num_qubits,
                    self._options
                )
                
                # Convert to Qiskit format
                exp_result = converter.convert_experiment(
                    lret_result,
                    circuit
                )
                experiment_results.append(exp_result)
            
            # Create Qiskit Result object
            self._result = Result(
                backend_name=self.backend().name(),
                backend_version=self.backend().backend_version,
                qobj_id=None,
                job_id=self.job_id(),
                success=True,
                results=experiment_results,
                date=datetime.now().isoformat(),
                status='COMPLETED',
                header={}
            )
            
            self._status = JobStatus.DONE
            
        except Exception as e:
            self._status = JobStatus.ERROR
            self._result = Result(
                backend_name=self.backend().name(),
                backend_version=self.backend().backend_version,
                qobj_id=None,
                job_id=self.job_id(),
                success=False,
                results=[],
                date=datetime.now().isoformat(),
                status='ERROR',
                header={'error': str(e)}
            )
            raise
    
    def _run_lret_simulation(self, lret_circuit, num_qubits, options):
        """
        Execute LRET simulation.
        
        Args:
            lret_circuit: Translated circuit for LRET
            num_qubits (int): Number of qubits
            options: Execution options
            
        Returns:
            dict: LRET simulation result
        """
        # Import LRET Python bindings
        try:
            from qlret import LRETSimulator
        except ImportError:
            raise ImportError(
                "LRET Python bindings not found. "
                "Please install: pip install qlret"
            )
        
        # Create simulator
        sim = LRETSimulator(
            n_qubits=num_qubits,
            epsilon=options.epsilon,
            initial_rank=options.initial_rank,
            seed=options.seed_simulator
        )
        
        # Apply noise model if specified
        if options.noise_model is not None:
            sim.set_noise_model(options.noise_model)
        
        # Run simulation
        result = sim.run(
            circuit=lret_circuit,
            shots=options.shots,
            parallelization=options.parallelization
        )
        
        return result
    
    def result(self):
        """
        Return the result object for this job.
        
        Returns:
            Result: Qiskit Result object
        """
        # Wait for completion if still running
        if self._status == JobStatus.RUNNING:
            self._wait_for_completion()
        
        return self._result
    
    def status(self):
        """Return current job status."""
        return self._status
    
    def _wait_for_completion(self):
        """Wait for job to complete."""
        import time
        while self._status == JobStatus.RUNNING:
            time.sleep(0.1)
```

**Deliverables Day 2:**
- ✅ BackendV2 implementation with Target
- ✅ Support for 50+ gate types
- ✅ Job management with status tracking
- ✅ Integration with LRET Python bindings

---

#### **Day 3: Circuit Translation Layer**

**Morning (4 hours): Gate Mapper**

Create `python/lret_qiskit/translators/gate_mapper.py`:

```python
"""
Gate Mapper: Qiskit → LRET
==========================

Maps Qiskit gate operations to LRET gate format.
"""

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.library import *

class GateMapper:
    """
    Maps Qiskit gates to LRET format.
    
    LRET expects gates in format:
    {
        'name': str,
        'qubits': list[int],
        'params': list[float],
        'matrix': Optional[np.ndarray]  # For custom gates
    }
    """
    
    def __init__(self):
        """Initialize gate mapper with lookup tables."""
        self._single_qubit_map = self._build_single_qubit_map()
        self._two_qubit_map = self._build_two_qubit_map()
    
    def _build_single_qubit_map(self):
        """Build mapping for single-qubit gates."""
        return {
            'id': ('I', []),
            'x': ('X', []),
            'y': ('Y', []),
            'z': ('Z', []),
            'h': ('H', []),
            's': ('S', []),
            'sdg': ('Sdg', []),
            't': ('T', []),
            'tdg': ('Tdg', []),
            'sx': ('SX', []),
            'sxdg': ('SXdg', []),
            'rx': ('RX', ['theta']),
            'ry': ('RY', ['theta']),
            'rz': ('RZ', ['theta']),
            'p': ('Phase', ['theta']),
            'u1': ('U1', ['lambda']),
            'u2': ('U2', ['phi', 'lambda']),
            'u3': ('U3', ['theta', 'phi', 'lambda']),
        }
    
    def _build_two_qubit_map(self):
        """Build mapping for two-qubit gates."""
        return {
            'cx': ('CNOT', []),
            'cy': ('CY', []),
            'cz': ('CZ', []),
            'ch': ('CH', []),
            'swap': ('SWAP', []),
            'iswap': ('iSWAP', []),
            'crx': ('CRX', ['theta']),
            'cry': ('CRY', ['theta']),
            'crz': ('CRZ', ['theta']),
            'rxx': ('RXX', ['theta']),
            'ryy': ('RYY', ['theta']),
            'rzz': ('RZZ', ['theta']),
            'ecr': ('ECR', []),
        }
    
    def map_gate(self, instruction):
        """
        Map Qiskit instruction to LRET gate.
        
        Args:
            instruction: Qiskit CircuitInstruction
            
        Returns:
            dict: LRET gate specification
        """
        gate = instruction.operation
        gate_name = gate.name.lower()
        qubits = [q.index for q in instruction.qubits]
        
        # Check for measurement
        if gate_name == 'measure':
            return {
                'type': 'measurement',
                'qubits': qubits,
                'classical_bits': [c.index for c in instruction.clbits]
            }
        
        # Single-qubit gates
        if len(qubits) == 1:
            if gate_name in self._single_qubit_map:
                lret_name, param_names = self._single_qubit_map[gate_name]
                params = self._extract_params(gate, param_names)
                
                return {
                    'type': 'gate',
                    'name': lret_name,
                    'qubits': qubits,
                    'params': params
                }
        
        # Two-qubit gates
        elif len(qubits) == 2:
            if gate_name in self._two_qubit_map:
                lret_name, param_names = self._two_qubit_map[gate_name]
                params = self._extract_params(gate, param_names)
                
                return {
                    'type': 'gate',
                    'name': lret_name,
                    'qubits': qubits,
                    'params': params
                }
        
        # Custom/unsupported gate - use matrix
        if hasattr(gate, 'to_matrix'):
            return {
                'type': 'gate',
                'name': 'Custom',
                'qubits': qubits,
                'params': [],
                'matrix': gate.to_matrix().tolist()
            }
        
        raise ValueError(f"Unsupported gate: {gate_name}")
    
    def _extract_params(self, gate, param_names):
        """Extract parameter values from gate."""
        params = []
        for name in param_names:
            # Map parameter name to gate attribute
            attr_map = {
                'theta': 'theta',
                'phi': 'phi',
                'lambda': 'lam',
            }
            attr_name = attr_map.get(name, name)
            
            if hasattr(gate, attr_name):
                value = getattr(gate, attr_name)
                # Handle Parameter objects (unbound parameters)
                if hasattr(value, 'numeric'):
                    value = float(value.numeric())
                params.append(float(value))
            elif hasattr(gate, 'params') and gate.params:
                params.append(float(gate.params[len(params)]))
        
        return params
```

**Afternoon (4 hours): Circuit Translator**

Create `python/lret_qiskit/translators/circuit_translator.py`:

```python
"""
Circuit Translator: Qiskit → LRET
==================================

Translates Qiskit QuantumCircuit to LRET circuit format.
"""

from .gate_mapper import GateMapper
import warnings

class CircuitTranslator:
    """
    Translates Qiskit circuits to LRET format.
    
    Handles:
    - Gate translation
    - Parameter binding
    - Conditional operations (if supported)
    - Measurement extraction
    """
    
    def __init__(self):
        """Initialize translator."""
        self.gate_mapper = GateMapper()
    
    def translate(self, circuit):
        """
        Translate Qiskit QuantumCircuit to LRET format.
        
        Args:
            circuit: Qiskit QuantumCircuit
            
        Returns:
            dict: LRET circuit specification
        """
        # Check for unsupported features
        self._validate_circuit(circuit)
        
        # Initialize LRET circuit
        lret_circuit = {
            'num_qubits': circuit.num_qubits,
            'num_clbits': circuit.num_clbits,
            'operations': [],
            'measurements': []
        }
        
        # Translate each instruction
        for instruction in circuit.data:
            lret_op = self.gate_mapper.map_gate(instruction)
            
            if lret_op['type'] == 'measurement':
                lret_circuit['measurements'].append(lret_op)
            else:
                lret_circuit['operations'].append(lret_op)
        
        return lret_circuit
    
    def _validate_circuit(self, circuit):
        """Check for unsupported features."""
        # Check for conditionals
        has_conditionals = any(
            inst.operation.condition is not None
            for inst in circuit.data
        )
        if has_conditionals:
            warnings.warn(
                "Circuit contains conditional operations. "
                "LRET currently treats these as unconditional.",
                UserWarning
            )
        
        # Check for barriers
        has_barriers = any(
            inst.operation.name == 'barrier'
            for inst in circuit.data
        )
        if has_barriers:
            # Barriers are ignored in simulation
            pass
        
        # Check for resets
        has_resets = any(
            inst.operation.name == 'reset'
            for inst in circuit.data
        )
        if has_resets:
            raise NotImplementedError(
                "LRET does not currently support reset operations"
            )
```

**Deliverables Day 3:**
- ✅ Gate mapper for 50+ Qiskit gates
- ✅ Circuit translator with validation
- ✅ Parameter extraction logic
- ✅ Measurement handling

---

#### **Day 4: Result Conversion & Noise Model Import**

**Morning (4 hours): Result Converter**

Create `python/lret_qiskit/translators/result_converter.py`:

```python
"""
Result Converter: LRET → Qiskit
================================

Converts LRET simulation results to Qiskit Result format.
"""

from qiskit.result.models import ExperimentResult, ExperimentResultData
import numpy as np

class ResultConverter:
    """
    Converts LRET results to Qiskit format.
    
    Handles:
    - Count dictionary formatting
    - Memory (shot-by-shot) results
    - Statevector/density matrix (if requested)
    - Metadata (rank, fidelity, etc.)
    """
    
    def convert_experiment(self, lret_result, circuit):
        """
        Convert single circuit result.
        
        Args:
            lret_result (dict): LRET simulation output
            circuit: Original Qiskit QuantumCircuit
            
        Returns:
            ExperimentResult: Qiskit experiment result
        """
        # Extract counts
        counts = self._format_counts(
            lret_result['counts'],
            circuit.num_clbits
        )
        
        # Build data object
        data = ExperimentResultData(
            counts=counts
        )
        
        # Add statevector if available
        if 'statevector' in lret_result:
            data.statevector = lret_result['statevector']
        
        # Add density matrix if available
        if 'density_matrix' in lret_result:
            data.density_matrix = lret_result['density_matrix']
        
        # Create experiment result
        result = ExperimentResult(
            shots=lret_result.get('shots', 1024),
            success=True,
            data=data,
            header={
                'name': circuit.name,
                'num_qubits': circuit.num_qubits,
                'num_clbits': circuit.num_clbits,
                'metadata': circuit.metadata or {}
            },
            status='DONE',
            seed_simulator=lret_result.get('seed'),
            metadata=self._extract_metadata(lret_result)
        )
        
        return result
    
    def _format_counts(self, lret_counts, num_clbits):
        """
        Format counts to Qiskit hex string format.
        
        Args:
            lret_counts (dict): LRET counts {state_int: count}
            num_clbits (int): Number of classical bits
            
        Returns:
            dict: Qiskit counts {'0x0': count, ...}
        """
        qiskit_counts = {}
        
        for state_int, count in lret_counts.items():
            # Convert integer to binary string
            binary = format(state_int, f'0{num_clbits}b')
            
            # Qiskit uses hex format
            hex_string = f"0x{int(binary, 2):x}"
            
            qiskit_counts[hex_string] = count
        
        return qiskit_counts
    
    def _extract_metadata(self, lret_result):
        """Extract LRET-specific metadata."""
        metadata = {}
        
        # Rank information
        if 'final_rank' in lret_result:
            metadata['final_rank'] = lret_result['final_rank']
        if 'max_rank' in lret_result:
            metadata['max_rank'] = lret_result['max_rank']
        
        # Truncation info
        if 'num_truncations' in lret_result:
            metadata['num_truncations'] = lret_result['num_truncations']
        
        # Fidelity
        if 'fidelity' in lret_result:
            metadata['fidelity'] = lret_result['fidelity']
        
        # Execution time
        if 'execution_time' in lret_result:
            metadata['execution_time_seconds'] = lret_result['execution_time']
        
        return metadata
```

**Afternoon (4 hours): Noise Model Import**

Create `python/lret_qiskit/noise_model_importer.py`:

```python
"""
Noise Model Importer
====================

Import Qiskit Aer noise models to LRET format.
"""

from qiskit_aer.noise import NoiseModel
import numpy as np

class NoiseModelImporter:
    """
    Import Qiskit Aer noise models for LRET.
    
    Converts Qiskit noise specifications to LRET-compatible
    Kraus operator format.
    """
    
    def __init__(self):
        """Initialize importer."""
        pass
    
    def import_noise_model(self, qiskit_noise_model):
        """
        Import Qiskit NoiseModel to LRET format.
        
        Args:
            qiskit_noise_model: Qiskit NoiseModel instance
            
        Returns:
            dict: LRET noise model specification
        """
        lret_noise = {
            'gate_errors': {},
            'readout_errors': {},
            'thermal_relaxation': {}
        }
        
        # Extract quantum errors
        for gate_name, error_dict in qiskit_noise_model._local_quantum_errors.items():
            for qubits, error in error_dict.items():
                self._add_gate_error(lret_noise, gate_name, qubits, error)
        
        # Extract readout errors
        for qubit, error in qiskit_noise_model._local_readout_errors.items():
            self._add_readout_error(lret_noise, qubit, error)
        
        return lret_noise
    
    def _add_gate_error(self, lret_noise, gate_name, qubits, error):
        """Add gate error to LRET noise model."""
        # Convert to Kraus operators
        kraus_ops = self._error_to_kraus(error)
        
        # Store in LRET format
        key = f"{gate_name}_{'_'.join(map(str, qubits))}"
        lret_noise['gate_errors'][key] = {
            'gate': gate_name,
            'qubits': list(qubits),
            'kraus_operators': kraus_ops
        }
    
    def _add_readout_error(self, lret_noise, qubit, error):
        """Add readout error to LRET noise model."""
        # Extract confusion matrix
        probabilities = error.probabilities
        
        lret_noise['readout_errors'][str(qubit)] = {
            'qubit': qubit,
            'confusion_matrix': probabilities.tolist()
        }
    
    def _error_to_kraus(self, error):
        """
        Convert Qiskit error to Kraus operators.
        
        Args:
            error: Qiskit QuantumError
            
        Returns:
            list: Kraus operators as numpy arrays
        """
        kraus_ops = []
        
        # Get Kraus representation
        if hasattr(error, 'to_quantumchannel'):
            channel = error.to_quantumchannel()
            kraus_ops = [K.data for K in channel.data]
        elif hasattr(error, 'to_instruction'):
            # Handle instruction-based errors
            inst = error.to_instruction()
            if hasattr(inst, 'to_matrix'):
                kraus_ops = [inst.to_matrix()]
        
        # Convert to list format
        return [K.tolist() if hasattr(K, 'tolist') else K for K in kraus_ops]
    
    @staticmethod
    def from_backend(backend):
        """
        Create LRET noise model from real backend.
        
        Args:
            backend: Qiskit Backend with calibration data
            
        Returns:
            dict: LRET noise model
        """
        # Get Qiskit noise model from backend
        from qiskit_aer.noise import NoiseModel as AerNoiseModel
        qiskit_noise = AerNoiseModel.from_backend(backend)
        
        # Import to LRET
        importer = NoiseModelImporter()
        return importer.import_noise_model(qiskit_noise)
```

**Deliverables Day 4:**
- ✅ Result converter with hex formatting
- ✅ Metadata extraction (rank, fidelity)
- ✅ Noise model import from Qiskit Aer
- ✅ Real backend calibration import

---

#### **Day 5: Testing & Integration**

**Full Day (8 hours): Comprehensive Testing**

Create `python/lret_qiskit/tests/test_backend.py`:

```python
"""
Backend Tests
=============

Test LRET backend functionality.
"""

import pytest
from qiskit import QuantumCircuit
from lret_qiskit import LRETProvider

class TestLRETBackend:
    """Test LRET backend interface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.provider = LRETProvider()
        self.backend = self.provider.get_backend('lret_simulator')
    
    def test_backend_initialization(self):
        """Test backend can be initialized."""
        assert self.backend is not None
        assert self.backend.name() == 'lret_simulator'
        assert self.backend.backend_version == '1.0.0'
    
    def test_target_definition(self):
        """Test backend target has gates."""
        target = self.backend.target
        assert target.num_qubits == 28
        
        # Check single-qubit gates
        assert target.operation_names
        assert 'h' in target.operation_names
        assert 'cx' in target.operation_names
    
    def test_bell_state(self):
        """Test Bell state preparation."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        job = self.backend.run(circuit, shots=1000)
        result = job.result()
        
        assert result.success
        counts = result.get_counts()
        
        # Should see |00⟩ and |11⟩
        assert '0x0' in counts or '0x3' in counts
    
    def test_multiple_circuits(self):
        """Test batch execution."""
        circuits = []
        for i in range(3):
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            circuits.append(qc)
        
        job = self.backend.run(circuits, shots=100)
        result = job.result()
        
        assert len(result.results) == 3
    
    def test_parameterized_circuit(self):
        """Test parameterized gates."""
        circuit = QuantumCircuit(1, 1)
        circuit.rx(np.pi/4, 0)
        circuit.ry(np.pi/3, 0)
        circuit.measure(0, 0)
        
        job = self.backend.run(circuit, shots=1000)
        result = job.result()
        
        assert result.success
```

Create `python/lret_qiskit/tests/test_integration.py`:

```python
"""
Integration Tests
=================

End-to-end integration tests with real Qiskit workflows.
"""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, EfficientSU2
from lret_qiskit import LRETProvider

class TestIntegration:
    """Integration tests with Qiskit ecosystem."""
    
    def test_qft_circuit(self):
        """Test Quantum Fourier Transform."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Create QFT circuit
        qc = QFT(4)
        qc.measure_all()
        
        job = backend.run(qc, shots=1000)
        result = job.result()
        
        assert result.success
        assert 'final_rank' in result.results[0].metadata
    
    def test_variational_circuit(self):
        """Test variational ansatz (VQE-like)."""
        provider = LRETProvider()
        backend = provider.get_backend()
        
        # Create EfficientSU2 ansatz
        ansatz = EfficientSU2(4, reps=2)
        qc = QuantumCircuit(4)
        qc.compose(ansatz, inplace=True)
        qc.measure_all()
        
        # Bind parameters
        from qiskit.circuit import Parameter
        import numpy as np
        
        params = {p: np.random.random() for p in ansatz.parameters}
        qc_bound = qc.assign_parameters(params)
        
        job = backend.run(qc_bound, shots=100)
        result = job.result()
        
        assert result.success
```

**Deliverables Day 5:**
- ✅ 20+ unit tests
- ✅ Integration tests with Qiskit library circuits
- ✅ Performance benchmarks
- ✅ CI/CD configuration

---

#### **Day 6-7: Documentation & Examples**

**Day 6 Morning: API Documentation**

Create `python/lret_qiskit/docs/api_reference.md`:

```markdown
# LRET Qiskit Integration - API Reference

## Provider

### LRETProvider

Main entry point for accessing LRET backends.

**Usage:**
```python
from lret_qiskit import LRETProvider

provider = LRETProvider()
backend = provider.get_backend('lret_simulator')
```

**Methods:**
- `backends(name=None, filters=None)` - List available backends
- `get_backend(name)` - Get specific backend

**Available Backends:**
- `lret_simulator` - Default (ε=1e-4)
- `lret_simulator_accurate` - High accuracy (ε=1e-6)
- `lret_simulator_fast` - Fast mode (ε=1e-3)

## Backend

### LRETBackend

Qiskit BackendV2 implementation for LRET.

**Options:**
- `shots` (int): Number of measurement shots (default: 1024)
- `epsilon` (float): SVD truncation threshold (default: 1e-4)
- `seed_simulator` (int): Random seed for reproducibility
- `noise_model`: Qiskit NoiseModel or LRET noise dict
- `initial_rank` (int): Initial rank of density matrix (default: 1)
- `parallelization` (str): 'auto', 'row', 'column', 'hybrid'

**Example:**
```python
backend = provider.get_backend('lret_simulator')
job = backend.run(circuit, shots=2000, epsilon=1e-5)
```

## Noise Models

### NoiseModelImporter

Import Qiskit Aer noise models.

**Usage:**
```python
from qiskit_aer.noise import NoiseModel
from lret_qiskit import NoiseModelImporter

# From Qiskit noise model
qiskit_noise = NoiseModel()
# ... add errors ...

importer = NoiseModelImporter()
lret_noise = importer.import_noise_model(qiskit_noise)

# Use with backend
job = backend.run(circuit, noise_model=lret_noise)
```

**From Real Backend:**
```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
real_backend = service.backend('ibm_osaka')

lret_noise = NoiseModelImporter.from_backend(real_backend)
job = backend.run(circuit, noise_model=lret_noise)
```
```

**Day 6 Afternoon + Day 7: User Guides & Examples**

Create `python/lret_qiskit/examples/01_getting_started.py`:

```python
"""
Example 1: Getting Started with LRET
=====================================

Basic usage of LRET backend with Qiskit.
"""

from qiskit import QuantumCircuit
from lret_qiskit import LRETProvider

# Initialize provider
provider = LRETProvider()
backend = provider.get_backend('lret_simulator')

print(f"Backend: {backend.name()}")
print(f"Max qubits: {backend.target.num_qubits}")
print(f"Supported gates: {list(backend.target.operation_names)[:10]}...")

# Create simple circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

print("\nCircuit:")
print(circuit)

# Run simulation
job = backend.run(circuit, shots=1000)
result = job.result()

print(f"\nJob Status: {job.status()}")
print(f"Success: {result.success}")

# Get counts
counts = result.get_counts()
print(f"\nMeasurement counts: {counts}")

# Get metadata
metadata = result.results[0].metadata
print(f"\nLRET Metadata:")
print(f"  Final rank: {metadata.get('final_rank', 'N/A')}")
print(f"  Execution time: {metadata.get('execution_time_seconds', 'N/A')}s")
```

Create `python/lret_qiskit/examples/02_vqe_example.py`:

```python
"""
Example 2: VQE with LRET Backend
=================================

Variational Quantum Eigensolver using LRET for simulation.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from lret_qiskit import LRETProvider
import numpy as np

# Define Hamiltonian (H2 molecule)
hamiltonian = SparsePauliOp.from_list([
    ('II', -1.05),
    ('IZ', 0.39),
    ('ZI', -0.39),
    ('ZZ', -0.01),
    ('XX', 0.18)
])

# Create ansatz
ansatz = TwoLocal(2, 'ry', 'cz', reps=3, entanglement='linear')

# Get LRET backend
provider = LRETProvider()
backend = provider.get_backend('lret_simulator')

# Create estimator (Qiskit primitive)
estimator = Estimator(backend=backend)

# Optimization loop
def cost_function(params):
    """Compute expectation value."""
    result = estimator.run(ansatz, hamiltonian, params).result()
    return result.values[0]

# Simple gradient descent
params = np.random.random(ansatz.num_parameters) * 2 * np.pi
learning_rate = 0.1

print("VQE Optimization:")
for iteration in range(20):
    energy = cost_function(params)
    print(f"Iteration {iteration}: Energy = {energy:.6f}")
    
    # Gradient (finite difference)
    gradient = np.zeros_like(params)
    epsilon = 0.01
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        
        gradient[i] = (cost_function(params_plus) - cost_function(params_minus)) / (2 * epsilon)
    
    # Update parameters
    params -= learning_rate * gradient

print(f"\nFinal energy: {energy:.6f}")
print(f"Target energy: -1.85 (FCI)")
```

Create `python/lret_qiskit/examples/03_noise_simulation.py`:

```python
"""
Example 3: Noisy Simulation with Real Device Data
==================================================

Simulate circuits with noise models from real IBM devices.
"""

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer.noise import NoiseModel
from lret_qiskit import LRETProvider, NoiseModelImporter

# Get LRET backend
provider = LRETProvider()
backend = provider.get_backend('lret_simulator')

# Option 1: Import from real IBM device
try:
    service = QiskitRuntimeService()
    ibm_backend = service.backend('ibm_osaka')
    
    # Create LRET noise model from IBM backend
    lret_noise = NoiseModelImporter.from_backend(ibm_backend)
    
    print("Loaded noise model from IBM Osaka (127 qubits)")
    
except Exception as e:
    print(f"Could not load IBM backend: {e}")
    print("Using synthetic noise model instead")
    
    # Option 2: Create synthetic noise model
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error
    
    noise_model = NoiseModel()
    
    # Add depolarizing error to gates
    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    # Import to LRET format
    importer = NoiseModelImporter()
    lret_noise = importer.import_noise_model(noise_model)

# Create test circuit
circuit = QuantumCircuit(4, 4)
circuit.h(0)
for i in range(3):
    circuit.cx(i, i+1)
circuit.measure(range(4), range(4))

print("\nCircuit depth:", circuit.depth())

# Run without noise
print("\nRunning noiseless simulation...")
job_ideal = backend.run(circuit, shots=2000)
result_ideal = job_ideal.result()
counts_ideal = result_ideal.get_counts()

print("Ideal counts (top 3):")
for bitstring, count in sorted(counts_ideal.items(), key=lambda x: -x[1])[:3]:
    print(f"  {bitstring}: {count}")

# Run with noise
print("\nRunning noisy simulation...")
job_noisy = backend.run(circuit, shots=2000, noise_model=lret_noise)
result_noisy = job_noisy.result()
counts_noisy = result_noisy.get_counts()

print("Noisy counts (top 5):")
for bitstring, count in sorted(counts_noisy.items(), key=lambda x: -x[1])[:5]:
    print(f"  {bitstring}: {count}")

# Compare rank growth
metadata_ideal = result_ideal.results[0].metadata
metadata_noisy = result_noisy.results[0].metadata

print(f"\nRank comparison:")
print(f"  Ideal: {metadata_ideal.get('final_rank', 'N/A')}")
print(f"  Noisy: {metadata_noisy.get('final_rank', 'N/A')}")
```

**Deliverables Day 6-7:**
- ✅ Complete API reference documentation
- ✅ 5+ example scripts
- ✅ User guide with best practices
- ✅ README with quickstart

---

### **Testing Strategy**

**Unit Tests (50+ tests):**
```bash
pytest python/lret_qiskit/tests/test_backend.py -v
pytest python/lret_qiskit/tests/test_translator.py -v
pytest python/lret_qiskit/tests/test_noise_import.py -v
```

**Integration Tests:**
```bash
pytest python/lret_qiskit/tests/test_integration.py -v
```

**Performance Benchmarks:**
```python
# Compare LRET vs Qiskit Aer
python python/lret_qiskit/benchmarks/compare_aer.py

# Test scaling
python python/lret_qiskit/benchmarks/scaling_test.py
```

---

### **Success Metrics**

**Functionality:**
- ✅ All Qiskit gates supported (50+)
- ✅ Noise model import working
- ✅ Result format 100% compatible
- ✅ Pass Qiskit test suite

**Performance:**
- 🎯 2-10x faster than Aer for noisy circuits (>12 qubits)
- 🎯 10-100x less memory than Aer
- 🎯 < 5% overhead for translation layer

**Adoption:**
- 🎯 10,000+ PyPI downloads in first month
- 🎯 50+ GitHub stars
- 🎯 5+ community examples/tutorials

---

### **Installation & Distribution**

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='lret-qiskit',
    version='1.0.0',
    description='LRET Backend for Qiskit',
    author='LRET Development Team',
    author_email='contact@lret-quantum.org',
    url='https://github.com/kunal5556/LRET',
    packages=find_packages(),
    install_requires=[
        'qiskit>=1.0.0',
        'qiskit-aer>=0.13.0',
        'qlret>=1.0.0',  # LRET Python bindings
        'numpy>=1.20.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'pytest-cov', 'black', 'pylint'],
        'ibm': ['qiskit-ibm-runtime>=0.15.0'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
```

**PyPI Publishing:**
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

---

### **Maintenance & Updates**

**CI/CD Pipeline (.github/workflows/qiskit_tests.yml):**

```yaml
name: Qiskit Integration Tests

on:
  push:
    paths:
      - 'python/lret_qiskit/**'
  pull_request:
    paths:
      - 'python/lret_qiskit/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
        qiskit-version: ['1.0.0', '1.1.0']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install qiskit==${{ matrix.qiskit-version }}
          pip install -e python/lret_qiskit
      
      - name: Run tests
        run: |
          pytest python/lret_qiskit/tests/ -v --cov
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Summary: 7.1.1 Qiskit Integration

**Total Duration:** 5-7 days  
**Team Size:** 1 developer  
**Lines of Code:** ~3,000 (Python)

**Deliverables:**
- ✅ LRETProvider with 3 backend variants
- ✅ BackendV2 implementation (50+ gates)
- ✅ Circuit translator with validation
- ✅ Result converter with metadata
- ✅ Noise model importer (Qiskit Aer compatible)
- ✅ 50+ unit tests
- ✅ Complete documentation + 5 examples
- ✅ PyPI package ready

**Expected Impact:**
- 100,000+ potential users
- 10,000+ downloads in first month
- 50+ citations in first year
- Enterprise consulting opportunities

**Next Steps:**
- Monitor adoption metrics
- Gather community feedback
- Add requested features (pulse simulation, etc.)
- Optimize performance hotspots

---

