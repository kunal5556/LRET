# PROXIMA: AI Agent for Quantum Simulation Orchestration

## Complete Design & Build Guide

**Version:** 1.0.0  
**Date:** January 7, 2026  
**Inspired By:** OpenCode AI, Crush (Charmbracelet)  


---

# PART 1: STRATEGIC SYSTEM SKETCH

## 1.1 Overall Architecture Overview

Proxima is designed as a modular, event-driven AI agent system with clear separation of concerns. The architecture follows a layered approach with five primary tiers:

### Tier 1: User Interface Layer (UIL)
- **Terminal User Interface (TUI):** Built using a framework similar to Bubble Tea (Go) or Textual (Python)
- **Command Parser:** Interprets natural language and structured commands
- **Display Renderer:** Renders execution timers, progress bars, status updates, and results
- **Input Handler:** Manages keyboard shortcuts, permission dialogs, and user consent flows

### Tier 2: Agent Intelligence Layer (AIL)
- **LLM Orchestrator:** Manages connections to remote APIs (OpenAI, Anthropic, Google, etc.) and local LLMs (Ollama, LM Studio, llama.cpp)
- **Query Analyzer:** Parses user intent, extracts parameters, identifies backend requirements
- **Backend Selector:** AI-driven decision engine that chooses optimal backend when user does not specify
- **Planning Engine:** Generates execution plans with stages, dependencies, and rollback points
- **Insight Generator:** Transforms raw outputs into human-readable analytical summaries

### Tier 3: Execution Control Layer (ECL)
- **State Machine:** Manages execution states (IDLE, PLANNING, EXECUTING, PAUSED, ABORTING, ROLLING_BACK, COMPLETED, FAILED)
- **Timer Service:** Tracks execution duration at global and stage levels
- **Process Manager:** Spawns, monitors, and controls backend processes
- **Checkpoint Manager:** Creates snapshots for pause/resume and rollback functionality
- **Resource Monitor:** Continuously tracks RAM, CPU, GPU, and disk usage

### Tier 4: Backend Abstraction Layer (BAL)
- **Backend Registry:** Catalog of all available backends with capabilities metadata
- **Unified Backend Interface:** Common protocol all backends must implement
- **Backend Adapters:**
  - LRET Adapter (C++ via pybind11 or subprocess)
  - Cirq Adapter (Python native)
  - Qiskit Aer Adapter (Python native)
  - Future adapters (PennyLane, Stim, etc.)
- **Result Normalizer:** Converts backend-specific outputs to unified schema

### Tier 5: Persistence & Configuration Layer (PCL)
- **Session Manager:** Stores and retrieves conversation/execution sessions
- **Configuration Store:** Manages user preferences, API keys, backend settings
- **Result Archive:** Persists execution outputs (.csv, .xlsx, JSON)
- **Agent Instruction Parser:** Reads and interprets proxima_agent.md files

---

## 1.2 Core Components Detailed

### Component A: Command & Query Processing Unit (CQPU)
**Purpose:** First point of contact for all user inputs

**Responsibilities:**
- Parse natural language queries using LLM
- Extract explicit backend preferences if stated
- Identify simulation parameters (qubits, depth, noise, etc.)
- Detect comparison requests (multi-backend runs)
- Flag resource-intensive operations for pre-checks

**Data Flow:**
User Input → Tokenization → Intent Classification → Parameter Extraction → Structured Command Object

### Component B: Intelligent Backend Selector (IBS)
**Purpose:** Automatic backend selection when user does not specify

**Decision Factors:**
- Circuit complexity (qubit count, gate depth)
- Noise model requirements
- Simulation type (statevector vs density matrix)
- Available hardware resources
- Historical performance data
- Backend-specific feature support

**Selection Algorithm:**
1. Analyze query parameters
2. Filter backends by capability match
3. Score remaining backends by efficiency/suitability
4. Select highest-scoring backend
5. Generate explanation for user

### Component C: Resource Sentinel (RS)
**Purpose:** Continuous hardware monitoring and protection

**Monitored Metrics:**
- Available RAM (absolute and percentage)
- RAM consumption rate during execution
- CPU utilization per core
- GPU memory (if applicable)
- Disk I/O and free space
- Swap usage

**Threshold Actions:**
- WARNING (70% RAM): Notify user, continue execution
- CAUTION (85% RAM): Prompt user for continuation consent
- CRITICAL (95% RAM): Pause execution, request explicit force-continue or abort
- FATAL (99%+ RAM): Emergency abort with state preservation attempt

### Component D: Execution Orchestrator (EO)
**Purpose:** Central conductor of all execution flows

**State Transitions:**
```
IDLE → PLANNING → READY → EXECUTING → COMPLETED
                    ↓         ↓
                 ABORTED   PAUSED → RESUMING → EXECUTING
                    ↓         ↓
              ROLLING_BACK  FAILED
```

**Stage Tracking:**
- Each execution is divided into discrete stages
- Stages: INITIALIZATION, VALIDATION, PRE-EXECUTION, EXECUTION, POST-PROCESSING, INTERPRETATION, CLEANUP
- Each stage has entry/exit timestamps and status

### Component E: Result Interpretation Engine (RIE)
**Purpose:** Transform raw outputs into actionable insights

**Capabilities:**
- Parse CSV and XLSX files using pandas or openpyxl
- Statistical analysis (mean, variance, fidelity metrics)
- Comparative analysis across backends
- Trend detection and anomaly flagging
- Natural language summary generation via LLM
- Visualization data preparation (for future UI charts)

### Component F: LLM Integration Hub (LIH)
**Purpose:** Unified interface for all LLM interactions

**Supported Providers:**
- Remote: OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, Azure OpenAI
- Local: Ollama, LM Studio, llama.cpp, text-generation-webui

**Consent Model:**
- Every LLM invocation requires prior user consent (configurable per-session)
- Clear distinction between local and remote LLM calls
- Token usage and cost estimation displayed before confirmation
- Explicit labeling of which LLM is being used at any moment

### Component G: Agent Instruction Processor (AIP)
**Purpose:** Parse and execute proxima_agent.md files

**File Format Support:**
- Markdown-based instruction sets
- Command blocks with execution directives
- Variable definitions and substitutions
- Conditional execution blocks
- Loop constructs for batch operations

**Architecture Anticipation:**
- Plugin system for custom instruction handlers
- Sandboxed execution for untrusted instruction files
- Validation and dry-run modes

---

## 1.3 Data, Control, and Decision Flow

### Primary Data Flow
```
[User Input]
     ↓
[Command Parser] → extracts intent, params, preferences
     ↓
[Query Analyzer] → determines execution requirements
     ↓
[Backend Selector] → chooses backend (if not specified)
     ↓
[Planning Engine] → generates staged execution plan
     ↓
[Resource Sentinel] → pre-execution hardware check
     ↓
[User Consent Gate] → displays plan, requests approval
     ↓
[Execution Orchestrator] → runs stages sequentially
     ↓
[Backend Adapter] → invokes actual simulation
     ↓
[Result Normalizer] → standardizes output format
     ↓
[Interpretation Engine] → generates insights
     ↓
[Display Renderer] → presents results to user
     ↓
[Session Manager] → persists for future reference
```

### Control Flow for Abort/Pause/Resume
```
[User requests ABORT]
     ↓
[Execution Orchestrator] → sets state to ABORTING
     ↓
[Process Manager] → sends termination signal to backend
     ↓
[Checkpoint Manager] → saves current state (if possible)
     ↓
[Cleanup Handler] → releases resources
     ↓
[State] → ABORTED (with partial results if available)

[User requests PAUSE]
     ↓
[Execution Orchestrator] → sets state to PAUSING
     ↓
[Checkpoint Manager] → creates full snapshot
     ↓
[Process Manager] → suspends backend process (SIGSTOP or equivalent)
     ↓
[State] → PAUSED (snapshot ID stored)

[User requests RESUME]
     ↓
[Execution Orchestrator] → loads snapshot
     ↓
[Process Manager] → resumes backend process (SIGCONT or equivalent)
     ↓
[State] → EXECUTING (from checkpoint)
```

### Decision Flow for Backend Selection
```
[No backend specified by user]
     ↓
[Extract: qubit_count, gate_depth, noise_model, sim_type]
     ↓
[Filter backends by capability]
     ↓
[Score by:]
  - Memory efficiency for given qubit count
  - Speed benchmarks for similar circuits
  - Noise model support fidelity
  - User's historical preference (if any)
     ↓
[Select top scorer]
     ↓
[Generate explanation:]
  "Selected LRET because: 12-qubit circuit with noise benefits 
   from low-rank approximation, estimated 4x faster than 
   full density matrix simulation."
     ↓
[Present to user for confirmation]
```

---

## 1.4 Feature Interconnection Map

### Feature 1 (Execution Timer) connects to:
- Execution Orchestrator (stage timestamps)
- Display Renderer (live updates)
- Session Manager (historical timing data)

### Feature 2 (Backend Selection) connects to:
- Query Analyzer (parameter extraction)
- Backend Registry (capability lookup)
- LLM Orchestrator (explanation generation)
- User Consent Gate (selection confirmation)

### Feature 3 (Fail-Safe & Resource Awareness) connects to:
- Resource Sentinel (continuous monitoring)
- Execution Orchestrator (state transitions)
- User Consent Gate (force-execute dialogs)
- Checkpoint Manager (emergency snapshots)

### Feature 4 (Execution Control) connects to:
- State Machine (all transitions)
- Process Manager (signal handling)
- Checkpoint Manager (snapshots)
- Display Renderer (state visualization)

### Feature 5 (Result Interpretation) connects to:
- Result Normalizer (unified data)
- LLM Orchestrator (insight generation)
- Display Renderer (formatted output)
- Session Manager (insight persistence)

### Feature 6 (Multi-Backend Comparison) connects to:
- Backend Registry (parallel capability check)
- Execution Orchestrator (parallel/sequential runs)
- Result Normalizer (unified comparison schema)
- Interpretation Engine (comparative analysis)

### Feature 7 (Planning Pipeline) connects to:
- Planning Engine (plan generation)
- Display Renderer (plan visualization)
- User Consent Gate (plan approval)
- Execution Orchestrator (stage execution)
- all of this will be possible through model integrated by API keys

### Feature 8 (LLM Integration) connects to:
- LLM Integration Hub (provider management)
- Configuration Store (API keys, preferences)
- User Consent Gate (invocation approval)
- Interpretation Engine (insight generation)

### Feature 9 (agent.md Compatibility) connects to:
- Agent Instruction Processor (parsing)
- Command Parser (instruction to command)
- Execution Orchestrator (batch execution)
- Configuration Store (instruction file paths)

### Feature 10 (OpenCode/Crush Features) connects to:
- All components (session management, LSP, MCP, custom commands)

### Feature 11 (UI) connects to:
- Display Renderer (visual output)
- Input Handler (user interactions)
- All components via event bus

---

# PART 2: PHASED ROADMAP

## Phase Overview

| Phase | Name | Duration | Key Deliverables |
|-------|------|----------|------------------|
| 1 | Foundation & Core Infrastructure | 2-3 weeks | Project structure, config system, basic TUI shell |
| 2 | Backend Abstraction Layer | 2-3 weeks | Backend registry, unified interface, LRET adapter |
| 3 | Execution Control System | 2 weeks | State machine, timer service, process manager |
| 4 | Resource Monitoring & Fail-Safe | 1-2 weeks | Resource sentinel, consent gates, thresholds |
| 5 | Additional Backend Adapters | 2 weeks | Cirq adapter, Qiskit Aer adapter |
| 6 | LLM Integration Hub | 2-3 weeks | Remote providers, local LLM support, consent model |
| 7 | Intelligent Backend Selection | 1-2 weeks | Auto-selection algorithm, explanation generation |
| 8 | Planning & Staged Execution | 1-2 weeks | Planning engine, stage tracking, visualization |
| 9 | Result Interpretation & Insights | 2 weeks | File parsing, statistical analysis, LLM summaries |
| 10 | Multi-Backend Comparison | 1-2 weeks | Parallel runs, comparative analysis |
| 11 | Pause/Resume/Rollback | 2 weeks | Checkpoint manager, state restoration |
| 12 | Agent Instruction System | 2 weeks | proxima_agent.md parser, instruction execution |
| 13 | Session Management & Persistence | 1-2 weeks | Session storage, history, replay |
| 14 | Advanced Features (OpenCode/Crush) | 2-3 weeks | MCP, LSP, custom commands, skills |
| 15 | UI Polish & Documentation | 2 weeks | TUI refinement, help system, docs |
| 16 | Testing, Hardening & Release | 2-3 weeks | Integration tests, edge cases, packaging |

**Total Estimated Duration:** 24-34 weeks (6-8 months)

---

## Phase 1: Foundation & Core Infrastructure

### Objectives
- Establish project structure and build system
- Implement configuration management
- Create basic TUI shell with input handling
- Set up logging and error handling infrastructure

### Deliverables
1. Project repository with directory structure
2. Configuration file schema and loader
3. Basic TUI with command prompt
4. Logging system with levels and file output
5. Error handling framework with custom exceptions

### Technical Components
- **Language:** Go (for TUI performance, inspired by Crush) or Python (for ecosystem compatibility)
- **TUI Framework:** Bubble Tea (Go) or Textual (Python)
- **Configuration:** JSON schema with validation (similar to Crush's crush.json)
- **Logging:** Structured logging with rotation

### Exit Criteria
- User can launch Proxima and see welcome screen
- Configuration file is read and validated on startup
- Commands can be typed and echoed back
- Logs are written to file with timestamps

---

## Phase 2: Backend Abstraction Layer

### Objectives
- Design and implement unified backend interface
- Create backend registry with capability metadata
- Implement first backend adapter (LRET)
- Build result normalization system

### Deliverables
1. Backend interface specification
2. Backend registry with dynamic discovery
3. LRET adapter with full integration
4. Result schema and normalizer
5. Backend health check system

### Technical Components
- **Interface Pattern:** Abstract base class or interface definition
- **LRET Integration:** Via pybind11 Python bindings or subprocess with JSON I/O
- **Registry:** In-memory catalog with JSON-based capability definitions
- **Result Schema:** Unified JSON structure for all backend outputs

### Exit Criteria
- LRET can be invoked through Proxima
- Results are returned in normalized format
- Backend capabilities are queryable
- Health checks report backend availability

---

## Phase 3: Execution Control System

### Objectives
- Implement finite state machine for execution states
- Create timer service with stage-level granularity
- Build process manager for backend lifecycle
- Implement basic start/abort functionality

### Deliverables
1. State machine with all defined states and transitions
2. Timer service with start/stop/elapsed methods
3. Process manager with spawn/terminate capabilities
4. State change event system
5. Basic abort handling

### Technical Components
- **State Machine:** Enum-based states with transition validation
- **Timer:** High-resolution timer with stage tracking
- **Process Management:** subprocess (Python) or os/exec (Go)
- **Events:** Observer pattern for state change notifications

### Exit Criteria
- Execution starts and moves through states correctly
- Elapsed time is displayed and updated live
- Abort command terminates running execution
- State transitions are logged and visible

---

## Phase 4: Resource Monitoring & Fail-Safe

### Objectives
- Implement continuous resource monitoring
- Define and enforce resource thresholds
- Create user consent dialog system
- Implement force-execute with explicit consent

### Deliverables
1. Resource sentinel with RAM/CPU monitoring
2. Threshold configuration (warning, caution, critical)
3. Consent dialog component
4. Force-execute flow with acknowledgment
5. Pre-execution resource validation

### Technical Components
- **Monitoring:** psutil (Python) or gopsutil (Go) for system metrics
- **Thresholds:** Configurable in proxima.json
- **Dialogs:** TUI modal dialogs with explicit yes/no
- **Consent Logging:** All consent decisions recorded

### Exit Criteria
- Resource usage is displayed during execution
- Warnings appear at configured thresholds
- User can force-execute after warning
- Execution pauses at critical thresholds pending consent

---

## Phase 5: Additional Backend Adapters

### Objectives
- Implement Cirq adapter (DensityMatrixSimulator, Simulator)
- Implement Qiskit Aer adapter (DensityMatrix, Statevector)
- Ensure all adapters conform to unified interface
- Add backend-specific configuration options

### Deliverables
1. Cirq adapter with both simulators
2. Qiskit Aer adapter with both simulators
3. Backend-specific configuration sections
4. Extended result normalizer for new backends
5. Backend selection menu in TUI

### Technical Components
- **Cirq:** Python cirq library, cirq.DensityMatrixSimulator, cirq.Simulator
- **Qiskit Aer:** qiskit-aer library, AerSimulator with method='density_matrix' or 'statevector'
- **Configuration:** Per-backend options (shots, seed, noise model path)

### Exit Criteria
- All five backend variants are invokable
- Results from all backends normalize to same schema
- User can explicitly select any backend
- Backend configurations are editable

---

## Phase 6: LLM Integration Hub

### Objectives
- Implement multi-provider LLM client
- Add support for local LLMs
- Create consent model for LLM invocation
- Distinguish and label local vs remote usage

### Deliverables
1. LLM client with provider abstraction
2. Remote provider integrations (OpenAI, Anthropic, Gemini, Groq)
3. Local LLM integrations (Ollama, LM Studio)
4. API key management in configuration
5. Consent flow for each LLM call
6. Usage/cost estimation display

### Technical Components
- **Remote Clients:** openai, anthropic, google-generativeai SDKs
- **Local Clients:** HTTP to localhost (Ollama default port 11434, LM Studio 1234)
- **Consent:** Per-call or per-session approval modes
- **Labels:** Clear indicators in TUI for LLM source

### Exit Criteria
- Remote LLM calls work with valid API keys
- Local LLM calls work with running local server
- Each call shows consent prompt (unless session-approved)
- Output clearly labels which LLM was used

---

## Phase 7: Intelligent Backend Selection

### Objectives
- Implement query parameter extraction
- Build scoring algorithm for backend suitability
- Generate natural language explanations for selections
- Allow user override after auto-selection

### Deliverables
1. Parameter extractor (qubits, depth, noise, type)
2. Backend capability matcher
3. Scoring function with configurable weights
4. Explanation generator using LLM
5. Selection confirmation dialog

### Technical Components
- **Extraction:** LLM-based intent parsing or rule-based extraction
- **Matching:** Capability filter against registry
- **Scoring:** Weighted multi-factor scoring
- **Explanation:** Template + LLM completion

### Exit Criteria
- Queries without explicit backend trigger auto-selection
- Selection includes explanation of reasoning
- User can accept or override selection
- Selection history informs future suggestions

---

## Phase 8: Planning & Staged Execution

### Objectives
- Implement planning engine for execution stages
- Create plan visualization in TUI
- Add stage-level progress tracking
- Enable plan modification before execution

### Deliverables
1. Planning engine with stage generator
2. Plan data structure (stages, dependencies, estimates)
3. Plan viewer component in TUI
4. Stage progress indicators
5. Plan approval and modification interface

### Technical Components
- **Stages:** INIT, VALIDATE, PRE-EXEC, EXECUTE, POST-PROCESS, INTERPRET, CLEANUP
- **Visualization:** Tree or list view with status icons
- **Progress:** Percentage or step counters per stage
- **Modification:** Add/remove/reorder stages before start

### Exit Criteria
- Every execution shows generated plan first
- User approves plan before execution begins
- Each stage shows real-time progress
- Completed stages show duration and status

---

## Phase 9: Result Interpretation & Insights

### Objectives
- Implement file parsers for CSV and XLSX
- Build statistical analysis module
- Create LLM-powered insight generator
- Format insights for human readability

### Deliverables
1. CSV parser with automatic type detection
2. XLSX parser with sheet selection
3. Statistical analyzer (mean, std, fidelity, etc.)
4. Insight generator with LLM summarization
5. Formatted insight display in TUI

### Technical Components
- **Parsing:** pandas for CSV/XLSX handling
- **Statistics:** numpy, scipy for calculations
- **Insights:** LLM prompt engineering for summaries
- **Formatting:** Markdown-style output with sections

### Exit Criteria
- CSV and XLSX outputs are automatically parsed
- Statistical summaries are calculated and displayed
- LLM generates readable analytical insights
- Insights are decision-oriented, not data dumps

---

## Phase 10: Multi-Backend Comparison

### Objectives
- Enable running same simulation on multiple backends
- Collect and normalize results for comparison
- Generate comparative analysis
- Display side-by-side results

### Deliverables
1. Multi-backend execution mode
2. Parallel or sequential run coordination
3. Comparative result schema
4. Comparison analysis engine
5. Side-by-side display component

### Technical Components
- **Coordination:** Parallel with multiprocessing or sequential with aggregation
- **Comparison:** Metrics like fidelity difference, timing ratio, rank growth
- **Analysis:** LLM-assisted comparative narrative
- **Display:** Table or column-based comparison view

### Exit Criteria
- User can request multi-backend comparison run
- All selected backends execute with same parameters
- Results are displayed in comparative format
- Analysis highlights key differences and recommendations

---

## Phase 11: Pause/Resume/Rollback

### Objectives
- Implement checkpoint manager for state snapshots
- Enable pause with process suspension
- Enable resume from checkpoint
- Implement rollback to previous checkpoint

### Deliverables
1. Checkpoint manager with snapshot creation
2. Pause handler with process suspension
3. Resume handler with state restoration
4. Rollback mechanism to previous snapshot
5. Checkpoint viewer in TUI

### Technical Components
- **Snapshots:** Serialize execution state, queue pending operations
- **Suspension:** OS signals (SIGSTOP/SIGCONT on Unix, Job Objects on Windows)
- **Restoration:** Deserialize and resume from exact point
- **Rollback:** Load earlier checkpoint, discard subsequent state

### Exit Criteria
- Pause freezes execution and frees some resources
- Resume continues from exact pause point
- Rollback reverts to previous known-good state
- All state changes are logged and traceable

---

## Phase 12: Agent Instruction System

### Objectives
- Define proxima_agent.md file format specification
- Implement instruction parser
- Build instruction executor with sandbox
- Enable batch operations from instruction files

### Deliverables
1. File format specification document
2. Markdown instruction parser
3. Instruction executor with validation
4. Sandbox mode for untrusted files
5. Dry-run mode for testing instructions

### Technical Components
- **Format:** Markdown with fenced code blocks for commands
- **Parser:** Custom markdown parser with command extraction
- **Executor:** Dispatch commands to appropriate handlers
- **Sandbox:** Restricted permissions, no file system writes without consent

### Exit Criteria
- proxima_agent.md files are recognized and parsed
- Instructions execute in order with proper handling
- Dry-run shows what would execute without running
- Untrusted files run in restricted sandbox

---

## Phase 13: Session Management & Persistence

### Objectives
- Implement session storage with SQLite
- Enable session history navigation
- Support session replay for debugging
- Add session export capabilities

### Deliverables
1. Session data model and schema
2. SQLite persistence layer
3. Session list and selector in TUI
4. Session replay functionality
5. Export to JSON/Markdown

### Technical Components
- **Database:** SQLite with migrations (similar to Crush/OpenCode)
- **Schema:** Sessions, messages, executions, results
- **Selector:** List view with search and filter
- **Replay:** Step through historical session events

### Exit Criteria
- Sessions persist across Proxima restarts
- User can list and switch between sessions
- Historical sessions can be replayed read-only
- Sessions export to portable formats

---

## Phase 14: Advanced Features (OpenCode/Crush Inspired)

### Objectives
- Implement MCP (Model Context Protocol) support
- Add LSP integration for code intelligence
- Create custom command system
- Add Agent Skills support

### Deliverables
1. MCP client with stdio/http/sse transports
2. LSP client for relevant languages
3. Custom command definition and execution
4. Skills directory with discovery
5. Permissions system for tools

### Technical Components
- **MCP:** Implement MCP client per specification
- **LSP:** Language server connections for Python, C++
- **Commands:** User-defined command files (Markdown with prompts)
- **Skills:** SKILL.md files with agent capabilities

### Exit Criteria
- MCP servers can be connected and tools exposed
- LSP provides diagnostics to the agent
- Custom commands can be defined and invoked
- Skills extend agent capabilities dynamically

---

## Phase 15: UI Polish & Documentation

### Objectives
- Refine TUI with visual polish
- Implement comprehensive help system
- Create user documentation
- Add keyboard shortcut reference

### Deliverables
1. Polished TUI with consistent styling
2. In-app help command and overlays
3. User guide documentation (Markdown)
4. Developer guide documentation
5. Keyboard shortcut reference card

### Technical Components
- **Styling:** Consistent color palette, borders, spacing
- **Help:** Context-sensitive help, searchable
- **Docs:** Markdown files in /docs directory
- **Shortcuts:** Configurable, displayed in help

### Exit Criteria
- TUI looks professional and consistent
- Help is accessible from any screen
- Documentation covers all features
- Shortcuts are documented and work correctly

---

## Phase 16: Testing, Hardening & Release

### Objectives
- Comprehensive integration testing
- Edge case handling and hardening
- Performance optimization
- Release packaging and distribution

### Deliverables
1. Integration test suite
2. Edge case test coverage
3. Performance benchmarks
4. Release packages (binary, pip, Docker)
5. CI/CD pipeline

### Technical Components
- **Testing:** pytest (Python) or go test with integration tests
- **CI/CD:** GitHub Actions for automated testing and releases
- **Packaging:** PyPI, Homebrew, Scoop, Docker Hub
- **Benchmarks:** Timing and memory tests

### Exit Criteria
- All tests pass on CI
- Edge cases are handled gracefully
- Performance meets targets
- Releases are automated and reproducible

---

# PART 3: PHASE-BY-PHASE IMPLEMENTATION GUIDE

## PHASE 1: FOUNDATION & CORE INFRASTRUCTURE

### Step 1.1: Project Initialization and Structure

**Objective:** Create the foundational project structure with all necessary directories and initial configuration.

**Detailed Instructions:**

1. **Choose Primary Language:**
   - **Option A (Recommended for TUI performance):** Go with Bubble Tea framework
   - **Option B (Recommended for ecosystem compatibility):** Python with Textual framework
   - Decision factors: Go offers better concurrency and performance; Python offers richer quantum library ecosystem
   - For this guide, we proceed with Python + Textual for seamless integration with Cirq/Qiskit/LRET

2. **Create Directory Structure:**
   - Root directory: `proxima/`
   - Source directory: `proxima/src/` or `proxima/proxima/` (package name)
   - Subdirectories within source:
     - `core/` - Core infrastructure (config, logging, exceptions)
     - `ui/` - All TUI components
     - `backends/` - Backend adapters and registry
     - `execution/` - Execution control, state machine, timers
     - `intelligence/` - LLM integration, query analysis, planning
     - `monitoring/` - Resource monitoring, sentinel
     - `persistence/` - Session management, database
     - `utils/` - Utility functions and helpers
   - Configuration directory: `config/` at root
   - Documentation directory: `docs/` at root
   - Tests directory: `tests/` mirroring source structure
   - Assets directory: `assets/` for schemas, templates, etc.

3. **Initialize Package Management:**
   - Create `pyproject.toml` with Poetry or `setup.py` with setuptools
   - Define project metadata: name (proxima), version (0.1.0), description, authors
   - Specify minimum Python version (3.10+ for latest type hints)
   - Initial dependencies:
     - textual (TUI framework)
     - pydantic (configuration validation)
     - loguru (structured logging)
     - click (CLI argument parsing)
     - rich (terminal formatting, used by textual)

4. **Create Entry Point:**
   - Main script: `proxima/cli.py` or `proxima/__main__.py`
   - Define entry point that initializes configuration, logging, and launches TUI
   - Use click decorators for command-line interface
   - Support flags: `--config PATH`, `--debug`, `--version`

5. **Initialize Git Repository:**
   - Create `.gitignore` for Python (venv, __pycache__, .pyc, logs, etc.)
   - Create README.md with project overview and installation instructions
   - Create LICENSE file (MIT or Apache 2.0 recommended)
   - Initialize git and create initial commit

### Step 1.2: Configuration System Design and Implementation

**Objective:** Build a robust configuration system that handles user preferences, API keys, backend settings, and runtime options.

**Detailed Instructions:**

1. **Define Configuration Schema:**
   - Create `config/schema.json` defining all configuration options
   - Use JSON Schema specification for validation
   - Top-level sections:
     - `application` - global settings (debug, log_level, data_directory)
     - `providers` - LLM provider configurations
     - `backends` - quantum backend configurations
     - `monitoring` - resource threshold settings
     - `ui` - TUI preferences (theme, keybindings)
     - `permissions` - default consent settings

2. **Create Configuration Models:**
   - In `proxima/core/config.py`, define Pydantic models for configuration
   - Models should correspond to schema sections:
     - `ApplicationConfig` with fields: debug (bool), log_level (str), data_dir (Path)
     - `ProviderConfig` with fields: type (str), api_key (Optional[str]), base_url (Optional[str])
     - `BackendConfig` with fields: type (str), enabled (bool), priority (int), options (dict)
     - `MonitoringConfig` with fields: ram_warning_threshold (float), ram_critical_threshold (float)
     - `ProximaConfig` as root model containing all sub-configs
   - Use Pydantic validators for constraints (thresholds 0-1, valid paths, etc.)
   - Support environment variable expansion using `${VARIABLE}` syntax

3. **Implement Configuration Loader:**
   - Create `proxima/core/config_loader.py`
   - Define `ConfigLoader` class with methods:
     - `load_default()` - returns default configuration
     - `load_from_file(path: Path)` - loads and validates configuration file
     - `merge_configs(base, override)` - merges two configuration objects
     - `validate(config)` - validates against schema
   - Configuration search order (inspired by Crush):
     1. `./proxima.json` (project-local)
     2. `$XDG_CONFIG_HOME/proxima/proxima.json` (user config on Linux/Mac)
     3. `$HOME/.config/proxima/proxima.json` (user config fallback)
     4. `%LOCALAPPDATA%\proxima\proxima.json` (user config on Windows)
     5. Default built-in configuration
   - Merge configurations with precedence (1 overrides 2, 2 overrides 3, etc.)

4. **Environment Variable Support:**
   - Implement environment variable substitution in configuration values
   - Detect patterns like `${OPENAI_API_KEY}` or `$OPENAI_API_KEY`
   - Use `os.environ.get()` with optional defaults
   - Support escaping for literal dollar signs

5. **Configuration Validation and Error Handling:**
   - On load, validate entire configuration against schema
   - Collect all validation errors and present them clearly
   - Provide helpful error messages with path to problematic field
   - Support strict mode (fail on unknown keys) and lenient mode (warn only)

6. **Configuration Access Interface:**
   - Create global configuration singleton accessible throughout application
   - In `proxima/core/config.py`, define `get_config()` function returning current config
   - Support hot-reloading (optional feature for development)
   - Emit events when configuration changes

### Step 1.3: Logging Infrastructure

**Objective:** Establish comprehensive logging system with structured output, levels, rotation, and debugging support.

**Detailed Instructions:**

1. **Choose Logging Framework:**
   - Use `loguru` for rich features and simplicity
   - Advantages: automatic field capture, easy formatting, rotation, filtering

2. **Configure Log Levels:**
   - Standard levels: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Map to configuration: `log_level` in `application` section
   - Default level: INFO for production, DEBUG when `--debug` flag used

3. **Set Up Log Destinations:**
   - Console output: always enabled, formatted with colors using rich
   - File output: log to `{data_directory}/logs/proxima.log`
   - Log directory structure:
     - `proxima.log` - current log (rotated)
     - `proxima.YYYY-MM-DD.log` - archived logs by date
   - Implement rotation: size-based (50MB per file) or time-based (daily)
   - Retention policy: keep last 10 archived files, then delete oldest

4. **Structured Logging Format:**
   - JSON format for file logs (machine-readable)
   - Human-readable format for console
   - Include fields: timestamp, level, module, function, line, message, context
   - Context fields: session_id, execution_id, backend_name (when applicable)

5. **Create Logging Utilities:**
   - In `proxima/core/logging.py`, define helper functions:
     - `setup_logging(config)` - initializes loguru with configuration
     - `get_logger(name)` - returns logger instance for module
     - `log_execution_start(exec_id, backend, params)` - structured log for execution
     - `log_execution_end(exec_id, status, duration)` - structured log for completion
   - Use decorators for automatic function entry/exit logging (debug level)

6. **Debug Mode Enhancements:**
   - When debug=True, set level to DEBUG
   - Enable additional context capture (stack traces, variable dumps)
   - Log all backend communication (requests/responses)
   - Log all LLM interactions (prompts/completions)

7. **Log Viewing Command:**
   - Implement `proxima logs` command to display recent logs
   - Support flags: `--tail N` (last N lines), `--follow` (live tail), `--level LEVEL` (filter)
   - Use rich formatting for colored console output

### Step 1.4: Error Handling and Exception Framework

**Objective:** Design and implement custom exception hierarchy and error handling patterns for consistent error management.

**Detailed Instructions:**

1. **Define Exception Hierarchy:**
   - In `proxima/core/exceptions.py`, create base exception:
     - `ProximaError(Exception)` - base for all custom exceptions
   - Derive specific exception classes:
     - `ConfigurationError` - configuration issues
     - `BackendError` - backend-related failures
     - `ExecutionError` - execution failures
     - `ResourceError` - resource exhaustion or limits
     - `ValidationError` - input validation failures
     - `LLMError` - LLM communication failures
     - `PermissionDeniedError` - user denied consent
   - Each exception should include:
     - Descriptive message
     - Error code (for programmatic handling)
     - Context dictionary (relevant state information)
     - Optional `cause` (original exception)

2. **Error Context Capture:**
   - When raising exceptions, capture relevant context:
     - For BackendError: backend name, parameters, partial output
     - For ResourceError: current usage, threshold, limit
     - For ExecutionError: execution stage, elapsed time, state
   - Include context in exception initialization
   - Context should be structured (dict) for logging and display

3. **Error Formatting for Users:**
   - Create `proxima/core/error_formatter.py`
   - Define `format_error(exception)` function that:
     - Extracts error type, message, context
     - Formats as human-readable rich text (using rich library)
     - Provides actionable suggestions where applicable
     - Includes error code for reference
   - Different formats for console vs TUI display

4. **Global Exception Handler:**
   - Implement top-level exception handler in main entry point
   - Catch all `ProximaError` and `Exception` types
   - Log full stack trace to file (at ERROR level)
   - Display formatted error to user
   - Attempt graceful cleanup (release resources, save state)
   - Exit with appropriate error code (0=success, >0=error)

5. **Retry Mechanisms:**
   - For transient errors (network, LLM rate limits), implement retry logic
   - Create `proxima/utils/retry.py` with decorator:
     - `@retry(max_attempts=3, backoff=exponential, exceptions=[LLMError])`
   - Configurable retry policies per error type
   - Log each retry attempt with attempt number

6. **Error Recovery Strategies:**
   - Define recovery strategies for common errors:
     - ResourceError → suggest reducing parameters or closing other processes
     - BackendError → suggest alternative backend or configuration adjustment
     - LLMError → suggest checking API key or network connection
   - Implement automatic fallback for certain scenarios (e.g., switch to local LLM if remote fails)

### Step 1.5: Basic TUI Shell with Input Handling

**Objective:** Create the initial terminal user interface with command prompt, input handling, and basic navigation.

**Detailed Instructions:**

1. **Initialize Textual Application:**
   - In `proxima/ui/app.py`, create main `ProximaApp(App)` class
   - Define application metadata: title, name, stylesheet
   - Set up async event loop (Textual uses asyncio)

2. **Create Welcome Screen:**
   - Define `WelcomeScreen(Screen)` in `proxima/ui/screens/welcome.py`
   - Display:
     - Proxima logo (ASCII art)
     - Version number
     - Brief description
     - Quick start instructions
   - Add keybinding: Enter or Space to continue to main screen

3. **Design Main Layout:**
   - Create `MainScreen(Screen)` in `proxima/ui/screens/main.py`
   - Layout structure:
     - Header bar (top) - shows Proxima logo, current session, status
     - Main content area (center) - conversation history or results
     - Input area (bottom) - command prompt and text input
     - Footer bar (bottom-most) - keybinding hints
   - Use Textual containers: Vertical, Horizontal, ScrollableContainer

4. **Implement Command Input:**
   - In input area, use `Input` widget from Textual
   - Customize prompt prefix: `proxima>` or `❯`
   - Bind Enter key to submit command
   - Implement input history (up/down arrows navigate previous commands)
   - Store history in list, maintain cursor position

5. **Create Command Echo Display:**
   - When user submits command, display it in main content area
   - Format: `[timestamp] You: {command}`
   - Echo response: `[timestamp] Proxima: Command received: {command}`
   - Use rich Text with syntax highlighting for future code blocks

6. **Add Basic Keybindings:**
   - Ctrl+C or Ctrl+Q - quit application with confirmation dialog
   - Ctrl+L - clear screen/conversation history
   - Ctrl+? or F1 - show help overlay
   - Escape - cancel current input or close dialogs
   - Tab - autocomplete (placeholder for future)

7. **Implement Help Overlay:**
   - Create `HelpScreen(Screen)` in `proxima/ui/screens/help.py`
   - Display list of keybindings and commands
   - Navigate with j/k or arrow keys
   - Press Escape or q to close and return to main screen

8. **Status Display:**
   - In header bar, show status indicator:
     - IDLE (green) - ready for input
     - EXECUTING (yellow) - running backend
     - ERROR (red) - last operation failed
   - Update status reactively based on execution state

9. **Integrate with Configuration:**
   - Load UI preferences from configuration (theme, colors, keybindings)
   - Apply theme to Textual app via CSS
   - Support color schemes: default, dark, light, high-contrast

10. **Event Handling System:**
    - Set up event bus for inter-component communication
    - Define events: CommandSubmitted, ExecutionStarted, ExecutionCompleted, etc.
    - Use Textual's message system or implement custom event dispatcher
    - Components subscribe to relevant events and react accordingly

### Step 1.6: Testing and Validation

**Objective:** Ensure Phase 1 components work correctly through unit and integration tests.

**Detailed Instructions:**

1. **Set Up Testing Framework:**
   - Use pytest as test runner
   - Install pytest-asyncio for async tests (Textual is async)
   - Install pytest-mock for mocking
   - Create `tests/` directory mirroring `proxima/` structure

2. **Configuration Tests:**
   - Test schema validation with valid and invalid configs
   - Test configuration loading from file
   - Test configuration merging and precedence
   - Test environment variable expansion
   - Test error messages for malformed configs

3. **Logging Tests:**
   - Test log level filtering
   - Test log rotation and retention
   - Test structured logging output format
   - Test context capture in logs

4. **Exception Handling Tests:**
   - Test custom exception raising and catching
   - Test error context capture
   - Test error formatting for display
   - Test retry mechanisms with mock failures

5. **TUI Tests:**
   - Test application launches without errors
   - Test navigation between screens
   - Test command input and echo
   - Test keybindings trigger correct actions
   - Use Textual's testing utilities: `pilot` for simulating user interactions

6. **Integration Test:**
   - Test complete flow: launch app → load config → show welcome → enter command → echo response
   - Verify logs are written correctly
   - Verify configuration is applied to UI

7. **Manual Testing Checklist:**
   - Launch Proxima and verify welcome screen appears
   - Type command and verify echo in output
   - Test all keybindings (Ctrl+C, Ctrl+L, F1, etc.)
   - Check log file is created and contains entries
   - Modify config file and restart, verify changes applied

### Phase 1 Implementation Complete

**Artifacts Produced:**
- Functional project structure with all directories
- Configuration system with JSON schema and validation
- Logging infrastructure with console and file output
- Custom exception hierarchy and error handling
- Basic TUI with command input and echo
- Test suite covering core functionality

**Verification:**
- `python -m proxima` launches the application
- Welcome screen is displayed with proper formatting
- Commands can be typed and echoed back
- Configuration loads from file and applies settings
- Logs are written to file with rotation
- Tests pass with `pytest`

---

## PHASE 2: BACKEND ABSTRACTION LAYER

### Step 2.1: Design Unified Backend Interface

**Objective:** Create an abstract interface that all quantum backends must implement, ensuring consistent interaction patterns.

**Detailed Instructions:**

1. **Define Backend Capability Model:**
   - In `proxima/backends/capabilities.py`, create `BackendCapability` enum:
     - STATEVECTOR - supports statevector simulation
     - DENSITY_MATRIX - supports density matrix simulation
     - NOISE_MODELING - supports custom noise models
     - PARAMETRIC_CIRCUITS - supports parameterized circuits
     - MEASUREMENT_SAMPLING - supports shot-based measurements
     - GPU_ACCELERATION - can utilize GPU
     - CHECKPOINTING - supports pause/resume
   - Create `CapabilityMetadata` dataclass with fields:
     - max_qubits (int) - theoretical or practical qubit limit
     - supports_parallel (bool) - can run multiple circuits in parallel
     - memory_efficient (bool) - uses memory optimization techniques
     - typical_speed (str) - qualitative speed assessment

2. **Create Abstract Backend Interface:**
   - In `proxima/backends/base.py`, define `QuantumBackend(ABC)` abstract base class
   - Required abstract methods:
     - `get_name() -> str` - returns human-readable backend name
     - `get_id() -> str` - returns unique identifier (e.g., "lret-statevector")
     - `get_capabilities() -> Set[BackendCapability]` - returns supported capabilities
     - `get_metadata() -> CapabilityMetadata` - returns detailed capability info
     - `validate_circuit(circuit) -> ValidationResult` - checks if circuit is compatible
     - `estimate_resources(circuit, shots) -> ResourceEstimate` - predicts RAM/time requirements
     - `execute(circuit, shots, options) -> BackendResult` - runs simulation
     - `is_available() -> bool` - checks if backend dependencies are installed
     - `get_version() -> str` - returns backend library version
   - Optional methods with default implementations:
     - `supports_checkpoint() -> bool` - returns False by default
     - `create_checkpoint() -> CheckpointData` - raises NotImplementedError
     - `restore_checkpoint(data) -> None` - raises NotImplementedError

3. **Define Circuit Representation:**
   - Create unified circuit format in `proxima/backends/circuit.py`
   - Options:
     - **Option A:** Use OpenQASM 3.0 as interchange format (most universal)
     - **Option B:** Define custom `ProximaCircuit` class and converters
   - Recommendation: OpenQASM for maximum compatibility
   - Define `CircuitFormat` enum: OPENQASM3, QISKIT, CIRQ, NATIVE
   - Each backend adapter implements conversion to/from its native format

4. **Define Result Schema:**
   - In `proxima/backends/result.py`, create `BackendResult` dataclass:
     - `backend_id` (str) - which backend produced this result
     - `execution_time` (float) - wall-clock time in seconds
     - `memory_used` (int) - peak memory in bytes
     - `status` (ResultStatus) - SUCCESS, FAILED, TIMEOUT
     - `measurements` (Optional[dict]) - shot results if applicable
     - `statevector` (Optional[ndarray]) - final state if statevector sim
     - `density_matrix` (Optional[ndarray]) - final state if DM sim
     - `metadata` (dict) - backend-specific information
     - `error_message` (Optional[str]) - if status is FAILED
   - Create `ValidationResult` dataclass:
     - `valid` (bool)
     - `errors` (List[str]) - validation error messages
     - `warnings` (List[str]) - non-fatal issues
   - Create `ResourceEstimate` dataclass:
     - `estimated_memory_bytes` (int)
     - `estimated_time_seconds` (float)
     - `confidence` (str) - "low", "medium", "high"

5. **Create Result Normalizer:**
   - In `proxima/backends/normalizer.py`, define `ResultNormalizer` class
   - Method: `normalize(backend_result) -> NormalizedResult`
   - `NormalizedResult` includes:
     - Standardized measurement format (dict with bitstring keys, count values)
     - Fidelity metrics (if ground truth available)
     - Statistical summaries (mean, variance of measurement probabilities)
     - Execution metadata in uniform format
   - Handle different measurement formats from various backends
   - Convert complex numpy arrays to serializable formats

### Step 2.2: Implement Backend Registry

**Objective:** Create a dynamic registry that discovers, registers, and manages available quantum backends.

**Detailed Instructions:**

1. **Define Registry Data Structure:**
   - In `proxima/backends/registry.py`, create `BackendRegistry` singleton class
   - Internal storage: dict mapping backend_id to backend instance
   - Additional indexes:
     - By capability: dict mapping BackendCapability to list of backend_ids
     - By priority: sorted list of backend_ids

2. **Implement Registration Methods:**
   - `register(backend: QuantumBackend) -> None`
     - Validates backend implements required interface
     - Checks if backend is available (dependencies installed)
     - Stores backend instance with its ID as key
     - Updates capability and priority indexes
     - Logs registration with backend name and capabilities
   - `unregister(backend_id: str) -> None`
     - Removes backend from registry
     - Updates all indexes
   - `get_backend(backend_id: str) -> Optional[QuantumBackend]`
     - Retrieves backend by ID
     - Returns None if not found

3. **Implement Discovery Methods:**
   - `list_all() -> List[QuantumBackend]`
     - Returns all registered backends
   - `list_by_capability(capability: BackendCapability) -> List[QuantumBackend]`
     - Returns backends supporting given capability
   - `find_by_name(name: str) -> Optional[QuantumBackend]`
     - Fuzzy search by backend name
   - `get_default() -> QuantumBackend`
     - Returns highest priority available backend

4. **Implement Health Check System:**
   - `check_health(backend_id: str) -> HealthStatus`
     - Calls backend's `is_available()` method
     - Performs quick validation test (e.g., simulate 1-qubit circuit)
     - Returns HealthStatus with fields: available (bool), latency (float), error (Optional[str])
   - `check_all_health() -> Dict[str, HealthStatus]`
     - Runs health check on all registered backends
     - Returns map of backend_id to health status

5. **Auto-Discovery on Startup:**
   - In `proxima/backends/__init__.py`, implement `discover_backends()`
   - Scan for available backend libraries:
     - Try importing cirq, qiskit_aer, proxima_lret_adapter
     - For each successful import, instantiate corresponding adapter
     - Call `register()` on each adapter
   - Handle import errors gracefully (log warning, continue)
   - Called automatically during Proxima initialization

6. **Configuration Integration:**
   - Read backend configurations from `proxima.json` under `backends` section
   - For each configured backend:
     - Check if registered
     - Apply configuration options (enabled, priority, custom settings)
     - If enabled=false, skip registration or mark as disabled
   - Allow users to override backend priorities

### Step 2.3: Implement LRET Backend Adapter

**Objective:** Create the first concrete backend adapter for LRET quantum simulator with full integration.

**Detailed Instructions:**

1. **Choose Integration Method:**
   - **Option A:** Python bindings via pybind11 (requires LRET Python package)
   - **Option B:** Subprocess with JSON I/O (more flexible, easier to isolate)
   - Recommendation: Start with subprocess approach for robustness, add pybind11 later for performance
   - Check if LRET Python bindings (qlret) are available; use if present, else fall back to subprocess

2. **Create LRET Adapter Class:**
   - In `proxima/backends/adapters/lret_adapter.py`, define `LRETBackend(QuantumBackend)`
   - Implement all required methods from abstract base class
   - `get_id()` returns "lret-statevector" or "lret-density-matrix" depending on mode
   - `get_capabilities()` returns {STATEVECTOR, DENSITY_MATRIX, NOISE_MODELING, PARAMETRIC_CIRCUITS}

3. **Implement Circuit Conversion:**
   - LRET may use its own circuit format or accept OpenQASM
   - If using subprocess, define JSON schema for circuit representation:
     - `{"qubits": N, "gates": [{"type": "H", "qubits": [0]}, ...]}`
   - Create converter methods:
     - `openqasm_to_lret(qasm_str: str) -> dict` - parse QASM and convert to LRET format
     - Use pyqasm or manual parsing for QASM interpretation
   - If using Python bindings (qlret):
     - Convert to qlret.Circuit object
     - Use qlret gate methods directly

4. **Implement Resource Estimation:**
   - `estimate_resources(circuit, shots)`:
     - For LRET, estimate based on qubit count and expected rank
     - Memory ≈ O(2^n * rank) where rank typically grows linearly with depth
     - Provide conservative estimates with "medium" confidence
     - Use empirical lookup table from LRET benchmarks if available
   - Return ResourceEstimate object with estimates

5. **Implement Execution (Subprocess Method):**
   - `execute(circuit, shots, options)`:
     - Convert circuit to LRET JSON format
     - Write to temporary file or pass via stdin
     - Construct command: `quantum_sim --input circuit.json --shots {shots} --output results.json`
     - Add LRET-specific options from `options` dict (e.g., `--noise`, `--mode hybrid`)
     - Use subprocess.Popen with stdout/stderr capture
     - Monitor process with timeout (from configuration or default 30 minutes)
     - Parse output JSON when process completes
     - Handle errors (non-zero exit code, timeout, parse errors)
   - Convert LRET output format to BackendResult:
     - Extract measurements from results.json
     - Extract timing information
     - Extract final rank if available (store in metadata)
   - Return BackendResult object

6. **Implement Execution (Python Bindings Method):**
   - If qlret available:
     - Create qlret.Simulator instance
     - Convert ProximaCircuit to qlret.Circuit
     - Call `simulator.run(circuit, shots=shots)`
     - Extract results from simulator output
     - Measure execution time with time.perf_counter
   - Handle exceptions from qlret library
   - Wrap in BackendError if simulation fails

7. **Implement Availability Check:**
   - `is_available()`:
     - Check if LRET executable is in PATH (for subprocess) or qlret is importable (for bindings)
     - Try running `quantum_sim --version` and check output
     - Return True if successful, False otherwise
   - Cache availability result to avoid repeated checks

8. **Configuration Support:**
   - Read LRET-specific settings from `backends.lret` configuration section
   - Support options:
     - `executable_path` - custom path to quantum_sim binary
     - `default_mode` - "hybrid", "sequential", "row", "column"
     - `noise_model_path` - path to default noise model JSON
     - `truncation_threshold` - rank truncation threshold (default 1e-4)
   - Apply options to every execution

9. **Error Handling:**
   - Catch LRET-specific errors:
     - Executable not found → BackendError with installation instructions
     - Circuit incompatible → ValidationError with details
     - Simulation timeout → ExecutionError with partial results if available
   - Provide helpful error messages with suggestions

10. **Testing:**
    - Create `tests/backends/test_lret_adapter.py`
    - Test backend registration and metadata retrieval
    - Test circuit conversion (OpenQASM → LRET format)
    - Test execution with simple circuits (Hadamard, CNOT)
    - Mock subprocess calls for unit tests
    - Integration test with actual LRET if available in CI environment

### Step 2.4: Result Normalization and Validation

**Objective:** Ensure all backend outputs are converted to a consistent format for downstream processing.

**Detailed Instructions:**

1. **Define Measurement Format Standard:**
   - In `proxima/backends/formats.py`, define standard measurement format:
     - Bitstrings as keys (e.g., "00", "01", "10", "11")
     - Counts as integer values
     - Example: `{"00": 45, "01": 5, "10": 3, "11": 47}` for 100 shots
   - For statevector: complex amplitudes as list or numpy array
   - For density matrix: 2D complex array as nested list or numpy array

2. **Implement Measurement Normalizer:**
   - In `proxima/backends/normalizer.py`, extend `ResultNormalizer`
   - Method: `normalize_measurements(raw_measurements, backend_type) -> dict`
   - Handle different input formats:
     - LRET: may return dict with bitstrings or integer indices
     - Cirq: returns collections.Counter or Measurement objects
     - Qiskit: returns Counts object with hex or binary strings
   - Convert all to standard bitstring:count dict
   - Validate total counts match requested shots (within tolerance)

3. **Implement State Normalizer:**
   - Method: `normalize_statevector(raw_state, backend_type) -> ndarray`
   - Convert to numpy complex128 array
   - Validate normalization: sum of |amplitude|^2 should be 1.0 (within epsilon)
   - Method: `normalize_density_matrix(raw_dm, backend_type) -> ndarray`
   - Convert to numpy complex128 2D array
   - Validate trace: should be 1.0 (within epsilon)
   - Validate Hermitian property: DM = DM†

4. **Implement Metadata Normalizer:**
   - Method: `normalize_metadata(raw_metadata, backend_type) -> dict`
   - Extract common fields:
     - `execution_time_seconds` - wall-clock duration
     - `memory_peak_bytes` - maximum memory used
     - `backend_version` - library version string
   - Preserve backend-specific fields under `backend_specific` key
   - Example: for LRET, preserve `final_rank` and `truncation_count`

5. **Implement Validation:**
   - Method: `validate_result(backend_result) -> ValidationResult`
   - Checks:
     - Required fields are present (backend_id, status)
     - Measurements sum to expected shot count (if applicable)
     - State normalization within tolerance
     - Metadata fields have valid types and ranges
   - Return ValidationResult with any errors or warnings
   - Log validation warnings but don't fail execution

6. **Create Result Comparison Utilities:**
   - In `proxima/backends/comparison.py`, define comparison functions:
     - `compare_measurements(result1, result2) -> float`
       - Calculate statistical distance (e.g., total variation distance)
       - Return value between 0 (identical) and 1 (completely different)
     - `compare_statevectors(sv1, sv2) -> float`
       - Calculate fidelity: F = |⟨ψ1|ψ2⟩|^2
       - Return value between 0 (orthogonal) and 1 (identical)
     - `compare_density_matrices(dm1, dm2) -> float`
       - Calculate trace distance: D = (1/2) * Tr|ρ1 - ρ2|
       - Return value between 0 (identical) and 1 (orthogonal)
   - These will be used for multi-backend comparison feature

### Step 2.5: Integration and Testing

**Objective:** Ensure Backend Abstraction Layer works cohesively with Phase 1 infrastructure.

**Detailed Instructions:**

1. **Integration with Configuration:**
   - Backend settings from proxima.json are loaded at startup
   - Registry populates based on available backends and configuration
   - Test loading with various backend configurations (enabled/disabled, priorities)

2. **Integration with Logging:**
   - All backend operations are logged at appropriate levels
   - Backend registration logged at INFO
   - Execution start/end logged at INFO with duration
   - Backend errors logged at ERROR with full context
   - Debug mode logs circuit details and full results

3. **Integration with Error Handling:**
   - Backend errors are wrapped in appropriate ProximaError subclasses
   - Backend failures don't crash application, return graceful error messages
   - Validation errors provide actionable feedback to user

4. **Create Backend Selection Command:**
   - In TUI, add command: `select backend <backend_id>`
   - Command lists all available backends with capabilities
   - User can choose interactively or specify backend ID directly
   - Selected backend is stored in session state
   - Confirmation message displays selected backend info

5. **Create Backend Health Check Command:**
   - Add command: `check backends` or `health`
   - Runs health check on all registered backends
   - Displays table with columns: Backend, Status, Latency, Notes
   - Color-code status: green for available, red for unavailable, yellow for degraded

6. **Unit Tests:**
   - Test BackendRegistry registration and discovery
   - Test Backend interface compliance with mock backend
   - Test LRET adapter methods individually
   - Test result normalizer with various input formats
   - Test validation logic with valid and invalid results

7. **Integration Tests:**
   - Test end-to-end: register backends → select backend → execute simple circuit → receive normalized result
   - Test with missing backends (dependencies not installed)
   - Test with invalid circuit inputs
   - Test with simulated backend failures

8. **Manual Testing:**
   - Launch Proxima and run `check backends`, verify LRET appears if installed
   - Run `select backend lret-statevector`, verify confirmation
   - Prepare simple circuit (Hadamard + measure), simulate and verify result appears
   - Inspect logs to confirm backend operations are recorded

### Phase 2 Implementation Complete

**Artifacts Produced:**
- Abstract QuantumBackend interface
- BackendRegistry for dynamic backend management
- LRET backend adapter with full integration
- Result normalizer and validator
- Backend health check system
- Unified circuit and result formats

**Verification:**
- Backend registry correctly discovers and registers LRET
- Health check shows LRET as available
- Simple circuits execute through LRET adapter
- Results are normalized and validated
- Backend operations are logged correctly
- Tests pass with coverage >80%

---

## PHASE 3: EXECUTION CONTROL SYSTEM

### Step 3.1: Design and Implement State Machine

**Objective:** Create a robust finite state machine to manage all execution states and transitions with validation.

**Detailed Instructions:**

1. **Define Execution States:**
   - In `proxima/execution/states.py`, create `ExecutionState` enum:
     - IDLE - no execution in progress, ready for new commands
     - PLANNING - generating execution plan
     - VALIDATING - validating circuit and resources
     - READY - plan approved, waiting to start
     - EXECUTING - actively running simulation
     - PAUSED - execution suspended, can be resumed
     - PAUSING - in process of pausing
     - RESUMING - in process of resuming
     - ABORTING - in process of aborting
     - ABORTED - execution was aborted by user
     - ROLLING_BACK - reverting to previous checkpoint
     - COMPLETED - execution finished successfully
     - FAILED - execution encountered fatal error

2. **Define State Transition Rules:**
   - Create `StateTransition` dataclass with fields:
     - `from_state` (ExecutionState)
     - `to_state` (ExecutionState)
     - `trigger` (str) - event that causes transition (e.g., "user_start", "error_occurred")
     - `conditions` (List[Callable]) - optional preconditions that must be true
   - Define all valid transitions:
     - IDLE → PLANNING (trigger: user submits execution request)
     - PLANNING → VALIDATING (trigger: plan generation complete)
     - VALIDATING → READY (trigger: validation passed)
     - VALIDATING → FAILED (trigger: validation failed)
     - READY → EXECUTING (trigger: user confirms start)
     - EXECUTING → PAUSING (trigger: user requests pause)
     - PAUSING → PAUSED (trigger: checkpoint created)
     - PAUSED → RESUMING (trigger: user requests resume)
     - RESUMING → EXECUTING (trigger: state restored)
     - EXECUTING → ABORTING (trigger: user requests abort or timeout)
     - ABORTING → ABORTED (trigger: process terminated)
     - EXECUTING → COMPLETED (trigger: backend returns success)
     - EXECUTING → FAILED (trigger: backend returns error)
     - EXECUTING → ROLLING_BACK (trigger: user requests rollback)
     - ROLLING_BACK → PAUSED (trigger: rollback to checkpoint complete)
     - Any state → IDLE (trigger: reset/new session)

3. **Implement State Machine Class:**
   - In `proxima/execution/state_machine.py`, create `ExecutionStateMachine` class
   - Constructor initializes with state = IDLE
   - Method: `transition(trigger: str, **context) -> bool`
     - Looks up valid transition from current state with given trigger
     - Evaluates any preconditions
     - If valid, changes state and emits state change event
     - If invalid, raises InvalidStateTransition exception
     - Logs transition with old state, new state, trigger
     - Returns True if transition successful
   - Method: `get_state() -> ExecutionState`
     - Returns current state
   - Method: `can_transition(trigger: str) -> bool`
     - Checks if transition is valid from current state
     - Does not perform transition
   - Method: `reset()`
     - Returns to IDLE state
     - Clears all context

4. **Implement State Change Events:**
   - Use observer pattern or event emitter
   - When state changes, emit `StateChangedEvent` with fields:
     - `old_state` (ExecutionState)
     - `new_state` (ExecutionState)
     - `trigger` (str)
     - `timestamp` (datetime)
     - `context` (dict) - additional information
   - Components can subscribe to state change events
   - UI updates display based on state changes

5. **Add State Persistence:**
   - Method: `save_state() -> dict`
     - Serializes current state and context to dict
     - Includes state history (last N transitions)
   - Method: `load_state(state_dict: dict)`
     - Restores state from serialized format
     - Used for pause/resume and crash recovery

6. **Create State Validator:**
   - Method: `validate_state_consistency() -> List[str]`
     - Checks if current state is consistent with system state
     - Example checks:
       - If state is EXECUTING, backend process should be running
       - If state is PAUSED, checkpoint should exist
       - If state is IDLE, no execution context should be present
     - Returns list of inconsistencies found (empty if valid)

### Step 3.2: Implement Timer Service

**Objective:** Track execution duration at both global and stage-level granularity with live updates.

**Detailed Instructions:**

1. **Create Timer Class:**
   - In `proxima/execution/timer.py`, define `ExecutionTimer` class
   - Fields:
     - `start_time` (Optional[float]) - global execution start (unix timestamp)
     - `stage_timers` (Dict[str, StageTimer]) - timers for each stage
     - `current_stage` (Optional[str]) - name of currently active stage
     - `paused_time` (float) - cumulative time spent paused
     - `pause_start` (Optional[float]) - when current pause started

2. **Define StageTimer Class:**
   - Fields:
     - `name` (str) - stage identifier
     - `start_time` (float)
     - `end_time` (Optional[float])
     - `duration` (Optional[float]) - cached duration in seconds
     - `status` (StageStatus) - NOT_STARTED, RUNNING, COMPLETED, FAILED

3. **Implement Timer Methods:**
   - `start() -> None`
     - Records global start time
     - Resets all stage timers
     - Sets state to running
   - `stop() -> float`
     - Records end time
     - Returns total elapsed time (excluding paused time)
   - `pause() -> None`
     - Records pause start time
     - Current stage timer is paused
   - `resume() -> None`
     - Calculates duration paused, adds to paused_time
     - Resets pause_start
     - Current stage timer is resumed
   - `start_stage(stage_name: str) -> None`
     - Ends previous stage if any
     - Creates new StageTimer and marks as running
     - Updates current_stage
   - `end_stage(stage_name: str, status: StageStatus) -> None`
     - Finds stage timer by name
     - Records end time and calculates duration
     - Updates status
   - `get_elapsed() -> float`
     - Returns total time elapsed since start (minus paused time)
     - Calculated as: (current_time or end_time) - start_time - paused_time
   - `get_stage_elapsed(stage_name: str) -> Optional[float]`
     - Returns elapsed time for specific stage
     - If stage is running, returns time since stage start
     - If stage is complete, returns recorded duration

4. **Implement Live Timer Display:**
   - In TUI, create `TimerDisplay` widget
   - Shows:
     - Total elapsed time (formatted as HH:MM:SS or MM:SS.ms)
     - Current stage name and its elapsed time
     - Progress bar for stage (if estimable)
   - Update frequency: 10 Hz (every 100ms) for smooth display
   - Use Textual's reactive attributes for auto-updating

5. **Integration with State Machine:**
   - When state transitions to EXECUTING, automatically call `timer.start()`
   - When state transitions to PAUSED, call `timer.pause()`
   - When state transitions to RESUMING, call `timer.resume()`
   - When state transitions to COMPLETED or FAILED, call `timer.stop()`
   - Timer is reset when returning to IDLE

6. **Timer Persistence:**
   - Include timer state in checkpoint data
   - On pause, save all timer values
   - On resume, restore timer values and adjust for time spent paused
   - Ensure elapsed time calculation is correct after resume

7. **Stage Estimation (Optional):**
   - Optionally predict stage durations based on:
     - Historical data from similar executions
     - Resource estimates from backend
     - Heuristics (e.g., EXECUTE stage typically takes 90% of total time)
   - Show estimated completion time in UI
   - Update estimates in real-time as stages complete

### Step 3.3: Implement Process Manager

**Objective:** Manage backend process lifecycle including spawning, monitoring, signal handling, and cleanup.

**Detailed Instructions:**

1. **Create ProcessManager Class:**
   - In `proxima/execution/process_manager.py`, define `ProcessManager` class
   - Responsibilities:
     - Spawn backend processes
     - Monitor process health and resource usage
     - Send control signals (terminate, kill, suspend, resume)
     - Capture stdout/stderr
     - Handle timeouts
     - Cleanup on exit

2. **Implement Process Spawning:**
   - Method: `spawn(command: List[str], options: ProcessOptions) -> ProcessHandle`
   - ProcessOptions includes:
     - `timeout` (int) - maximum execution time in seconds
     - `cwd` (Path) - working directory
     - `env` (dict) - environment variables
     - `capture_output` (bool) - whether to capture stdout/stderr
   - Use subprocess.Popen for cross-platform compatibility
   - Set appropriate flags:
     - `stdout=subprocess.PIPE, stderr=subprocess.PIPE` if capturing output
     - `start_new_session=True` on Unix for proper signal handling
     - `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` on Windows
   - Return ProcessHandle with fields:
     - `pid` (int) - process ID
     - `process` (Popen) - subprocess object
     - `start_time` (float) - when process started

3. **Implement Process Monitoring:**
   - Method: `poll(handle: ProcessHandle) -> ProcessStatus`
   - Check if process is still running with `handle.process.poll()`
   - If still running, return RUNNING
   - If exited, return COMPLETED or FAILED based on exit code
   - Monitor resource usage with psutil:
     - Get process by PID
     - Read memory usage, CPU usage
     - Store in ProcessStatus object
   - Method: `wait(handle: ProcessHandle, timeout: Optional[float]) -> ProcessStatus`
     - Blocks until process completes or timeout
     - Returns final status

4. **Implement Timeout Handling:**
   - Start timer when process spawns
   - In monitoring loop or separate thread, check if elapsed time > timeout
   - If timeout exceeded:
     - Attempt graceful termination (SIGTERM on Unix, CTRL_C_EVENT on Windows)
     - Wait 5 seconds for process to exit
     - If still running, force kill (SIGKILL on Unix, TerminateProcess on Windows)
     - Return TIMEOUT status

5. **Implement Signal Handling:**
   - Method: `terminate(handle: ProcessHandle) -> None`
     - Send SIGTERM (Unix) or CTRL_BREAK_EVENT (Windows)
     - Set flag indicating user-requested termination
   - Method: `kill(handle: ProcessHandle) -> None`
     - Send SIGKILL (Unix) or TerminateProcess (Windows)
     - Immediate forceful termination
   - Method: `suspend(handle: ProcessHandle) -> None`
     - Send SIGSTOP (Unix only) to pause process
     - On Windows, use Job Objects or NtSuspendProcess (requires ctypes/win32 API)
   - Method: `resume(handle: ProcessHandle) -> None`
     - Send SIGCONT (Unix) to unpause process
     - On Windows, use NtResumeProcess

6. **Implement Output Capture:**
   - For captured processes, continuously read stdout/stderr
   - Use non-blocking reads or separate threads to avoid deadlock
   - Store output in rolling buffer (e.g., last 10000 lines)
   - Method: `get_output(handle: ProcessHandle) -> Tuple[str, str]`
     - Returns (stdout, stderr) as strings
   - Optionally stream output to log file in real-time

7. **Implement Cleanup:**
   - Method: `cleanup(handle: ProcessHandle) -> None`
     - Ensure process is terminated
     - Close file handles (stdout, stderr)
     - Delete temporary files if any
     - Remove from active process registry
   - Automatic cleanup on Python exit using atexit handlers
   - Clean up zombie processes

8. **Handle Multiple Processes:**
   - Support running multiple backends in parallel (for comparison feature)
   - Maintain registry of active ProcessHandles
   - Method: `list_active() -> List[ProcessHandle]`
   - Method: `terminate_all() -> None` - emergency shutdown

9. **Error Handling:**
   - Catch exceptions during process spawn (executable not found, permission denied)
   - Wrap in ProcessError with helpful message
   - Handle process crashes (unexpected exit)
   - Distinguish between normal exit, user termination, timeout, and crash

10. **Integration with State Machine:**
    - When state transitions to EXECUTING, spawn backend process
    - When state transitions to ABORTING, call `terminate()`
    - When state transitions to PAUSING, call `suspend()` (if supported)
    - When state transitions to RESUMING, call `resume()` (if supported)
    - Monitor process in background thread, update state on completion/failure

### Step 3.4: Implement Basic Start/Abort Functionality

**Objective:** Enable users to start execution and abort it mid-flight with proper cleanup.

**Detailed Instructions:**

1. **Create ExecutionController Class:**
   - In `proxima/execution/controller.py`, define `ExecutionController` class
   - Orchestrates state machine, timer, process manager
   - Constructor takes dependencies: state_machine, timer, process_manager, backend_registry
   - Maintains current execution context (backend, circuit, options)

2. **Implement Start Flow:**
   - Method: `start_execution(backend_id: str, circuit: Circuit, options: dict) -> ExecutionId`
   - Steps:
     1. Validate current state is READY
     2. Transition state to EXECUTING
     3. Start global timer
     4. Start stage: INITIALIZATION
     5. Get backend from registry
     6. Prepare execution command/call
     7. Spawn backend process via process manager
     8. Store process handle in execution context
     9. Start monitoring thread
     10. Return unique execution ID
   - Emit ExecutionStarted event with execution_id, backend_id, timestamp

3. **Implement Monitoring Thread:**
   - Background thread that:
     - Periodically polls process status (every 500ms)
     - Updates timer and progress
     - Checks for timeout
     - Checks for completion
     - Updates state machine on completion/failure
   - On completion:
     - Retrieve process output
     - Parse backend result
     - Transition state to COMPLETED
     - Emit ExecutionCompleted event
   - On failure:
     - Capture error output
     - Transition state to FAILED
     - Emit ExecutionFailed event

4. **Implement Abort Flow:**
   - Method: `abort_execution() -> None`
   - Steps:
     1. Validate current state is EXECUTING or PAUSED
     2. Transition state to ABORTING
     3. Call process_manager.terminate() with current process handle
     4. Wait up to 10 seconds for graceful termination
     5. If still running, call process_manager.kill()
     6. Stop timer
     7. Cleanup resources (temp files, etc.)
     8. Transition state to ABORTED
     9. Emit ExecutionAborted event
   - User receives confirmation: "Execution aborted. Partial results may be available."

5. **Implement User Commands:**
   - In TUI command parser, add commands:
     - `run <backend> <circuit>` - starts execution
     - `abort` - aborts current execution
   - For `run`:
     - Parse backend and circuit arguments
     - Validate inputs
     - Call ExecutionController.start_execution()
     - Display "Execution started..." message
   - For `abort`:
     - Confirm with user (unless --force flag)
     - Call ExecutionController.abort_execution()
     - Display "Aborting execution..." message

6. **Implement Status Display:**
   - In TUI main screen, show execution status panel:
     - Current state (from state machine)
     - Elapsed time (from timer)
     - Current stage
     - Backend being used
     - Progress indicator (spinner or progress bar)
   - Update reactively as state/timer changes
   - Color-code by state: green for running, red for failed, yellow for paused

7. **Implement Event Handlers in UI:**
   - Subscribe to execution events (Started, Completed, Failed, Aborted)
   - On ExecutionStarted:
     - Update status panel
     - Show notification: "Execution started with {backend}"
   - On ExecutionCompleted:
     - Update status panel
     - Show notification: "Execution completed in {duration}"
     - Display results
   - On ExecutionFailed:
     - Update status panel
     - Show error notification with error message
     - Offer retry or help options
   - On ExecutionAborted:
     - Update status panel
     - Show notification: "Execution aborted by user"

8. **Handle Cleanup on Exit:**
   - Register cleanup handlers with atexit
   - On Proxima exit (Ctrl+C, quit command):
     - Check if execution is in progress
     - If EXECUTING or PAUSED, prompt user:
       - "Execution in progress. Abort before quitting? (y/n)"
     - If user confirms, call abort_execution()
     - Wait for clean termination
     - Then exit Proxima

9. **Testing:**
   - Unit test state machine transitions for start/abort flows
   - Unit test timer behavior during execution
   - Unit test process manager spawn and terminate
   - Integration test: start execution → wait 2 seconds → abort → verify cleanup
   - Test with mock backend that sleeps for 60 seconds
   - Test timeout handling with mock backend that never completes

10. **Error Scenarios:**
    - Handle backend process crash during execution
    - Handle backend returning malformed results
    - Handle user requesting abort during ABORTING state (already aborting)
    - Handle start requested while execution already running (reject with error)

### Step 3.5: Integration and Testing

**Objective:** Ensure Execution Control System integrates properly with Backend Abstraction Layer and Phase 1 infrastructure.

**Detailed Instructions:**

1. **Integration with Backend Adapters:**
   - ExecutionController uses backend from registry
   - Passes circuit and options to backend.execute()
   - Handles BackendResult and errors from backend
   - Test with LRET adapter to ensure end-to-end flow works

2. **Integration with Configuration:**
   - Timeout values loaded from configuration
   - Default backend loaded from configuration
   - Execution options (shots, etc.) configurable

3. **Integration with Logging:**
   - All state transitions logged at INFO level
   - Execution start/end logged with backend, duration, status
   - Timer events logged at DEBUG level
   - Process manager actions logged (spawn, terminate, etc.)

4. **Create Execution History:**
   - Store completed executions in session database
   - Each execution record includes:
     - execution_id, backend_id, start_time, end_time, status
     - circuit (serialized), options, result (serialized)
   - Command to view history: `history` or `list executions`

5. **End-to-End Test:**
   - Full flow: Launch Proxima → select LRET → prepare circuit → run → wait for completion → view results
   - Full flow: Launch Proxima → run → abort mid-execution → verify cleanup
   - Verify state machine transitions correctly
   - Verify timer shows accurate elapsed time
   - Verify logs contain all expected entries

6. **Manual Testing:**
   - Run long execution (10+ seconds), observe timer updating in real-time
   - Abort during execution, verify immediate termination
   - Try starting new execution while one is running, verify rejection
   - Exit Proxima during execution, verify abort prompt appears

### Phase 3 Implementation Complete

**Artifacts Produced:**
- ExecutionStateMachine with all states and transitions
- ExecutionTimer with stage-level tracking
- ProcessManager for backend process lifecycle
- ExecutionController orchestrating entire execution flow
- Start and abort functionality fully operational
- Real-time status display in TUI

**Verification:**
- Executions start and complete successfully
- Timer accurately tracks execution duration
- Abort immediately terminates execution
- State transitions are correct and logged
- UI updates reflect execution status in real-time
- Tests pass with coverage >80%

---

## PHASE 4: RESOURCE MONITORING & FAIL-SAFE

### Step 4.1: Implement Resource Monitoring System

**Objective:** Create a continuous monitoring system that tracks system resources (RAM, CPU, GPU, disk) in real-time and detects potential exhaustion scenarios.

**Detailed Instructions:**

1. **Choose Resource Monitoring Library:**
   - Use `psutil` library for cross-platform system resource monitoring
   - Advantages: Python native, supports Windows/Linux/macOS, comprehensive metrics
   - Install as dependency: add `psutil>=5.9.0` to project requirements
   - Alternative for GPU monitoring: `GPUtil` for NVIDIA GPUs or `pynvml` (NVIDIA Management Library Python bindings)

2. **Create ResourceMonitor Class:**
   - In `proxima/monitoring/resource_monitor.py`, define `ResourceMonitor` class
   - Constructor parameters:
     - `update_interval` (float) - how often to sample resources (default 1.0 second)
     - `history_length` (int) - how many samples to retain (default 300 = 5 minutes at 1s interval)
   - Internal state:
     - `_running` (bool) - monitoring thread active flag
     - `_monitor_thread` (Optional[Thread]) - background monitoring thread
     - `_metrics_history` (deque) - ring buffer of historical metrics
     - `_current_metrics` (Optional[ResourceMetrics]) - most recent reading
     - `_callbacks` (List[Callable]) - threshold breach callbacks

3. **Define ResourceMetrics Dataclass:**
   - In `proxima/monitoring/types.py`, create `ResourceMetrics` dataclass with fields:
     - `timestamp` (float) - when measurement was taken
     - `ram_total_bytes` (int) - total system RAM
     - `ram_available_bytes` (int) - available RAM
     - `ram_used_bytes` (int) - used RAM
     - `ram_percent` (float) - percentage used (0-100)
     - `cpu_percent` (float) - overall CPU utilization (0-100)
     - `cpu_per_core` (List[float]) - per-core utilization
     - `disk_io_read_bytes` (int) - cumulative bytes read
     - `disk_io_write_bytes` (int) - cumulative bytes written
     - `swap_total_bytes` (int) - total swap space
     - `swap_used_bytes` (int) - used swap space
     - `swap_percent` (float) - swap usage percentage
     - `gpu_memory_used_bytes` (Optional[int]) - GPU memory if available
     - `gpu_memory_total_bytes` (Optional[int]) - total GPU memory
     - `gpu_utilization_percent` (Optional[float]) - GPU compute utilization

4. **Implement Metric Collection:**
   - Method: `_collect_metrics() -> ResourceMetrics`
   - Use psutil functions:
     - `psutil.virtual_memory()` returns object with total, available, used, percent
     - `psutil.cpu_percent(interval=0.1, percpu=True)` for CPU metrics (interval prevents blocking)
     - `psutil.swap_memory()` for swap metrics
     - `psutil.disk_io_counters()` for disk I/O
   - For GPU metrics (if NVIDIA GPU present):
     - Try importing GPUtil or pynvml
     - If available, query GPU 0 (or all GPUs): memory used/total, utilization
     - If not available or import fails, set GPU fields to None
   - Handle exceptions gracefully (e.g., permission denied on some metrics)
   - Create ResourceMetrics object with all collected values
   - Return metrics object

5. **Implement Monitoring Thread:**
   - Method: `start_monitoring() -> None`
     - Create and start daemon thread running `_monitor_loop()`
     - Set `_running = True`
   - Method: `_monitor_loop()`
     - While `_running`:
       - Collect metrics with `_collect_metrics()`
       - Store in `_current_metrics`
       - Append to `_metrics_history` (deque with maxlen handles overflow)
       - Check thresholds with `_check_thresholds()`
       - Sleep for `update_interval` seconds
   - Method: `stop_monitoring() -> None`
     - Set `_running = False`
     - Wait for thread to join with timeout
   - Ensure thread is daemon so it doesn't prevent Proxima exit

6. **Implement Threshold Checking:**
   - Method: `_check_thresholds(metrics: ResourceMetrics) -> List[ThresholdBreach]`
   - Load threshold configuration from `MonitoringConfig`
   - Defined thresholds (configurable):
     - `ram_warning_percent` (default 70.0) - warn user
     - `ram_caution_percent` (default 85.0) - request consent to continue
     - `ram_critical_percent` (default 95.0) - pause and require explicit force-continue
     - `swap_warning_percent` (default 50.0) - warn if swapping occurs
     - `cpu_sustained_threshold` (default 90.0 for 60 seconds) - warn if CPU pegged
   - Check current metrics against thresholds
   - For sustained thresholds, check recent history (last N samples)
   - Create `ThresholdBreach` objects for each breach:
     - `metric_name` (str) - which metric breached
     - `threshold_level` (str) - WARNING, CAUTION, CRITICAL
     - `current_value` (float) - current metric value
     - `threshold_value` (float) - threshold that was breached
     - `recommendation` (str) - suggested action
   - Invoke registered callbacks for each breach
   - Return list of breaches

7. **Implement Getter Methods:**
   - Method: `get_current_metrics() -> Optional[ResourceMetrics]`
     - Returns `_current_metrics` (most recent sample)
   - Method: `get_metrics_history(duration_seconds: float) -> List[ResourceMetrics]`
     - Filters history to last `duration_seconds` worth of samples
     - Returns chronologically sorted list
   - Method: `get_average_metrics(duration_seconds: float) -> ResourceMetrics`
     - Calculates average of each metric over specified duration
     - Useful for smoothing out spikes

8. **Implement Callback Registration:**
   - Method: `register_threshold_callback(callback: Callable[[ThresholdBreach], None]) -> None`
     - Adds callback to list
     - Callback will be invoked when threshold is breached
   - Method: `unregister_threshold_callback(callback) -> None`
     - Removes callback from list

9. **Implement Rate-of-Change Detection:**
   - Method: `get_ram_consumption_rate() -> float`
     - Looks at last N samples (e.g., 10 samples = 10 seconds)
     - Calculates linear regression slope of RAM usage over time
     - Returns bytes per second increase rate
     - Negative values indicate memory is being freed
   - Useful for predicting when memory will be exhausted
   - Method: `estimate_time_to_exhaustion() -> Optional[float]`
     - Uses consumption rate and available memory
     - Calculates: `time = available_bytes / consumption_rate`
     - Returns estimated seconds until RAM exhausted
     - Returns None if consumption rate is negative or zero

10. **Testing:**
    - Unit tests for metric collection (mock psutil functions)
    - Test threshold checking logic with various metric values
    - Test callback invocation on threshold breach
    - Test monitoring thread starts and stops cleanly
    - Integration test: start monitoring → trigger threshold → verify callback fired

### Step 4.2: Implement Threshold Configuration and Management

**Objective:** Allow users to configure resource thresholds and define actions for different threshold levels.

**Detailed Instructions:**

1. **Extend Configuration Schema:**
   - In `config/schema.json`, add `monitoring` section with properties:
     - `enabled` (boolean, default true) - enable resource monitoring
     - `update_interval_seconds` (number, default 1.0) - sampling frequency
     - `thresholds` (object) with nested properties:
       - `ram_warning_percent` (number, default 70, range 0-100)
       - `ram_caution_percent` (number, default 85, range 0-100)
       - `ram_critical_percent` (number, default 95, range 0-100)
       - `swap_warning_percent` (number, default 50, range 0-100)
       - `cpu_sustained_percent` (number, default 90, range 0-100)
       - `cpu_sustained_duration_seconds` (number, default 60)
     - `actions` (object) defining response to threshold breaches:
       - `on_warning` (string, enum: "notify", "ignore") - default "notify"
       - `on_caution` (string, enum: "notify", "request_consent", "ignore") - default "request_consent"
       - `on_critical` (string, enum: "pause", "abort", "request_consent") - default "pause"

2. **Create MonitoringConfig Pydantic Model:**
   - In `proxima/core/config.py`, define `MonitoringConfig` model:
     - Fields correspond to schema properties
     - Use Pydantic validators to ensure:
       - warning < caution < critical for RAM thresholds
       - percentages are in valid range 0-100
       - durations are positive
     - Nested models: `ThresholdConfig`, `ActionConfig`

3. **Implement Threshold Validation:**
   - Custom validator on `MonitoringConfig`:
     - Ensure threshold progression is logical (each higher severity has higher threshold)
     - Warn if thresholds are too close together (< 5% apart)
     - Validate that action for higher severity is at least as severe as lower

4. **Implement Dynamic Threshold Updates:**
   - Method in ResourceMonitor: `update_thresholds(new_config: ThresholdConfig) -> None`
     - Validates new threshold configuration
     - Updates internal threshold values
     - Logs threshold change at INFO level
     - Does not require restart of monitoring
   - Allow updating via command: `set threshold ram_warning 75`
   - Changes persist for session but require config file edit for permanence

5. **Implement Threshold Profile System:**
   - Define preset threshold profiles:
     - `conservative` - low thresholds, early warnings (warning=60%, caution=75%, critical=90%)
     - `balanced` - default thresholds (warning=70%, caution=85%, critical=95%)
     - `aggressive` - high thresholds, late warnings (warning=80%, caution=90%, critical=98%)
     - `custom` - user-defined thresholds
   - Method: `load_threshold_profile(profile_name: str) -> None`
     - Loads predefined threshold set
     - Updates ResourceMonitor configuration
   - Command: `load profile conservative`

6. **Implement Per-Backend Threshold Overrides:**
   - Allow specifying different thresholds for different backends
   - In backend configuration, optional `resource_thresholds` section
   - Example: LRET might have higher thresholds because it's memory-efficient
   - ResourceMonitor checks active backend and applies overrides if present

### Step 4.3: Implement User Consent Dialog System

**Objective:** Create modal dialogs in TUI that request explicit user consent for risky operations with clear explanations.

**Detailed Instructions:**

1. **Define ConsentRequest Dataclass:**
   - In `proxima/monitoring/consent.py`, create `ConsentRequest` dataclass:
     - `title` (str) - dialog title (e.g., "Resource Warning")
     - `message` (str) - detailed explanation of situation
     - `severity` (ConsentSeverity) - WARNING, CAUTION, CRITICAL
     - `options` (List[ConsentOption]) - available choices
     - `default_option` (str) - which option is selected by default
     - `additional_info` (Optional[str]) - extra context, risks, recommendations
     - `timeout_seconds` (Optional[float]) - auto-select default after timeout

2. **Define ConsentOption Dataclass:**
   - Fields:
     - `id` (str) - unique identifier (e.g., "continue", "abort", "force")
     - `label` (str) - display text (e.g., "Continue Anyway")
     - `description` (str) - explains what happens if selected
     - `dangerous` (bool) - marks risky options (displayed in red)

3. **Create ConsentDialog Widget:**
   - In `proxima/ui/widgets/consent_dialog.py`, create `ConsentDialog(Screen)` Textual widget
   - Layout:
     - Title bar with severity indicator (color-coded icon)
     - Message area (scrollable if long)
     - Additional info section (collapsible)
     - Options list with radio buttons or numbered choices
     - Button bar: Confirm (Enter), Cancel (Esc)
   - Styling:
     - WARNING severity: yellow border, info icon
     - CAUTION severity: orange border, warning icon
     - CRITICAL severity: red border, danger icon
   - Keyboard navigation:
     - Up/Down or j/k to select options
     - Enter to confirm selection
     - Escape to cancel (if allowed)
     - Number keys 1-9 for quick selection

4. **Implement ConsentManager Class:**
   - In `proxima/monitoring/consent_manager.py`, define `ConsentManager` singleton
   - Responsibilities:
     - Queue consent requests
     - Display dialog to user via TUI
     - Wait for user response
     - Return user's choice
     - Track consent history for session
   - Method: `request_consent(request: ConsentRequest) -> ConsentResponse`
     - Blocks until user responds or timeout
     - Returns `ConsentResponse` with fields:
       - `option_selected` (str) - ID of chosen option
       - `remember_for_session` (bool) - if user checked "don't ask again"
       - `timestamp` (datetime) - when consent was given

5. **Implement Consent History and Caching:**
   - Track previous consent decisions in session
   - If user selected "remember for session" option, cache decision
   - Method: `get_cached_consent(request_key: str) -> Optional[ConsentResponse]`
     - Request key is hash of title + message
     - Returns cached response if available and not expired
   - Cached consents cleared on:
     - Session reset
     - Explicit user command: `clear consent cache`
     - Proxima restart

6. **Implement Asynchronous Consent Flow:**
   - Since consent request may occur during execution, handle asynchronously
   - Use asyncio to pause execution, show dialog, wait for response, resume
   - Integration with execution state machine:
     - When consent needed during EXECUTING state
     - Transition to AWAITING_CONSENT state
     - Pause backend process (if pausable) or let it continue with monitoring
     - Show dialog
     - On user response, transition back to EXECUTING

7. **Implement Consent Logging:**
   - Every consent request and response is logged at INFO level
   - Log includes:
     - Request details (severity, message)
     - User's choice
     - Timestamp
     - Context (what was happening when consent requested)
   - Useful for auditing and debugging user decisions

8. **Create Standard Consent Templates:**
   - Pre-define common consent requests:
     - `RESOURCE_WARNING_CONSENT` - RAM approaching limits
     - `RESOURCE_CRITICAL_CONSENT` - RAM critically low
     - `TIMEOUT_EXTENSION_CONSENT` - execution taking longer than expected
     - `FORCE_EXECUTE_CONSENT` - hardware incompatible but user wants to try
     - `UNSAFE_OPERATION_CONSENT` - operation may cause system instability
   - Templates include pre-written messages and options
   - Method: `create_from_template(template_name: str, **kwargs) -> ConsentRequest`

### Step 4.4: Implement Fail-Safe Actions and Emergency Handling

**Objective:** Implement automatic and manual fail-safe mechanisms that protect the system and user from resource exhaustion or crashes.

**Detailed Instructions:**

1. **Define Fail-Safe Actions:**
   - In `proxima/monitoring/failsafe.py`, create `FailSafeAction` enum:
     - NOTIFY - show warning, continue execution
     - PAUSE - pause execution, await user decision
     - ABORT - immediately terminate execution
     - FORCE_GARBAGE_COLLECT - trigger Python garbage collection
     - REDUCE_PRECISION - suggest reducing simulation precision (if backend supports)
     - CHECKPOINT_AND_EXIT - save state and exit Proxima gracefully

2. **Create FailSafeHandler Class:**
   - Constructor takes dependencies: execution_controller, resource_monitor, consent_manager
   - Method: `handle_threshold_breach(breach: ThresholdBreach) -> None`
     - Determines appropriate action based on severity and configuration
     - Executes action
     - Logs action taken
   - Registered as callback with ResourceMonitor

3. **Implement Action: NOTIFY**
   - Display non-blocking notification in TUI
   - Use toast/notification widget (small popup that auto-dismisses)
   - Include: metric name, current value, threshold
   - Example: "RAM usage 72% (threshold: 70%)"
   - Log at WARNING level
   - Execution continues uninterrupted

4. **Implement Action: PAUSE**
   - Pause execution via ExecutionController
   - Create consent request explaining situation
   - Options:
     - "Continue" - resume execution, accept risk
     - "Abort" - terminate execution
     - "Reduce Load" - adjust parameters if possible
   - Wait for user decision synchronously
   - Based on choice, resume, abort, or reconfigure

5. **Implement Action: ABORT**
   - Immediately call ExecutionController.abort_execution()
   - Show notification: "Execution aborted due to critical resource shortage"
   - Attempt to save partial results if available
   - Log reason for abort at ERROR level
   - Offer post-abort actions:
     - View partial results
     - Retry with reduced parameters
     - View resource usage graph

6. **Implement Action: FORCE_GARBAGE_COLLECT**
   - Explicitly call Python's garbage collector: `gc.collect()`
   - May free memory from unreferenced objects
   - Measure RAM before and after, report freed amount
   - If freed > 100MB, log success at INFO
   - If freed < 10MB, log that GC was ineffective at DEBUG
   - Continue execution after GC

7. **Implement Action: CHECKPOINT_AND_EXIT**
   - Emergency graceful shutdown
   - Create execution checkpoint with current state
   - Save checkpoint to persistent storage
   - Display: "Critical resource shortage. State saved to {path}. Exiting."
   - Cleanly terminate backend process
   - Exit Proxima with exit code indicating emergency shutdown
   - On next launch, offer to resume from checkpoint

8. **Implement Hardware Compatibility Checker:**
   - Method: `check_hardware_compatibility(backend_id: str, circuit: Circuit) -> CompatibilityResult`
   - CompatibilityResult has fields:
     - `compatible` (bool)
     - `issues` (List[str]) - detected incompatibilities
     - `warnings` (List[str]) - potential problems
     - `recommendations` (List[str]) - suggested mitigations
   - Checks:
     - Estimated memory requirement vs available RAM
     - GPU availability if backend requires GPU
     - CPU instruction sets (e.g., AVX for some simulators)
     - Disk space for result files
   - Run before execution start (during VALIDATING state)

9. **Implement Force-Execute Option:**
   - If hardware check fails (compatible=False), show critical consent dialog:
     - Title: "Hardware Incompatibility Detected"
     - Message: Lists all issues and warnings
     - Options:
       - "Cancel" (default, safe choice)
       - "Force Execute Anyway" (dangerous, marked in red)
     - Additional info: "Forcing execution may cause system crash, data loss, or freeze. Proceed at your own risk."
   - If user selects "Force Execute Anyway":
     - Log user decision at WARNING level
     - Proceed with execution
     - Enable extra monitoring (shorter sample intervals)
     - Auto-abort if critical threshold breached
   - If user selects "Cancel":
     - Return to IDLE state
     - Suggest alternative backends or reduced parameters

10. **Implement Resource Pre-Allocation Check:**
    - Before spawning backend process, attempt to pre-allocate some memory
    - Create numpy array of estimated size (then delete)
    - If allocation fails, know for certain that execution will fail
    - Prevents starting execution that's guaranteed to crash
    - Method: `can_allocate_memory(bytes: int) -> bool`
    - Uses numpy.zeros() with try/except for MemoryError

11. **Implement Watchdog Timer:**
    - Secondary monitoring thread that watches for Proxima freeze
    - If main thread stops responding for > 30 seconds, assume deadlock
    - Watchdog actions:
      - Log emergency state at CRITICAL level
      - Dump stack traces of all threads
      - Attempt to kill backend process
      - Exit Proxima with error code
    - Implemented using threading.Timer or separate process

12. **Testing:**
    - Test threshold breach handling with simulated high RAM usage
    - Test consent dialog display and response capture
    - Test force-execute flow with mock incompatibility
    - Test emergency abort under simulated resource exhaustion
    - Test watchdog timer activation with artificial freeze

### Step 4.5: Integrate Resource Monitoring with Execution Flow

**Objective:** Ensure resource monitoring is active during executions and properly integrated with state machine and UI.

**Detailed Instructions:**

1. **Initialize ResourceMonitor at Startup:**
   - In Proxima main initialization (after config load):
     - Create ResourceMonitor instance with config
     - Create FailSafeHandler instance
     - Register FailSafeHandler as callback on ResourceMonitor
     - Start monitoring thread
   - Monitoring runs continuously, not just during executions

2. **Display Current Resource Usage in UI:**
   - In TUI header or status bar, add resource usage display
   - Show:
     - RAM: `RAM: 8.2/16.0 GB (51%)` with colored indicator
     - CPU: `CPU: 23%` with small bar graph
     - Optionally GPU if available
   - Update every 1-2 seconds (ResourceMonitor provides data)
   - Use Textual reactive attributes for auto-update
   - Color coding:
     - Green: < warning threshold
     - Yellow: >= warning, < caution
     - Orange: >= caution, < critical
     - Red: >= critical

3. **Pre-Execution Resource Validation:**
   - In ExecutionController, before transitioning READY → EXECUTING:
     - Get resource estimate from backend
     - Get current available resources from ResourceMonitor
     - Compare: `required_memory < available_memory * 0.8` (leave 20% buffer)
     - If insufficient:
       - Show warning dialog with option to continue or cancel
       - If user cancels, transition to IDLE
       - If user continues, proceed with extra monitoring
     - Run hardware compatibility check
     - Handle incompatibilities as per Step 4.4.9

4. **During-Execution Resource Monitoring:**
   - While state is EXECUTING:
     - ResourceMonitor continues sampling
     - FailSafeHandler reacts to threshold breaches
     - If PAUSE action triggered, execution pauses and awaits user consent
     - If ABORT action triggered, execution aborts immediately
   - Update UI with real-time resource graph (optional):
     - Small line chart showing RAM usage over last 60 seconds
     - Visible in status panel during execution
     - Use Textual's plotting capabilities or ASCII art graph

5. **Post-Execution Resource Report:**
   - After execution completes (COMPLETED or FAILED state):
     - Collect resource metrics from entire execution
     - Calculate statistics:
       - Peak RAM usage
       - Average CPU utilization
       - Total execution time vs. time spent in GC
     - Display in execution summary:
       - "Peak Memory: 12.4 GB (77% of system)"
       - "Average CPU: 68%"
       - "Memory-Efficient: Yes" (if stayed below caution threshold)
   - Store metrics with execution record in session database

6. **Implement Resource Usage History Command:**
   - Add command: `resources` or `memory`
   - Displays:
     - Current resource usage snapshot
     - Peak usage in current session
     - Historical graph (ASCII) for last 5-10 minutes
     - Threshold levels marked on graph
   - Example output:
     ```
     Current: RAM 6.8 GB (42%), CPU 15%
     Peak:    RAM 14.1 GB (88%), CPU 92%
     
     RAM Usage (Last 10 min):
     100% |                    ▄▄▄
      75% |            ▄▄▄▄▄▄▄███
      50% |   ▄▄▄▄▄▄▄▄███████████
      25% |▄▄▄███████████████████
       0% └──────────────────────
     ```

7. **Implement Resource Alerts Configuration:**
   - Allow users to enable/disable specific alerts
   - Command: `alerts disable ram_warning`
   - Persisted in configuration
   - Method in ResourceMonitor: `set_alert_enabled(alert_name: str, enabled: bool)`

8. **Cleanup on Exit:**
   - When Proxima exits, stop ResourceMonitor gracefully
   - Join monitoring thread with timeout
   - Save final resource report to log
   - If any resource leaks detected (high usage at exit), warn in log

9. **Integration Tests:**
   - End-to-end test: Start execution → simulate high RAM → verify pause triggered → user continues → execution completes
   - Test resource display updates in UI during execution
   - Test pre-execution validation rejects circuit requiring too much memory
   - Test force-execute flow with incompatible hardware

10. **Documentation:**
    - Add user guide section explaining:
      - How to interpret resource indicators
      - How to configure thresholds
      - What each alert level means
      - How force-execute works and its risks
      - How to reduce memory usage (parameter suggestions)

### Phase 4 Implementation Complete

**Artifacts Produced:**
- ResourceMonitor with continuous system metrics tracking
- Threshold configuration system with presets and per-backend overrides
- ConsentDialog widget for user consent requests
- FailSafeHandler with multiple protective actions
- Hardware compatibility checker with force-execute option
- Integrated resource display in TUI with real-time updates
- Resource usage history and reporting

**Verification:**
- ResourceMonitor correctly samples system metrics every second
- Threshold breaches trigger appropriate fail-safe actions
- Consent dialogs appear and block until user responds
- Pre-execution validation catches insufficient resources
- Force-execute flow allows override with explicit consent
- Resource usage displayed in UI updates in real-time
- All resource operations logged appropriately
- Tests pass with coverage >80%

---

## PHASE 5: ADDITIONAL BACKEND ADAPTERS

### Step 5.1: Implement Cirq Backend Adapters

**Objective:** Create backend adapters for both Cirq DensityMatrixSimulator and Cirq Simulator (statevector), ensuring full integration with the unified backend interface.

**Detailed Instructions:**

1. **Install Cirq Dependencies:**
   - Add `cirq-core>=1.3.0` to project requirements
   - Cirq is modular; cirq-core provides base functionality
   - Import as: `import cirq`
   - Verify installation with version check: `cirq.__version__`

2. **Create Cirq Base Adapter Class:**
   - In `proxima/backends/adapters/cirq_base.py`, define `CirqBaseAdapter(QuantumBackend)` abstract class
   - Shared functionality for both Cirq simulators:
     - Circuit conversion (ProximaCircuit/OpenQASM → cirq.Circuit)
     - Result normalization (cirq result objects → BackendResult)
     - Error handling (Cirq exceptions → ProximaError)
   - Abstract method: `_create_simulator()` - subclasses implement to return specific simulator

3. **Implement Circuit Conversion to Cirq:**
   - Method: `_convert_to_cirq_circuit(circuit) -> cirq.Circuit`
   - If input is OpenQASM string:
     - Use `cirq.contrib.qasm_import.circuit_from_qasm()` to parse QASM
     - Handle QASM 2.0 and 3.0 differences
     - Map standard gates (H, CNOT, RZ, etc.) to cirq equivalents
   - If input is custom ProximaCircuit:
     - Create cirq.Circuit with appropriate qubits
     - Use `cirq.LineQubit` or `cirq.GridQubit` for qubit representation
     - Iterate through gate list, converting each to cirq gate
     - Gate mapping:
       - Hadamard → `cirq.H`
       - CNOT → `cirq.CNOT`
       - Pauli X/Y/Z → `cirq.X`, `cirq.Y`, `cirq.Z`
       - Rotation gates → `cirq.Rx(angle)`, `cirq.Ry(angle)`, `cirq.Rz(angle)`
       - Measurement → `cirq.measure(qubit, key='result')`
   - Handle parameterized circuits:
     - Use `cirq.ParamResolver` for parameter substitution
     - Store parameter mappings for later binding

4. **Implement Cirq DensityMatrixSimulator Adapter:**
   - In `proxima/backends/adapters/cirq_dm_adapter.py`, define `CirqDensityMatrixAdapter(CirqBaseAdapter)`
   - Method: `get_id()` returns "cirq-density-matrix"
   - Method: `get_name()` returns "Cirq Density Matrix Simulator"
   - Method: `get_capabilities()` returns:
     - {DENSITY_MATRIX, NOISE_MODELING, MEASUREMENT_SAMPLING, PARAMETRIC_CIRCUITS}
   - Method: `get_metadata()`:
     - `max_qubits`: 10-12 (density matrix is memory-intensive)
     - `supports_parallel`: False (Cirq DM simulator is single-threaded)
     - `memory_efficient`: False
     - `typical_speed`: "medium"

5. **Implement DensityMatrixSimulator Execution:**
   - Method: `execute(circuit, shots, options) -> BackendResult`
   - Steps:
     1. Convert circuit to cirq.Circuit
     2. Create simulator: `simulator = cirq.DensityMatrixSimulator()`
     3. If noise model specified in options:
        - Apply noise using cirq noise models
        - Use `cirq.NoiseModel` or `cirq.ConstantQubitNoiseModel`
        - Apply depolarizing noise: `cirq.depolarize(p=noise_level)`
        - Combine circuit with noise: `noisy_circuit = circuit.with_noise(noise_model)`
     4. Set up measurement:
        - Ensure circuit has measurements
        - If not, add measurements to all qubits
     5. Run simulation:
        - If shots specified: `result = simulator.run(circuit, repetitions=shots)`
        - If shots=0 (density matrix only): `result = simulator.simulate(circuit)`
     6. Extract results:
        - Measurements: `result.histogram(key='result')` gives Counter with bitstrings
        - Density matrix: `result.final_density_matrix` (if simulate() was used)
     7. Measure execution time with `time.perf_counter()`
     8. Estimate memory used (Cirq doesn't provide directly, use psutil for process memory)
   - Handle errors:
     - `ValueError` from Cirq → ValidationError
     - Memory errors → ResourceError
     - Generic exceptions → BackendError with context

6. **Implement Cirq Statevector Simulator Adapter:**
   - In `proxima/backends/adapters/cirq_sv_adapter.py`, define `CirqStatevectorAdapter(CirqBaseAdapter)`
   - Method: `get_id()` returns "cirq-statevector"
   - Method: `get_name()` returns "Cirq Statevector Simulator"
   - Method: `get_capabilities()` returns:
     - {STATEVECTOR, MEASUREMENT_SAMPLING, PARAMETRIC_CIRCUITS}
     - Note: No NOISE_MODELING (statevector is noiseless by design)
   - Method: `get_metadata()`:
     - `max_qubits`: 20-25 (statevector is more memory-efficient than DM)
     - `supports_parallel`: False
     - `memory_efficient`: True (relative to density matrix)
     - `typical_speed`: "fast"

7. **Implement Statevector Simulator Execution:**
   - Method: `execute(circuit, shots, options) -> BackendResult`
   - Similar to DM execution but using `cirq.Simulator()` instead
   - Steps:
     1. Convert circuit to cirq.Circuit
     2. Create simulator: `simulator = cirq.Simulator()`
     3. Run simulation:
        - If shots > 0: `result = simulator.run(circuit, repetitions=shots)`
        - If shots=0: `result = simulator.simulate(circuit)`
     4. Extract results:
        - Measurements: `result.histogram(key='result')`
        - Statevector: `result.final_state_vector` (complex numpy array)
     5. Normalize statevector if needed
     6. Calculate fidelity with ideal state (if ground truth provided)
   - Statevector operations are faster than density matrix
   - Handle edge cases: empty circuits, circuits without measurements

8. **Implement Resource Estimation for Cirq:**
   - Method: `estimate_resources(circuit, shots) -> ResourceEstimate`
   - For DensityMatrixSimulator:
     - Memory: `16 * (2^n)^2 bytes` (complex128 matrix)
     - For n=10: ~16 MB, n=12: ~256 MB, n=15: ~32 GB
     - Time: O(n * 2^(2n)) per gate (rough estimate)
   - For Statevector Simulator:
     - Memory: `16 * 2^n bytes` (complex128 vector)
     - For n=10: ~16 KB, n=20: ~16 MB, n=25: ~512 MB
     - Time: O(n * 2^n) per gate
   - Add overhead for shots: minimal memory, linear time increase
   - Confidence: "medium" (based on theoretical analysis)

9. **Implement Cirq-Specific Options:**
   - Support options passed via `options` dict:
     - `noise_level` (float, 0-1): depolarizing noise probability
     - `seed` (int): random seed for reproducibility
     - `dtype` (str): "complex64" or "complex128" for precision
   - Apply options during simulator creation or execution
   - Document supported options in adapter metadata

10. **Testing:**
    - Unit test circuit conversion (ProximaCircuit → cirq.Circuit)
    - Test both simulators with simple circuits (Bell state, GHZ state)
    - Test measurement extraction and normalization
    - Test noise model application (DM simulator only)
    - Test parameterized circuit execution
    - Compare Cirq results with known analytical results
    - Integration test with backend registry

### Step 5.2: Implement Qiskit Aer Backend Adapters

**Objective:** Create backend adapters for Qiskit Aer with both density matrix and statevector simulation methods.

**Detailed Instructions:**

1. **Install Qiskit Aer Dependencies:**
   - Add `qiskit>=1.0.0` to requirements (core Qiskit)
   - Add `qiskit-aer>=0.13.0` to requirements (Aer simulator)
   - Import as: `from qiskit import QuantumCircuit, transpile`
   - Import Aer: `from qiskit_aer import AerSimulator`

2. **Create Qiskit Base Adapter Class:**
   - In `proxima/backends/adapters/qiskit_base.py`, define `QiskitBaseAdapter(QuantumBackend)`
   - Shared functionality:
     - Circuit conversion (OpenQASM/ProximaCircuit → qiskit.QuantumCircuit)
     - Result extraction from qiskit.Result objects
     - Noise model handling using qiskit_aer.noise
   - Abstract method: `_get_simulation_method()` returns "density_matrix" or "statevector"

3. **Implement Circuit Conversion to Qiskit:**
   - Method: `_convert_to_qiskit_circuit(circuit) -> QuantumCircuit`
   - If input is OpenQASM string:
     - Use `QuantumCircuit.from_qasm_str(qasm_string)` for QASM 2.0
     - Or `qasm3.loads(qasm3_string)` for QASM 3.0 (requires qiskit-qasm3-import)
   - If input is custom ProximaCircuit:
     - Create QuantumCircuit: `qc = QuantumCircuit(n_qubits)`
     - Iterate through gates, converting each:
       - Hadamard → `qc.h(qubit)`
       - CNOT → `qc.cx(control, target)`
       - Pauli gates → `qc.x()`, `qc.y()`, `qc.z()`
       - Rotation gates → `qc.rx(angle, qubit)`, etc.
       - Custom gates → define with `qc.unitary(matrix, qubits)`
     - Add measurements: `qc.measure_all()` or specific qubits
   - Handle classical registers for measurement storage

4. **Implement Qiskit Density Matrix Adapter:**
   - In `proxima/backends/adapters/qiskit_dm_adapter.py`, define `QiskitDensityMatrixAdapter(QiskitBaseAdapter)`
   - Method: `get_id()` returns "qiskit-aer-density-matrix"
   - Method: `get_name()` returns "Qiskit Aer Density Matrix"
   - Method: `get_capabilities()` returns:
     - {DENSITY_MATRIX, NOISE_MODELING, MEASUREMENT_SAMPLING, PARAMETRIC_CIRCUITS, GPU_ACCELERATION}
   - Method: `get_metadata()`:
     - `max_qubits`: 10-12 (CPU), 20+ (with GPU)
     - `supports_parallel`: True (Qiskit supports parallel shot execution)
     - `memory_efficient`: False
     - `typical_speed`: "medium" (CPU) or "fast" (GPU)

5. **Implement DensityMatrix Execution with Qiskit Aer:**
   - Method: `execute(circuit, shots, options) -> BackendResult`
   - Steps:
     1. Convert circuit to QuantumCircuit
     2. Create Aer simulator with density_matrix method:
        ```
        simulator = AerSimulator(method='density_matrix')
        ```
     3. Configure simulator options:
        - Set seed: `simulator.set_options(seed_simulator=seed)`
        - Enable GPU: `simulator.set_options(device='GPU')` if available
        - Set precision: `simulator.set_options(precision='double')`
     4. Apply noise model if specified:
        - Import: `from qiskit_aer.noise import NoiseModel, depolarizing_error`
        - Create noise: `noise_model = NoiseModel()`
        - Add errors: `noise_model.add_all_qubit_quantum_error(depolarizing_error(p, 1), ['u1', 'u2', 'u3'])`
        - Pass to run: `noise_model=noise_model`
     5. Transpile circuit for backend:
        - `transpiled = transpile(circuit, simulator, optimization_level=1)`
        - Ensures circuit uses simulator's native gate set
     6. Run simulation:
        - `job = simulator.run(transpiled, shots=shots, noise_model=noise_model)`
        - `result = job.result()`
     7. Extract results:
        - Counts: `result.get_counts()` returns dict with bitstrings
        - Density matrix: `result.data()['density_matrix']` (if save_density_matrix=True)
        - Execution time: `result.time_taken`
        - Memory: `result.metadata` may contain memory info
     8. Convert to BackendResult format
   - Handle Qiskit-specific exceptions (QiskitError, AerError)

6. **Implement Qiskit Statevector Adapter:**
   - In `proxima/backends/adapters/qiskit_sv_adapter.py`, define `QiskitStatevectorAdapter(QiskitBaseAdapter)`
   - Method: `get_id()` returns "qiskit-aer-statevector"
   - Method: `get_name()` returns "Qiskit Aer Statevector"
   - Method: `get_capabilities()` returns:
     - {STATEVECTOR, MEASUREMENT_SAMPLING, PARAMETRIC_CIRCUITS, GPU_ACCELERATION}
   - Method: `get_metadata()`:
     - `max_qubits`: 25-30 (CPU), 40+ (with GPU and sufficient VRAM)
     - `supports_parallel`: True
     - `memory_efficient`: True
     - `typical_speed`: "fast"

7. **Implement Statevector Execution with Qiskit Aer:**
   - Method: `execute(circuit, shots, options) -> BackendResult`
   - Similar to density matrix but with `method='statevector'`
   - Steps:
     1. Convert and transpile circuit
     2. Create simulator: `AerSimulator(method='statevector')`
     3. Configure options (seed, GPU, precision)
     4. Run simulation: `job = simulator.run(circuit, shots=shots)`
     5. Extract statevector: `result.get_statevector()` (Statevector object)
     6. Convert to numpy array: `sv.data` (complex array)
     7. Extract measurement counts if shots > 0
   - Statevector method is faster and more memory-efficient
   - GPU acceleration significantly speeds up large circuits

8. **Implement GPU Detection and Configuration:**
   - Method: `_detect_gpu() -> bool`
   - Try to create AerSimulator with device='GPU'
   - If successful, GPU is available
   - If raises exception, fall back to CPU
   - Store GPU availability in metadata
   - Method: `_configure_gpu_options()`
     - Set GPU memory limit if specified in config
     - Set number of GPUs to use (for multi-GPU systems)
     - Configure CUDA streams and blocks

9. **Implement Noise Model Integration:**
   - Support loading noise models from:
     - IBM device calibration data (via qiskit-ibm-runtime)
     - Custom JSON noise specifications
     - Predefined noise models (depolarizing, thermal, etc.)
   - Method: `_load_noise_model(noise_spec) -> NoiseModel`
   - Handle different noise formats:
     - Depolarizing: single error probability
     - Thermal: T1 and T2 relaxation times
     - Readout: measurement error probabilities
     - Crosstalk: multi-qubit correlated errors
   - Apply noise model to execution

10. **Implement Resource Estimation for Qiskit Aer:**
    - Method: `estimate_resources(circuit, shots) -> ResourceEstimate`
    - For density_matrix method:
      - Memory: `16 * (2^n)^2 bytes` (same as Cirq)
      - Add overhead for Aer's internal structures (~20%)
    - For statevector method:
      - Memory: `16 * 2^n bytes`
      - GPU reduces memory pressure on CPU but uses VRAM
    - Time estimates based on Aer benchmarks
    - Account for transpilation overhead (~5-10% of total time)
    - Confidence: "high" (Aer provides timing metadata)

11. **Testing:**
    - Test both simulation methods with identical circuits
    - Compare results with Cirq for consistency
    - Test GPU acceleration if available (verify speedup)
    - Test noise model application and effects
    - Test IBM device noise model import (if credentials available)
    - Test large circuits (20+ qubits on statevector)
    - Performance benchmarks comparing CPU vs GPU

### Step 5.3: Backend Configuration and Selection Extensions

**Objective:** Extend configuration system and selection UI to support multiple backends with per-backend settings.

**Detailed Instructions:**

1. **Extend Configuration Schema for New Backends:**
   - In `config/schema.json`, under `backends` section:
     - Add `cirq_density_matrix` object with properties:
       - `enabled` (boolean, default true)
       - `priority` (int, 1-10, default 5)
       - `default_noise_level` (number, 0-1, default 0)
       - `dtype` (string, enum ["complex64", "complex128"], default "complex128")
     - Add `cirq_statevector` object (similar properties)
     - Add `qiskit_dm` object with properties:
       - `enabled` (boolean)
       - `priority` (int)
       - `use_gpu` (boolean, default false)
       - `gpu_memory_limit_mb` (int, optional)
       - `optimization_level` (int, 0-3, default 1)
       - `noise_model_path` (string, optional path to JSON)
     - Add `qiskit_sv` object (similar properties)

2. **Update Backend Registry Auto-Discovery:**
   - In `proxima/backends/__init__.py`, enhance `discover_backends()`:
     - Try importing cirq:
       - If successful, instantiate CirqDensityMatrixAdapter and CirqStatevectorAdapter
       - Register both with registry
     - Try importing qiskit_aer:
       - If successful, instantiate both Qiskit adapters
       - Test GPU availability for Qiskit adapters
       - Register with appropriate GPU metadata
     - Log each successful registration at INFO level
     - Warn if import fails but don't crash

3. **Implement Backend Comparison View in TUI:**
   - Create `BackendComparisonScreen(Screen)` in `proxima/ui/screens/backend_comparison.py`
   - Display table with columns:
     - Backend Name
     - Type (DM or SV)
     - Max Qubits
     - Supports Noise
     - GPU Available
     - Status (Available/Unavailable)
     - Priority
   - Allow sorting by any column
   - Keyboard shortcuts:
     - Enter: Select highlighted backend
     - e: Edit backend configuration
     - t: Test backend with sample circuit
     - h: Show detailed backend info

4. **Implement Backend Testing Utility:**
   - Method in backend adapters: `run_self_test() -> TestResult`
   - Runs simple test circuit (2-qubit Bell state)
   - Verifies result is correct (50/50 |00⟩ and |11⟩)
   - Measures execution time and memory
   - Returns TestResult with:
     - `success` (bool)
     - `latency_ms` (float)
     - `memory_mb` (float)
     - `errors` (List[str]) if failed
   - Command: `test backend cirq-statevector`
   - Displays test results in UI

5. **Implement Backend Health Monitoring:**
   - Extend health check to run periodically (every 5 minutes)
   - Detect if backend becomes unavailable (e.g., GPU driver issue)
   - Update backend status in real-time
   - Notify user if selected backend becomes unavailable
   - Automatically fallback to alternative backend if needed

6. **Create Backend Recommendation System:**
   - Method in BackendRegistry: `recommend_backend(circuit, requirements) -> List[Tuple[str, float]]`
   - Scores each backend based on:
     - Circuit qubit count vs backend max_qubits (must fit)
     - Need for noise modeling (if required, filter out statevector)
     - Available memory vs estimated requirement
     - Historical performance for similar circuits
     - User priority settings
   - Returns list of (backend_id, score) tuples sorted by score
   - Display recommendation with explanation in UI

### Step 5.4: Result Normalization for All Backends

**Objective:** Ensure consistent result format across all five backend implementations (LRET, Cirq DM, Cirq SV, Qiskit DM, Qiskit SV).

**Detailed Instructions:**

1. **Extend Result Normalizer for Cirq:**
   - In `proxima/backends/normalizer.py`, add methods:
     - `normalize_cirq_measurements(cirq_histogram) -> dict`
       - Input: collections.Counter from Cirq
       - Convert Counter keys to standard bitstrings
       - Handle different key formats (tuples, ints, strings)
     - `normalize_cirq_statevector(cirq_sv) -> ndarray`
       - Input: numpy array from Cirq
       - Ensure complex128 dtype
       - Verify normalization
     - `normalize_cirq_density_matrix(cirq_dm) -> ndarray`
       - Input: 2D numpy array
       - Verify trace and Hermiticity

2. **Extend Result Normalizer for Qiskit:**
   - Add methods:
     - `normalize_qiskit_counts(qiskit_counts) -> dict`
       - Input: Counts object or dict from Qiskit
       - Keys may be hex ('0x3') or binary ('11')
       - Convert all to binary strings
       - Handle endianness (Qiskit is little-endian)
     - `normalize_qiskit_statevector(qiskit_sv) -> ndarray`
       - Input: Statevector object
       - Extract data array
       - Convert to standard format
     - `normalize_qiskit_density_matrix(qiskit_dm) -> ndarray`
       - Input: DensityMatrix object or array
       - Extract and normalize

3. **Implement Unified Result Validator:**
   - Method: `validate_normalized_result(result: BackendResult) -> ValidationResult`
   - Checks applicable to all backends:
     - If measurements present: sum of counts == shots
     - If statevector present: norm == 1.0 (within 1e-6)
     - If density matrix present: trace == 1.0, Hermitian
     - Execution time is positive
     - Backend ID is valid
   - Backend-specific checks:
     - DM backends: density_matrix field populated
     - SV backends: statevector field populated
     - LRET: metadata contains rank information
   - Return validation result with warnings/errors

4. **Implement Cross-Backend Consistency Tests:**
   - Test suite that runs same circuit on all available backends
   - Compare normalized results:
     - Measurement distributions should be statistically similar
     - Statevectors should have high fidelity (>0.999)
     - Density matrices should have low trace distance (<0.01)
   - Identify discrepancies and investigate
   - Used for regression testing when updating backends

5. **Create Result Archiving System:**
   - After normalization, store results in session database
   - Schema: execution_results table with columns:
     - id, execution_id, backend_id, timestamp
     - circuit_hash (to identify identical circuits)
     - measurements_json, statevector_blob, density_matrix_blob
     - metadata_json, status
   - Blob storage for large arrays (compressed with zlib or blosc)
   - Query interface: `get_results_by_circuit(circuit_hash) -> List[BackendResult]`

### Step 5.5: Integration and Testing

**Objective:** Verify all five backends work cohesively and produce consistent results.

**Detailed Instructions:**

1. **Integration with Execution Controller:**
   - Execution controller can now invoke any of five backends
   - Backend selection passed to execute method
   - Results normalized regardless of backend
   - Errors handled uniformly

2. **Create Multi-Backend Test Suite:**
   - Test case: Bell state circuit
     - Run on all backends
     - Verify measurements: ~50% |00⟩, ~50% |11⟩
     - Verify statevector: (|00⟩ + |11⟩) / √2
   - Test case: GHZ state (3 qubits)
     - Verify measurements: 50% |000⟩, 50% |111⟩
   - Test case: Random circuit (8 qubits, depth 10)
     - Compare results across backends (fidelity >0.99)
   - Test case: Noisy circuit (depolarizing noise)
     - Only DM backends, compare Cirq vs Qiskit

3. **Performance Benchmarking:**
   - Benchmark script: runs identical circuits on all backends
   - Metrics: execution time, memory usage, result accuracy
   - Generate performance report (CSV or JSON)
   - Display benchmark results in table format
   - Command: `benchmark --qubits 10 --depth 20 --shots 1000`

4. **Create Backend Showcase in Documentation:**
   - Document each backend with:
     - Capabilities and limitations
     - Typical use cases
     - Performance characteristics
     - Configuration options
     - Example commands
   - Include comparison table
   - Link to official backend documentation

5. **User Guide for Backend Selection:**
   - When to use each backend:
     - LRET: Noisy circuits, memory-constrained systems
     - Cirq DM: General density matrix simulation, custom noise
     - Cirq SV: Fast statevector, no noise
     - Qiskit DM: IBM device noise models, GPU acceleration
     - Qiskit SV: Large statevector, GPU acceleration
   - How to configure each backend
   - Troubleshooting common issues

6. **Final Integration Tests:**
   - End-to-end: User selects Cirq backend → runs circuit → views results
   - Test backend switching mid-session
   - Test with unavailable backends (e.g., Qiskit not installed)
   - Test error recovery when backend fails
   - Test GPU backends on systems without GPU

### Phase 5 Implementation Complete

**Artifacts Produced:**
- Cirq DensityMatrix and Statevector adapters
- Qiskit Aer DensityMatrix and Statevector adapters
- Extended backend configuration schema
- Enhanced backend selection UI
- Cross-backend result normalization
- Multi-backend test suite and benchmarks

**Verification:**
- All five backends (LRET + 2 Cirq + 2 Qiskit) register successfully
- Backend selection UI shows all available backends
- Each backend executes circuits and returns normalized results
- Results from different backends are statistically consistent
- GPU acceleration works for Qiskit backends (if GPU available)
- Configuration options apply correctly to each backend
- Tests pass with coverage >80%

---

## PHASE 6: LLM INTEGRATION HUB

### Step 6.1: Design LLM Provider Abstraction

**Objective:** Create a unified interface for interacting with multiple LLM providers (remote and local) with consistent API.

**Detailed Instructions:**

1. **Define LLM Provider Interface:**
   - In `proxima/intelligence/llm_provider.py`, create abstract base class `LLMProvider(ABC)`
   - Required abstract methods:
     - `get_provider_id() -> str` - unique identifier (e.g., "openai", "anthropic")
     - `get_provider_name() -> str` - human-readable name
     - `get_available_models() -> List[ModelInfo]` - list of models this provider offers
     - `is_available() -> bool` - check if provider is accessible (API key valid, server reachable)
     - `generate_completion(prompt: str, options: CompletionOptions) -> CompletionResult` - main generation method
     - `estimate_tokens(text: str) -> int` - estimate token count
     - `estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float` - estimate cost in USD
   - Optional methods with default implementations:
     - `supports_streaming() -> bool` - default False
     - `generate_completion_stream(prompt, options) -> Iterator[str]` - streaming generation
     - `supports_vision() -> bool` - default False
     - `supports_function_calling() -> bool` - default False

2. **Define Supporting Data Structures:**
   - In `proxima/intelligence/types.py`, create dataclasses:
     - `ModelInfo`:
       - `model_id` (str) - identifier used in API calls
       - `display_name` (str) - user-friendly name
       - `context_window` (int) - max tokens in context
       - `max_output_tokens` (int) - max tokens in response
       - `cost_per_million_input` (float) - pricing
       - `cost_per_million_output` (float)
       - `capabilities` (Set[str]) - features like "vision", "function_calling"
     - `CompletionOptions`:
       - `model` (str) - which model to use
       - `temperature` (float, 0-2, default 0.7) - randomness
       - `max_tokens` (int) - response length limit
       - `top_p` (float, 0-1, default 1.0) - nucleus sampling
       - `stop_sequences` (List[str]) - sequences that stop generation
       - `system_prompt` (Optional[str]) - system message
       - `timeout` (float, default 30) - request timeout in seconds
     - `CompletionResult`:
       - `text` (str) - generated text
       - `model_used` (str) - actual model used
       - `tokens_input` (int) - tokens in prompt
       - `tokens_output` (int) - tokens in completion
       - `finish_reason` (str) - "stop", "length", "error"
       - `metadata` (dict) - provider-specific info
       - `cost_usd` (Optional[float]) - estimated cost

3. **Create LLM Provider Registry:**
   - In `proxima/intelligence/provider_registry.py`, define `LLMProviderRegistry` singleton
   - Similar to BackendRegistry but for LLM providers
   - Methods:
     - `register(provider: LLMProvider)`
     - `get_provider(provider_id: str) -> Optional[LLMProvider]`
     - `list_providers() -> List[LLMProvider]`
     - `list_available_providers() -> List[LLMProvider]` - only those accessible
     - `get_default_provider() -> LLMProvider`
   - Auto-discover providers at startup

4. **Implement Token Counting Utilities:**
   - Different providers use different tokenizers
   - Use tiktoken for OpenAI: `tiktoken.encoding_for_model(model_name)`
   - Use Anthropic tokenizer API for Claude
   - For local models, use model-specific tokenizer via transformers library
   - Fallback: estimate as `len(text) / 4` (rough approximation)
   - Method: `get_tokenizer(provider_id: str, model: str) -> Tokenizer`

### Step 6.2: Implement Remote LLM Providers

**Objective:** Integrate major remote LLM providers with API key authentication and proper error handling.

**Detailed Instructions:**

1. **Implement OpenAI Provider:**
   - In `proxima/intelligence/providers/openai_provider.py`, define `OpenAIProvider(LLMProvider)`
   - Install dependency: `openai>=1.0.0` (official SDK)
   - Configuration:
     - Read API key from config: `providers.openai.api_key` or env var `OPENAI_API_KEY`
     - Optional: custom base URL for OpenAI-compatible endpoints
   - Initialize client: `from openai import OpenAI; client = OpenAI(api_key=key)`
   - Method: `get_available_models()`:
     - Return list of supported models:
       - gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
       - gpt-3.5-turbo
     - Include context windows and pricing from hardcoded data (updated periodically)
   - Method: `is_available()`:
     - Try making test API call: `client.models.list()`
     - Return True if successful, False if API key invalid or network error
   - Method: `generate_completion(prompt, options)`:
     - Construct messages: `[{"role": "user", "content": prompt}]`
     - If system_prompt provided, prepend system message
     - Call API: `response = client.chat.completions.create(model=..., messages=..., temperature=...)`
     - Extract response text: `response.choices[0].message.content`
     - Extract token counts: `response.usage.prompt_tokens`, `response.usage.completion_tokens`
     - Calculate cost using known pricing
     - Handle errors:
       - `openai.AuthenticationError` → LLMError("Invalid API key")
       - `openai.RateLimitError` → LLMError("Rate limit exceeded, retry after delay")
       - `openai.APITimeoutError` → LLMError("Request timeout")
     - Return CompletionResult

2. **Implement Anthropic Provider:**
   - In `proxima/intelligence/providers/anthropic_provider.py`, define `AnthropicProvider(LLMProvider)`
   - Install: `anthropic>=0.18.0`
   - Initialize: `from anthropic import Anthropic; client = Anthropic(api_key=key)`
   - Available models:
     - claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3.5-sonnet
     - Include model capabilities (vision support for certain models)
   - Method: `generate_completion(prompt, options)`:
     - Anthropic API structure is slightly different:
       - System prompt separate parameter
       - Messages: `[{"role": "user", "content": prompt}]`
     - Call: `response = client.messages.create(model=..., system=..., messages=..., max_tokens=...)`
     - Extract text: `response.content[0].text`
     - Token usage: `response.usage.input_tokens`, `response.usage.output_tokens`
     - Handle specific exceptions (anthropic.APIError variants)
   - Anthropic has different token counting (use their count_tokens API)

3. **Implement Google Gemini Provider:**
   - In `proxima/intelligence/providers/gemini_provider.py`, define `GeminiProvider(LLMProvider)`
   - Install: `google-generativeai>=0.3.0`
   - Initialize: `import google.generativeai as genai; genai.configure(api_key=key)`
   - Available models: gemini-pro, gemini-pro-vision
   - Method: `generate_completion(prompt, options)`:
     - Create model: `model = genai.GenerativeModel(model_name)`
     - Generate: `response = model.generate_content(prompt, generation_config={...})`
     - Extract text: `response.text`
     - Token counts may not be directly available, use tiktoken estimate
   - Handle safety settings and content filtering

4. **Implement Groq Provider:**
   - In `proxima/intelligence/providers/groq_provider.py`, define `GroqProvider(LLMProvider)`
   - Groq uses OpenAI-compatible API
   - Install: `openai` (reuse client)
   - Custom base URL: `https://api.groq.com/openai/v1`
   - Available models: llama3-70b, mixtral-8x7b, gemma-7b
   - Implementation similar to OpenAI but with different model list
   - Very fast inference, good for latency-sensitive applications

5. **Implement OpenRouter Provider:**
   - In `proxima/intelligence/providers/openrouter_provider.py`, define `OpenRouterProvider(LLMProvider)`
   - OpenRouter aggregates many models from various providers
   - Uses OpenAI-compatible API
   - Base URL: `https://openrouter.ai/api/v1`
   - Available models: query from API or hardcode popular ones
   - Special header: `HTTP-Referer` and `X-Title` for tracking
   - Pricing varies by model, query from OpenRouter's pricing API

6. **Implement Provider-Specific Features:**
   - Function calling (OpenAI, Anthropic):
     - Extend CompletionOptions with `functions` parameter
     - Define function schemas
     - Handle function call responses
   - Vision (Anthropic Claude 3, Gemini):
     - Support passing images in prompts
     - Extend prompt structure to include image data
   - Streaming:
     - Implement `generate_completion_stream()` for each provider
     - Yield chunks as they arrive
     - Update UI progressively

7. **Implement Retry and Rate Limiting:**
   - Wrap API calls with retry logic using tenacity library
   - Exponential backoff for transient errors
   - Respect rate limits (429 errors)
   - Config: `max_retries` (default 3), `retry_delay_seconds` (default 1)
   - Different retry strategies per provider

8. **Testing Remote Providers:**
   - Unit tests with mocked API responses
   - Integration tests with real API calls (use test mode or sandbox keys)
   - Test error handling (invalid key, rate limits, timeouts)
   - Test cost estimation accuracy
   - Test token counting accuracy (compare with provider's count)

### Step 6.3: Implement Local LLM Support

**Objective:** Enable Proxima to use locally-installed LLM servers for privacy, cost savings, and offline operation.

**Detailed Instructions:**

1. **Implement Ollama Provider:**
   - In `proxima/intelligence/providers/ollama_provider.py`, define `OllamaProvider(LLMProvider)`
   - Ollama runs local LLMs via simple HTTP API
   - Default URL: `http://localhost:11434`
   - No API key required
   - Method: `is_available()`:
     - Try GET request to `/api/tags` endpoint
     - Returns list of installed models
     - If successful, Ollama is running
   - Method: `get_available_models()`:
     - Query `/api/tags` to get list of local models
     - Parse response JSON: `models` array with name, size, modified date
     - Create ModelInfo for each (context windows from known model specs)
   - Method: `generate_completion(prompt, options)`:
     - POST to `/api/generate` with JSON body:
       - `model`: model name
       - `prompt`: user prompt
       - `options`: temperature, top_p, etc.
       - `stream`: false for non-streaming
     - Response: JSON with `response` field containing generated text
     - No token counts provided by Ollama, estimate locally
     - No cost (free local execution)
   - Handle errors: connection refused (Ollama not running), model not found

2. **Implement LM Studio Provider:**
   - In `proxima/intelligence/providers/lmstudio_provider.py`, define `LMStudioProvider(LLMProvider)`
   - LM Studio provides OpenAI-compatible API
   - Default URL: `http://localhost:1234/v1`
   - Use openai library with custom base_url
   - Similar implementation to OpenAI provider
   - Method: `get_available_models()`:
     - Query `/v1/models` endpoint
     - Returns list of loaded models
   - Load any GGUF model from LM Studio's library

3. **Implement llama.cpp Server Provider:**
   - In `proxima/intelligence/providers/llamacpp_provider.py`, define `LlamaCppProvider(LLMProvider)`
   - llama.cpp server provides HTTP API
   - Default URL: `http://localhost:8080`
   - Endpoints: `/completion` for generation
   - Custom JSON structure, not OpenAI-compatible
   - POST body: `{"prompt": "...", "n_predict": 100, "temperature": 0.7}`
   - Response: `{"content": "generated text", "tokens_evaluated": N}`
   - Support various llama.cpp models (Llama 2, Mistral, etc.)

4. **Implement text-generation-webui Provider:**
   - In `proxima/intelligence/providers/textgen_provider.py`, define `TextGenProvider(LLMProvider)`
   - text-generation-webui (oobabooga) has multiple API modes
   - Support OpenAI-compatible API mode (recommended)
   - Default URL: `http://localhost:5000/v1`
   - Also support native API: `/api/v1/generate`
   - Can load HuggingFace models on demand

5. **Implement Local vs Remote Distinction:**
   - Add property to LLMProvider: `is_local() -> bool`
   - Remote providers return False
   - Local providers return True
   - UI displays indicators: 🌐 for remote, 💻 for local
   - In consent dialogs, clearly distinguish: "Using local LLM (no data sent online)" vs "Using remote LLM (data sent to {provider})"

6. **Implement Model Auto-Discovery for Local Providers:**
   - On startup or on-demand, query each local provider
   - Discover what models are available
   - Update provider registry with current model list
   - Command: `refresh local models` - re-scans local servers
   - Show notification if new local models detected

7. **Implement Local Model Performance Benchmarks:**
   - Test generation speed for each local model
   - Measure tokens/second throughput
   - Display in model selection UI to help user choose
   - Store benchmarks in config for future reference

8. **Testing Local Providers:**
   - Integration tests require running local LLM servers
   - Provide Docker compose file with Ollama for CI testing
   - Test model discovery
   - Test generation with various models
   - Test error handling when server not running

### Step 6.4: Implement Consent Model for LLM Usage

**Objective:** Ensure every LLM invocation requires explicit user consent with clear disclosure of data usage and costs.

**Detailed Instructions:**

1. **Define LLM Consent Levels:**
   - In `proxima/intelligence/consent.py`, create `LLMConsentLevel` enum:
     - ALWAYS_ASK - prompt for every LLM call
     - SESSION_APPROVE - approve once per session, cache for duration
     - AUTO_APPROVE_LOCAL - auto-approve local LLMs, ask for remote
     - AUTO_APPROVE_ALL - never ask (not recommended, warn user)
   - Store consent level in configuration: `llm.consent_level`

2. **Create LLM Consent Request:**
   - Extend ConsentRequest for LLM-specific information:
     - Provider name and type (local/remote)
     - Model being used
     - Estimated input tokens
     - Estimated output tokens
     - Estimated cost (for remote)
     - Purpose of LLM call (e.g., "Backend selection explanation", "Result analysis")
   - Template message:
     ```
     Proxima wants to use {provider} ({model}) to {purpose}.
     
     Provider Type: {Local/Remote}
     Estimated Tokens: {input} in, {output} out
     Estimated Cost: ${cost} USD
     
     {If remote: "Your prompt will be sent to {provider}'s servers."}
     {If local: "Processing happens entirely on your machine."}
     ```

3. **Implement LLM Consent Manager:**
   - In `proxima/intelligence/llm_consent.py`, define `LLMConsentManager`
   - Method: `request_llm_consent(provider, model, purpose, tokens_in, tokens_out) -> bool`
   - Decision tree:
     - If consent_level == AUTO_APPROVE_ALL: return True
     - If consent_level == AUTO_APPROVE_LOCAL and provider.is_local(): return True
     - Check session cache for previous approval
     - If cached and SESSION_APPROVE: return True
     - Show consent dialog to user
     - Wait for response
     - Cache decision if user selected "remember for session"
     - Return user's choice

4. **Integrate Consent into LLM Hub:**
   - Before every LLM generation call, check consent
   - If consent denied, raise PermissionDeniedError
   - Log all consent requests and decisions
   - UI shows indicator when LLM is being used (icon in status bar)

5. **Implement Token and Cost Estimation UI:**
   - Before requesting consent, show clear breakdown:
     - Input: {tokens} tokens (~{words} words)
     - Output: {tokens} tokens (estimated)
     - Cost: ${cost} (if remote)
   - For expensive calls (>$0.01), highlight in warning color
   - Allow user to adjust max_tokens before approving

6. **Implement LLM Usage Tracking:**
   - Track all LLM invocations in session
   - Store: timestamp, provider, model, purpose, tokens, cost
   - Command: `llm usage` - displays usage report
   - Example output:
     ```
     LLM Usage Summary (This Session):
     Total Calls: 12
     Remote Calls: 8 (OpenAI: 5, Anthropic: 3)
     Local Calls: 4 (Ollama: 4)
     Total Tokens: 24,580 (input) + 8,240 (output)
     Total Cost: $0.42
     ```

7. **Implement Budget Limits:**
   - Allow user to set spending limit: `llm.budget_limit_usd` in config
   - Track cumulative cost in session
   - If approaching limit (90%), warn user
   - If limit reached, require explicit override
   - Reset budget tracking: daily, weekly, monthly, or per-session

8. **Testing Consent Flow:**
   - Test all consent levels with mock LLM calls
   - Test session caching works correctly
   - Test budget limits prevent overspending
   - Test user can deny consent and operation handles gracefully
   - Test usage tracking accuracy

### Step 6.5: Implement LLM Integration Hub Orchestrator

**Objective:** Create central orchestrator that manages LLM provider selection, prompt engineering, and result processing.

**Detailed Instructions:**

1. **Create LLMHub Class:**
   - In `proxima/intelligence/llm_hub.py`, define `LLMHub` singleton
   - Dependencies: LLMProviderRegistry, LLMConsentManager, configuration
   - Responsibilities:
     - Select appropriate LLM provider and model for each task
     - Format prompts for specific tasks
     - Handle LLM responses and extract information
     - Cache responses where applicable
     - Manage fallback if primary LLM fails

2. **Implement Task-Specific Methods:**
   - Method: `explain_backend_selection(circuit, selected_backend, alternatives) -> str`
     - Purpose: Generate explanation for why backend was selected
     - Prompt template: "Given a quantum circuit with {qubits} qubits and {depth} gates, explain why {selected_backend} was chosen over {alternatives}. Consider factors: memory efficiency, speed, noise support, and user requirements."
     - Select appropriate model (fast, cheap model fine for this task)
     - Return generated explanation text
   - Method: `analyze_results(results: List[BackendResult]) -> str`
     - Purpose: Generate human-readable insights from simulation results
     - Prompt: "Analyze these quantum simulation results: {results_summary}. Provide insights on: measurement distribution, fidelity, potential errors, and recommendations."
     - Select model with good analytical capabilities
     - Return analysis text
   - Method: `suggest_parameter_adjustments(circuit, error) -> List[str]`
     - Purpose: When execution fails, suggest how to fix
     - Prompt: "A quantum simulation failed with error: {error}. Circuit has {params}. Suggest adjustments to parameters or circuit to resolve the issue."
     - Return list of suggestions

3. **Implement Prompt Templates:**
   - Store prompt templates in `proxima/intelligence/prompts/`
   - Templates as Jinja2 files for variable substitution
   - Categories:
     - Backend selection explanations
     - Result analysis
     - Error diagnosis
     - Circuit optimization suggestions
     - Noise model recommendations
   - Load and render templates with context variables

4. **Implement Response Parsing:**
   - Method: `parse_structured_response(response: str, schema: dict) -> dict`
   - Some LLM responses need structured extraction
   - Use JSON mode if provider supports (OpenAI, some others)
   - Fallback: regex or manual parsing
   - Validate parsed response against expected schema
   - Handle parsing failures gracefully

5. **Implement Provider Selection Logic:**
   - Method: `select_provider_for_task(task_type: str) -> Tuple[LLMProvider, str]`
   - Decision factors:
     - Task requirements (speed vs quality)
     - User preferences (prefer local, prefer specific provider)
     - Cost considerations
     - Availability
   - Task types:
     - "explanation" - fast, cheap model (gpt-3.5-turbo, llama2-7b local)
     - "analysis" - high quality (gpt-4, claude-sonnet, llama3-70b local)
     - "code_generation" - specialized model (gpt-4, claude)
   - Return (provider, model_id) tuple

6. **Implement Response Caching:**
   - Cache LLM responses to avoid redundant API calls
   - Key: hash of (task_type, prompt, model)
   - Store in memory cache (LRU with max size)
   - Persist cache to disk (JSON file in data directory)
   - Configurable: `llm.enable_caching` (default True)
   - Cache expiration: 24 hours or until Proxima restart
   - Command: `clear llm cache` - manual cache clear

7. **Implement Fallback Logic:**
   - If primary LLM provider fails, try fallback
   - Fallback order (configurable):
     1. User's preferred provider/model
     2. Alternative remote provider (if first was remote)
     3. Local LLM (if available)
     4. Return error if all fail
   - Log each fallback attempt
   - Notify user if fallback was used

8. **Implement LLM Task Queue:**
   - Some LLM calls can be non-blocking
   - Create async task queue for LLM invocations
   - Method: `queue_task(task_type, prompt, callback)`
   - Process queue in background
   - Callback receives result when ready
   - Used for non-critical tasks (e.g., post-execution analysis)

9. **Integration with Other Components:**
   - Backend selector uses LLMHub for explanations
   - Result interpreter uses LLMHub for analysis
   - Planning engine uses LLMHub for plan optimization
   - Error handler uses LLMHub for suggestions

10. **Testing LLMHub:**
    - Unit tests with mocked LLM providers
    - Test task-specific methods return expected format
    - Test provider selection logic chooses appropriately
    - Test caching reduces API calls
    - Test fallback works when primary fails
    - Integration test: full workflow with real LLM calls

### Phase 6 Implementation Complete

**Artifacts Produced:**
- LLMProvider abstract interface
- Remote provider implementations (OpenAI, Anthropic, Gemini, Groq, OpenRouter)
- Local provider implementations (Ollama, LM Studio, llama.cpp, text-gen-webui)
- LLM consent management system with multiple levels
- LLM usage tracking and budget limits
- LLMHub orchestrator with task-specific methods
- Response caching and fallback logic
- Integration with consent and cost estimation

**Verification:**
- All remote providers work with valid API keys
- Local providers detect and use running LLM servers
- Consent dialogs appear before LLM invocations
- Local vs remote LLMs are clearly distinguished
- Cost estimates are accurate (within 10%)
- Usage tracking records all LLM calls correctly
- Budget limits prevent overspending
- Response caching reduces redundant API calls
- Fallback works when primary provider fails
- Tests pass with coverage >80%

---

## PHASE 7: INTELLIGENT BACKEND SELECTION

### Step 7.1: Design Backend Selection Engine

**Objective:** Create an intelligent system that automatically recommends or selects the optimal quantum backend based on circuit characteristics, requirements, and system resources.

**Detailed Instructions:**

1. **Define Circuit Analysis Framework:**
   - In `proxima/intelligence/circuit_analyzer.py`, create `CircuitAnalyzer` class
   - Method: `analyze_circuit(circuit) -> CircuitProfile`
   - Extract circuit characteristics:
     - Qubit count
     - Gate count and circuit depth
     - Gate types used (identify exotic gates vs standard gates)
     - Parallelism level (gates per layer)
     - Measurement count and positions
     - Presence of noise operators
     - Parameter count (for parametric circuits)
   - CircuitProfile dataclass contains all extracted metrics
   - Use circuit traversal to build gate histogram
   - Calculate complexity metrics: circuit volume, entanglement entropy estimate

2. **Define Selection Requirements Framework:**
   - In `proxima/intelligence/selection_types.py`, create dataclasses:
     - `SelectionRequirements`:
       - `require_noise` (bool) - must support noise modeling
       - `require_density_matrix` (bool) - must be density matrix simulation
       - `require_statevector` (bool) - must be statevector simulation
       - `max_memory_mb` (Optional[int]) - memory constraint
       - `max_time_seconds` (Optional[int]) - time constraint
       - `prefer_local` (bool) - prefer local over remote backends
       - `prefer_gpu` (bool) - prefer GPU-accelerated backends
       - `accuracy_priority` (float, 0-1) - how much to prioritize accuracy over speed
       - `cost_priority` (float, 0-1) - how much to prioritize low cost (for remote backends)
   - User can specify requirements via command options or config
   - Defaults inferred from circuit if not specified

3. **Create Backend Scoring System:**
   - In `proxima/intelligence/backend_scorer.py`, create `BackendScorer` class
   - Method: `score_backend(backend, circuit_profile, requirements) -> BackendScore`
   - BackendScore dataclass:
     - `backend_id` (str)
     - `total_score` (float, 0-100) - overall suitability
     - `component_scores` (dict) - breakdown by criteria
     - `feasibility` (bool) - can this backend even run this circuit?
     - `warnings` (List[str]) - potential issues
     - `explanation` (str) - human-readable reasoning
   - Scoring components (each 0-100):
     - **Capacity score**: Can backend handle qubit count?
       - If qubits > max_qubits: score = 0, feasibility = False
       - If qubits close to limit (80-100%): score = 50
       - If qubits well within limit (<50%): score = 100
     - **Capability score**: Does backend support required features?
       - Check noise support, DM vs SV, etc.
       - Missing required capability: score = 0, feasibility = False
       - Has all capabilities: score = 100
     - **Performance score**: Estimated speed and efficiency
       - Estimate execution time based on circuit complexity and backend metadata
       - Compare against time constraint
       - GPU backends score higher if prefer_gpu=True
     - **Resource score**: Memory and computational resources
       - Estimate memory usage with `estimate_resources()`
       - Compare against available memory and constraint
       - Score inversely proportional to resource usage
     - **Accuracy score**: Expected result quality
       - DM simulations generally more accurate for noisy circuits
       - Statevector more accurate for noiseless
       - Based on backend metadata and user priority
     - **Cost score**: Execution cost (for remote backends)
       - Local backends score 100 (free)
       - Remote backends score based on estimated cost vs budget
   - Total score: weighted average of component scores
   - Weights determined by user priorities (accuracy_priority, cost_priority, etc.)

4. **Implement Feasibility Checker:**
   - Method: `check_feasibility(backend, circuit_profile, requirements) -> FeasibilityResult`
   - Hard constraints that must be met:
     - Qubit count within backend limits
     - Required capabilities present
     - Sufficient memory available
     - Backend is currently available (not crashed/offline)
   - FeasibilityResult:
     - `is_feasible` (bool)
     - `blocking_issues` (List[str]) - why it's not feasible
     - `warnings` (List[str]) - concerns but not blocking
   - Return immediately if any hard constraint fails

5. **Implement Resource Estimation Integration:**
   - Use backend's `estimate_resources()` method
   - Compare estimate with current system resources (from ResourceMonitor)
   - Apply safety margin (e.g., don't use if estimate >80% of available RAM)
   - Consider future resource drift during execution
   - Adjust feasibility based on resource projections

### Step 7.2: Implement Backend Selection Strategies

**Objective:** Create multiple selection strategies for different use cases (automatic, interactive, expert mode).

**Detailed Instructions:**

1. **Create Backend Selector Interface:**
   - In `proxima/intelligence/backend_selector.py`, define abstract `BackendSelector(ABC)`
   - Abstract method: `select_backend(circuit, requirements) -> BackendSelection`
   - BackendSelection dataclass:
     - `selected_backend_id` (str)
     - `confidence` (float, 0-1) - how confident in this selection
     - `alternatives` (List[Tuple[str, float]]) - other viable options with scores
     - `reasoning` (str) - explanation of choice
     - `require_user_confirmation` (bool)

2. **Implement Automatic Selector:**
   - In `proxima/intelligence/selectors/auto_selector.py`, define `AutomaticSelector(BackendSelector)`
   - Method: `select_backend(circuit, requirements)`:
     - Analyze circuit to get CircuitProfile
     - Score all available backends using BackendScorer
     - Filter out infeasible backends
     - Sort by total_score descending
     - Select highest scoring backend
     - If top backend score < 70, recommend user review (require_user_confirmation=True)
     - Return BackendSelection with selected backend and top 3 alternatives
   - Confidence calculation:
     - If clear winner (score gap >20 points): confidence = 0.95
     - If close competition (gap <10 points): confidence = 0.6
   - Use as default selection strategy

3. **Implement Interactive Selector:**
   - In `proxima/intelligence/selectors/interactive_selector.py`, define `InteractiveSelector(BackendSelector)`
   - Method: `select_backend(circuit, requirements)`:
     - Score all backends
     - Present ranked list to user in UI
     - Show score breakdown for top 3 candidates
     - Display resource estimates for each
     - Allow user to:
       - Select from recommendations
       - Override with any backend
       - Adjust requirements and re-score
     - Wait for user choice
     - Return BackendSelection with user's choice
   - Always require_user_confirmation=False (user already confirmed by selecting)

4. **Implement Expert Selector:**
   - In `proxima/intelligence/selectors/expert_selector.py`, define `ExpertSelector(BackendSelector)`
   - User has complete control with detailed information
   - Show comprehensive data:
     - All backend capabilities and metadata
     - Detailed resource estimates
     - Historical performance data
     - Current system resource status
   - Allow fine-grained requirement specification
   - Show real-time score updates as requirements change
   - Support "what-if" analysis: "What if I had more memory?" "What if I use GPU?"
   - Used by advanced users who understand quantum computing and backend trade-offs

5. **Implement Learning Selector:**
   - In `proxima/intelligence/selectors/learning_selector.py`, define `LearningSelector(BackendSelector)`
   - Learns from past selections and outcomes
   - Track historical data:
     - Circuit characteristics
     - Backend used
     - User satisfaction (did they manually override? did execution succeed?)
     - Actual performance (time, memory, accuracy)
   - Build simple ML model (decision tree or logistic regression):
     - Features: circuit profile metrics, requirements
     - Target: optimal backend_id
   - Train on historical data (minimum 50 executions before activating)
   - Use learned model to adjust scores or directly predict backend
   - Fall back to AutomaticSelector until enough training data
   - Store model in `data/selection_model.pkl` (pickle or joblib)

6. **Implement Selector Registry and Configuration:**
   - In `proxima/intelligence/selector_registry.py`, define `SelectorRegistry`
   - Register all selector implementations
   - Config option: `backend_selection.strategy` (values: "auto", "interactive", "expert", "learning")
   - Method: `get_selector(strategy_name) -> BackendSelector`
   - Allow user to switch strategies at runtime: `set strategy interactive`

7. **Implement Backend Comparison View:**
   - Create TUI screen: `BackendComparisonScreen` in `proxima/ui/screens/backend_comparison.py`
   - Display side-by-side comparison of 2-4 backends
   - Comparison metrics:
     - Estimated execution time
     - Estimated memory usage
     - Estimated cost (if remote)
     - Capability support
     - Accuracy expectations
   - Visual indicators: ✓ for met requirements, ✗ for failed, ⚠ for warnings
   - Allow user to select after comparison
   - Command: `compare backends cirq-dm qiskit-sv lret`

8. **Testing Selection Strategies:**
   - Test automatic selector with variety of circuits
   - Verify feasibility checker blocks impossible selections
   - Test interactive selector UI flow
   - Test learning selector improves over time (with synthetic historical data)
   - Test that selection respects user requirements
   - Integration test: select backend → execute circuit → verify backend was used

### Step 7.3: Implement LLM-Powered Explanations

**Objective:** Use LLM to generate natural language explanations for backend selection decisions.

**Detailed Instructions:**

1. **Create Explanation Generator:**
   - In `proxima/intelligence/selection_explainer.py`, define `SelectionExplainer` class
   - Method: `explain_selection(selection: BackendSelection, circuit_profile, scores: List[BackendScore]) -> str`
   - Generate detailed natural language explanation
   - Include:
     - Why selected backend was chosen
     - Why alternatives were not selected
     - Trade-offs involved
     - Risk factors (if any)
     - Recommendations for optimization

2. **Design Explanation Prompt Template:**
   - In `proxima/intelligence/prompts/backend_selection_explanation.jinja2`:
     - Template structure:
       ```
       You are explaining why backend {{ selected_backend }} was chosen for a quantum circuit simulation.
       
       Circuit Characteristics:
       - Qubits: {{ qubits }}
       - Gates: {{ gates }}
       - Depth: {{ depth }}
       - Requires Noise: {{ needs_noise }}
       
       Available Backends and Scores:
       {% for backend in scored_backends %}
       - {{ backend.name }}: Score {{ backend.score }}/100
         Pros: {{ backend.strengths }}
         Cons: {{ backend.weaknesses }}
       {% endfor %}
       
       Selection: {{ selected_backend }} (Score: {{ selected_score }})
       
       Provide a clear, concise explanation (3-5 sentences) of why this backend was optimal.
       Address: capacity match, feature requirements, performance expectations, and resource efficiency.
       ```
   - Variables filled from circuit_profile and scoring results

3. **Implement Explanation Generation Flow:**
   - After backend selected, offer to generate explanation
   - Check user preference: `backend_selection.auto_explain` (default true)
   - If enabled:
     - Request LLM consent (purpose: "Backend selection explanation")
     - Use fast, cheap LLM (GPT-3.5-turbo or local model)
     - Generate explanation
     - Display in UI (expand/collapse section)
   - Store explanation with execution record for later reference

4. **Implement Comparative Explanations:**
   - Method: `explain_comparison(backend_a, backend_b, circuit_profile) -> str`
   - Generate explanation comparing two specific backends
   - Use when user asks: "Why X instead of Y?"
   - Prompt focuses on differences and trade-offs
   - Example output:
     ```
     Cirq DM was chosen over Qiskit SV because your circuit requires noise modeling,
     which is supported by density matrix simulation but not pure statevector simulation.
     While Qiskit SV would be faster (~2x), it cannot capture the decoherence effects
     specified in your circuit. For a noiseless version, Qiskit SV would be optimal.
     ```

5. **Implement Alternative Suggestions:**
   - Method: `suggest_alternatives(selected, alternatives, circuit_profile) -> str`
   - LLM generates suggestions for when alternatives might be better
   - Example:
     ```
     Selected: LRET (Score: 85)
     Alternatives:
     - Cirq DM (Score: 82): Consider if you need more detailed noise control
     - Qiskit SV+GPU (Score: 78): Consider if you can remove noise for faster results
     ```

6. **Implement Interactive Q&A:**
   - After selection explanation shown, allow user to ask follow-up questions
   - Command: `why not qiskit?` or `explain score`
   - Maintain context of the selection decision
   - Use LLM to answer questions about the choice
   - Context includes: circuit, all backend scores, reasoning

7. **Implement Explanation Caching:**
   - Cache explanations based on circuit characteristics and selection
   - Key: hash of (circuit_profile, selected_backend, alternatives)
   - Reuse explanation if identical selection scenario occurs
   - Reduces LLM API calls for repeated patterns

8. **Testing Explanation Generation:**
   - Test explanation quality with sample selections
   - Verify explanations are accurate and helpful
   - Test comparative explanations highlight real differences
   - Test follow-up Q&A maintains context
   - User testing: are explanations clear to non-experts?

### Step 7.4: Implement Selection Constraints and Overrides

**Objective:** Give users control to constrain or override automatic selections while maintaining safety.

**Detailed Instructions:**

1. **Implement Backend Pinning:**
   - Config option: `backend_selection.pinned_backend` (default null)
   - If set, always use specified backend (unless infeasible)
   - Override automatic selection completely
   - Useful for:
     - Reproducibility (always use same backend)
     - Benchmarking specific backend
     - User preference
   - Show warning if pinned backend is suboptimal
   - Command: `pin backend cirq-sv` or `unpin backend`

2. **Implement Backend Exclusions:**
   - Config option: `backend_selection.excluded_backends` (list of backend IDs)
   - Never select excluded backends
   - Reasons to exclude:
     - Known issues with specific backend
     - License restrictions
     - User preference
   - Selection engine skips excluded backends
   - Command: `exclude backend lret` or `include backend lret`

3. **Implement Circuit-Specific Overrides:**
   - Allow attaching metadata to circuits: `circuit.metadata['backend'] = 'qiskit-sv'`
   - If metadata specifies backend, use it (with feasibility check)
   - Useful for:
     - Reproducibility
     - Research requiring specific backend
     - Testing backend differences
   - Override shown in UI with indicator

4. **Implement Requirement Profiles:**
   - Predefined requirement sets for common scenarios
   - In config: `backend_selection.profiles`:
     - `speed`: Prioritize fast execution (accuracy_priority=0.3, gpu preferred)
     - `accuracy`: Prioritize accuracy (accuracy_priority=0.9, DM preferred)
     - `cost`: Minimize cost (prefer local, avoid expensive remote)
     - `memory`: Minimize memory usage
   - Command: `use profile speed` or `use profile accuracy`
   - User can create custom profiles

5. **Implement Manual Override Flow:**
   - After automatic selection, show selection to user
   - If user disagrees, allow override:
     - Command: `use backend qiskit-dm instead`
     - UI: "Change Backend" button
   - Perform feasibility check on override
   - If infeasible, explain why and prevent override
   - If feasible but suboptimal, show warnings and require confirmation
   - Record override in history (helps learning selector)

6. **Implement Safety Guardrails:**
   - Even with overrides, prevent dangerous selections:
     - Never allow backend if estimated memory > available memory
     - Never allow if backend reported as crashed/unavailable
     - Never allow if circuit exceeds max_qubits
   - Show clear error message explaining why override rejected
   - Provide suggestion for how to proceed (e.g., "reduce qubit count" or "free up memory")

7. **Implement Selection History:**
   - Track all selection decisions in session
   - Store: timestamp, circuit_hash, automatic_choice, user_override, reasoning
   - Command: `selection history` - show list of past selections
   - Useful for:
     - Understanding patterns
     - Debugging selection issues
     - Training learning selector
   - Persist to database for long-term learning

8. **Testing Constraints and Overrides:**
   - Test pinning forces specific backend
   - Test exclusions prevent backend selection
   - Test safety guardrails block dangerous overrides
   - Test requirement profiles change selection behavior
   - Test manual override flow end-to-end
   - Test history tracking records all selections

### Step 7.5: Integration and Intelligence Enhancement

**Objective:** Integrate intelligent backend selection throughout Proxima and add advanced intelligence features.

**Detailed Instructions:**

1. **Integrate with Execution Flow:**
   - In execution controller, before running circuit:
     - If backend not specified by user, invoke selector
     - Apply backend selection strategy from config
     - Display selection and explanation (if enabled)
     - Request user confirmation if required
     - Proceed with selected backend
   - Log selection decision for audit trail

2. **Implement Backend Performance Tracking:**
   - After each execution, record actual performance
   - Metrics: actual time, actual memory, success/failure, fidelity (if known)
   - Compare actual vs estimated
   - Update backend metadata with rolling averages
   - Use for improving future estimates and selections
   - Store in `backend_performance_history` table

3. **Implement Adaptive Scoring:**
   - Adjust backend scores based on historical performance
   - If backend consistently outperforms estimates: boost score
   - If backend frequently fails or underperforms: reduce score
   - Implement exponential moving average for smooth adaptation
   - Config: `backend_selection.enable_adaptive_scoring` (default true)

4. **Implement Multi-Objective Optimization:**
   - Some scenarios require balancing multiple objectives
   - Define objective functions:
     - Minimize execution time
     - Minimize memory usage
     - Minimize cost
     - Maximize accuracy
   - Use Pareto optimization to find non-dominated backends
   - Present Pareto frontier to user: "These backends offer different trade-offs..."
   - Let user choose based on their priorities

5. **Implement Contextual Selection:**
   - Consider broader context beyond just current circuit:
     - Is this part of a batch job? (prefer consistent backend)
     - Is user doing comparison study? (use multiple backends)
     - Has user shown preference pattern? (learn preference)
     - System load: if high load, prefer lighter backend
   - Method: `select_with_context(circuit, requirements, context) -> BackendSelection`
   - Context dataclass includes session state, recent history, user patterns

6. **Implement Backend Health Scoring:**
   - Continuously monitor backend health
   - Factors:
     - Availability (is backend responding?)
     - Reliability (recent success rate)
     - Performance (recent speed vs historical average)
     - Resource stability (consistent memory usage?)
   - Health score (0-100) influences selection
   - Downweight unhealthy backends even if theoretically optimal

7. **Implement Selection Explanation Dashboard:**
   - Create UI view showing selection statistics
   - Displays:
     - Backend usage frequency (pie chart)
     - Selection confidence distribution
     - User override rate
     - Most common selection factors
     - Average scores by backend
   - Helps users understand selection patterns
   - Command: `show selection stats`

8. **Implement Export/Import Selection Policies:**
   - Allow exporting selection configuration as JSON
   - Includes: strategy, profiles, exclusions, preferences, learned model
   - Useful for:
     - Sharing configuration with team
     - Reproducible research setups
     - Backup/restore preferences
   - Command: `export selection policy policy.json`
   - Command: `import selection policy policy.json`

9. **Testing Integration:**
   - End-to-end test: run circuit with automatic selection
   - Test selection persists across session restarts
   - Test performance tracking updates backend metadata
   - Test adaptive scoring improves selection over time
   - Test contextual selection considers session state
   - Test health monitoring downgrades failing backends
   - Test multi-objective optimization presents trade-offs

### Phase 7 Implementation Complete

**Artifacts Produced:**
- CircuitAnalyzer for circuit characteristic extraction
- BackendScorer with multi-component scoring system
- Multiple selection strategies (Automatic, Interactive, Expert, Learning)
- LLM-powered explanation generator with natural language reasoning
- Constraint system (pinning, exclusions, profiles)
- Performance tracking and adaptive scoring
- Contextual selection considering broader session state
- Selection dashboard and export/import functionality

**Verification:**
- Automatic selector chooses appropriate backend for varied circuits
- Interactive selector presents clear options to user
- Expert selector provides comprehensive information
- Learning selector improves selection accuracy over time
- LLM explanations are clear and accurate
- User constraints (pin, exclude) are respected
- Safety guardrails prevent infeasible selections
- Performance tracking improves future estimates
- Adaptive scoring adjusts based on actual outcomes
- Tests pass with coverage >80%

---

## PHASE 8: PLANNING & STAGED EXECUTION

### Step 8.1: Design Planning Engine Architecture

**Objective:** Create a system that breaks complex quantum simulation tasks into logical stages with dependencies and orchestrates multi-step executions.

**Detailed Instructions:**

1. **Define Plan and Stage Abstractions:**
   - In `proxima/planning/types.py`, create core dataclasses:
     - `Stage`:
       - `stage_id` (str) - unique identifier (e.g., "stage_1", "preprocessing")
       - `name` (str) - human-readable name
       - `description` (str) - what this stage does
       - `stage_type` (StageType enum) - CIRCUIT_EXECUTION, ANALYSIS, COMPARISON, etc.
       - `dependencies` (List[str]) - stage_ids that must complete first
       - `inputs` (dict) - input parameters for this stage
       - `outputs` (dict) - expected outputs (filled after execution)
       - `estimated_duration` (float) - seconds
       - `status` (StageStatus enum) - PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
       - `error` (Optional[str]) - error message if failed
     - `ExecutionPlan`:
       - `plan_id` (str) - unique identifier
       - `name` (str) - user-defined plan name
       - `description` (str) - plan purpose
       - `stages` (List[Stage]) - ordered list of stages
       - `created_at` (datetime)
       - `started_at` (Optional[datetime])
       - `completed_at` (Optional[datetime])
       - `total_duration` (Optional[float])
       - `status` (PlanStatus enum) - DRAFT, EXECUTING, PAUSED, COMPLETED, FAILED

2. **Define Stage Types:**
   - In `proxima/planning/stage_types.py`, create `StageType` enum:
     - `CIRCUIT_EXECUTION` - run quantum circuit on backend
     - `PARAMETER_SWEEP` - execute circuit with varying parameters
     - `MULTI_BACKEND_COMPARISON` - run same circuit on multiple backends
     - `NOISE_ANALYSIS` - test circuit with different noise levels
     - `RESULT_ANALYSIS` - analyze results with LLM or custom logic
     - `CHECKPOINT` - save state for later resume
     - `DATA_EXPORT` - export results to file
     - `VISUALIZATION` - generate plots/visualizations
     - `CUSTOM` - user-defined stage

3. **Create Plan Builder Interface:**
   - In `proxima/planning/plan_builder.py`, define `PlanBuilder` class
   - Fluent API for constructing plans:
     - `create_plan(name, description) -> PlanBuilder`
     - `add_stage(stage) -> PlanBuilder`
     - `add_execution_stage(circuit, backend, shots, **options) -> PlanBuilder`
     - `add_comparison_stage(circuit, backends) -> PlanBuilder`
     - `add_analysis_stage(analysis_type, inputs) -> PlanBuilder`
     - `add_dependency(stage_id, depends_on) -> PlanBuilder`
     - `build() -> ExecutionPlan`
   - Validates plan before building:
     - No circular dependencies
     - All dependencies reference valid stages
     - All required inputs specified

4. **Implement Dependency Graph:**
   - In `proxima/planning/dependency_graph.py`, create `DependencyGraph` class
   - Represent stages and dependencies as directed acyclic graph (DAG)
   - Method: `build_from_plan(plan) -> DependencyGraph`
   - Method: `topological_sort() -> List[Stage]` - returns execution order
   - Method: `detect_cycles() -> List[List[str]]` - find circular dependencies
   - Method: `get_ready_stages(completed_stages) -> List[Stage]` - stages ready to run
   - Use networkx library for graph operations if available, else implement manually

5. **Implement Plan Validation:**
   - In `proxima/planning/validator.py`, define `PlanValidator` class
   - Method: `validate_plan(plan) -> ValidationResult`
   - Validation checks:
     - No circular dependencies (use dependency graph)
     - All stage dependencies exist
     - Input/output compatibility (stage output matches downstream input type)
     - Resource feasibility (sum of estimated resources doesn't exceed limits)
     - Backend availability for execution stages
     - Valid stage configurations
   - ValidationResult contains: `is_valid` (bool), `errors` (List[str]), `warnings` (List[str])
   - Display validation errors to user before execution

### Step 8.2: Implement Stage Executors

**Objective:** Create execution logic for each stage type, handling stage-specific requirements and producing outputs.

**Detailed Instructions:**

1. **Create Stage Executor Interface:**
   - In `proxima/planning/executors/base.py`, define abstract `StageExecutor(ABC)`
   - Abstract methods:
     - `can_execute(stage: Stage) -> bool` - does this executor handle this stage type?
     - `execute(stage: Stage, context: ExecutionContext) -> StageResult`
     - `estimate_duration(stage: Stage) -> float` - seconds
     - `estimate_resources(stage: Stage) -> ResourceEstimate`
   - StageResult dataclass:
     - `success` (bool)
     - `outputs` (dict) - results produced by stage
     - `duration` (float) - actual execution time
     - `error` (Optional[str])
     - `metadata` (dict) - executor-specific info

2. **Implement Circuit Execution Stage Executor:**
   - In `proxima/planning/executors/circuit_executor.py`, define `CircuitExecutionExecutor(StageExecutor)`
   - Handles StageType.CIRCUIT_EXECUTION
   - Method: `execute(stage, context)`:
     - Extract inputs: circuit, backend_id, shots, options
     - Get backend from registry
     - Check resources before execution
     - Execute circuit via execution controller
     - Collect results
     - Store results in stage.outputs: `{"result": BackendResult, "success": bool}`
     - Return StageResult
   - If execution fails, mark stage as FAILED with error details
   - Support retry logic (configurable retries per stage)

3. **Implement Parameter Sweep Stage Executor:**
   - In `proxima/planning/executors/parameter_sweep_executor.py`, define `ParameterSweepExecutor(StageExecutor)`
   - Handles StageType.PARAMETER_SWEEP
   - Stage inputs:
     - `circuit` - parametric circuit
     - `parameter_name` - which parameter to vary
     - `parameter_values` - list/range of values
     - `backend_id` - which backend to use
     - `shots` - shots per execution
   - Execution logic:
     - Iterate through parameter values
     - For each value: substitute parameter in circuit, execute
     - Collect all results
     - Store in outputs: `{"results": List[BackendResult], "parameter_values": List[float]}`
   - Display progress: "Parameter sweep: 5/20 values completed"
   - Support parallel execution if backend supports it

4. **Implement Multi-Backend Comparison Stage Executor:**
   - In `proxima/planning/executors/comparison_executor.py`, define `ComparisonExecutor(StageExecutor)`
   - Handles StageType.MULTI_BACKEND_COMPARISON
   - Stage inputs:
     - `circuit`
     - `backend_ids` - list of backends to compare
     - `shots`
   - Execution logic:
     - Execute circuit on each backend sequentially or in parallel
     - Collect results from each
     - Compute comparison metrics: fidelity differences, execution time ratios, memory usage
     - Generate comparison summary
     - Store outputs: `{"results_by_backend": dict, "comparison_metrics": dict}`
   - Used for benchmarking and validation

5. **Implement Result Analysis Stage Executor:**
   - In `proxima/planning/executors/analysis_executor.py`, define `AnalysisExecutor(StageExecutor)`
   - Handles StageType.RESULT_ANALYSIS
   - Stage inputs:
     - `results` - BackendResult or list of results to analyze
     - `analysis_type` - "statistical", "llm_summary", "fidelity_check", etc.
   - Analysis types:
     - **Statistical**: compute mean, variance, distribution shape
     - **LLM Summary**: use LLM to generate natural language insights
     - **Fidelity Check**: compare against expected/theoretical result
     - **Error Analysis**: identify anomalies or outliers
   - Execution logic:
     - Dispatch to appropriate analysis function
     - Generate analysis report (text and/or structured data)
     - Store in outputs: `{"analysis_report": str, "metrics": dict}`
   - Request LLM consent if using LLM analysis

6. **Implement Checkpoint Stage Executor:**
   - In `proxima/planning/executors/checkpoint_executor.py`, define `CheckpointExecutor(StageExecutor)`
   - Handles StageType.CHECKPOINT
   - Saves current plan state to disk
   - Stage inputs: `checkpoint_name` (optional)
   - Execution logic:
     - Serialize current plan state (all stage outputs so far)
     - Save to checkpoint file in `data/checkpoints/{plan_id}/{checkpoint_name}.json`
     - Store checkpoint path in outputs
   - Used for resuming long-running plans

7. **Implement Data Export Stage Executor:**
   - In `proxima/planning/executors/export_executor.py`, define `ExportExecutor(StageExecutor)`
   - Handles StageType.DATA_EXPORT
   - Stage inputs:
     - `data` - what to export (results, analysis, etc.)
     - `format` - "json", "csv", "hdf5"
     - `filepath` - where to save
   - Execution logic:
     - Convert data to specified format
     - Write to file
     - Verify write success
     - Store filepath in outputs
   - Support multiple formats for flexibility

8. **Implement Visualization Stage Executor:**
   - In `proxima/planning/executors/visualization_executor.py`, define `VisualizationExecutor(StageExecutor)`
   - Handles StageType.VISUALIZATION
   - Stage inputs:
     - `data` - results to visualize
     - `plot_type` - "histogram", "line", "heatmap", "bloch_sphere"
     - `filepath` - where to save image
   - Execution logic:
     - Generate visualization using matplotlib or plotly
     - Save to file (PNG, SVG, or interactive HTML)
     - Store filepath in outputs
   - Used for generating reports

9. **Implement Executor Registry:**
   - In `proxima/planning/executor_registry.py`, define `ExecutorRegistry`
   - Register all executor implementations
   - Method: `get_executor(stage_type) -> StageExecutor`
   - Auto-discover executors at initialization
   - Allow custom executors to be registered by extensions

10. **Testing Stage Executors:**
    - Unit test each executor with mock stages
    - Test error handling (circuit fails, backend unavailable)
    - Test output format correctness
    - Integration test: execute each stage type in isolation
    - Test resource estimation accuracy

### Step 8.3: Implement Plan Orchestrator

**Objective:** Create the central orchestrator that manages plan execution, handles dependencies, and coordinates stage execution.

**Detailed Instructions:**

1. **Create Plan Orchestrator:**
   - In `proxima/planning/orchestrator.py`, define `PlanOrchestrator` class
   - Manages execution of entire plan
   - Dependencies: ExecutorRegistry, DependencyGraph, ResourceMonitor
   - State: current_plan, executing_stages, completed_stages, failed_stages

2. **Implement Plan Execution Flow:**
   - Method: `execute_plan(plan: ExecutionPlan) -> PlanResult`
   - Execution algorithm:
     1. Validate plan before execution
     2. Build dependency graph
     3. Perform topological sort to get execution order
     4. Initialize execution context (shared state across stages)
     5. While there are incomplete stages:
        a. Get ready stages (dependencies satisfied)
        b. For each ready stage:
           - Get appropriate executor
           - Check resources before execution
           - Execute stage
           - Update stage status
           - Store outputs in context
           - If failed: handle error (abort plan or continue based on policy)
        c. If no stages ready and some pending: dependency deadlock (shouldn't happen if validated)
     6. Mark plan as completed or failed
     7. Return PlanResult with summary
   - Support parallel stage execution where dependencies allow

3. **Implement Execution Context:**
   - In `proxima/planning/execution_context.py`, define `ExecutionContext` class
   - Shared state accessible to all stages
   - Stores:
     - Stage outputs (by stage_id)
     - Global variables (user-defined)
     - Execution metadata (start time, current stage, etc.)
   - Method: `get_stage_output(stage_id, key) -> Any` - access previous stage output
   - Method: `set_global(key, value)` - store global state
   - Thread-safe for parallel execution

4. **Implement Stage Dependencies Resolution:**
   - Method: `resolve_dependencies(stage, context) -> dict`
   - For each dependency in stage.dependencies:
     - Retrieve output from completed stage
     - Map to stage's expected input
   - Handle dependency mapping:
     - Stage A output "result" → Stage B input "previous_result"
   - Raise error if required dependency output missing

5. **Implement Parallel Stage Execution:**
   - Identify stages that can run in parallel (no dependencies on each other)
   - Use ThreadPoolExecutor or asyncio for concurrent execution
   - Config: `planning.max_parallel_stages` (default 2)
   - Monitor resources to ensure parallel execution doesn't exceed limits
   - Display parallel execution in UI: "Running stages 2, 3, 5 in parallel"

6. **Implement Error Handling Policies:**
   - Define `ErrorPolicy` enum:
     - ABORT_ON_ERROR - stop plan execution if any stage fails
     - CONTINUE_ON_ERROR - skip failed stage, continue with others
     - RETRY_ON_ERROR - retry failed stage N times before giving up
   - Config: `planning.error_policy` (default ABORT_ON_ERROR)
   - Store failed stage info for debugging
   - Allow conditional logic: continue only if non-critical stage fails

7. **Implement Plan Pause and Resume:**
   - Method: `pause_plan()` - gracefully pause execution
     - Finish currently running stages
     - Don't start new stages
     - Save current state to checkpoint
     - Update plan status to PAUSED
   - Method: `resume_plan(plan_id)` - continue from checkpoint
     - Load plan state from checkpoint
     - Restore completed stage outputs to context
     - Resume execution from next pending stage
   - Command: `pause plan` and `resume plan <plan_id>`

8. **Implement Real-Time Progress Tracking:**
   - Track execution progress:
     - Stages completed / total stages
     - Current stage and estimated time remaining
     - Overall plan progress percentage
   - Display in UI: progress bar, stage timeline
   - Update every second during execution
   - Emit events: stage_started, stage_completed, plan_completed

9. **Testing Orchestrator:**
   - Test simple linear plan (no parallel stages)
   - Test plan with parallel stages
   - Test error handling (stage failure)
   - Test pause and resume functionality
   - Test dependency resolution with complex dependencies
   - Test resource constraints prevent overload
   - Integration test: full plan execution end-to-end

### Step 8.4: Implement Plan Templates and LLM-Assisted Planning

**Objective:** Provide pre-built plan templates and use LLM to help users create custom plans.

**Detailed Instructions:**

1. **Create Plan Template System:**
   - In `proxima/planning/templates/`, create JSON files for common plan templates:
     - `single_execution.json` - simple circuit execution
     - `backend_comparison.json` - compare multiple backends
     - `parameter_sweep.json` - sweep circuit parameter
     - `noise_study.json` - test various noise levels
     - `full_analysis.json` - execution + analysis + visualization
   - Template structure: partial ExecutionPlan with placeholders
   - Placeholders: `{{circuit}}`, `{{backend}}`, `{{parameter_name}}`, etc.

2. **Implement Template Loader:**
   - In `proxima/planning/template_loader.py`, define `TemplateLoader` class
   - Method: `load_template(template_name) -> ExecutionPlan`
   - Method: `list_templates() -> List[TemplateInfo]`
   - Method: `instantiate_template(template, parameters) -> ExecutionPlan`
   - Replace placeholders with actual values
   - Validate instantiated plan before returning

3. **Implement Template Instantiation UI:**
   - Create TUI screen: `TemplateInstantiationScreen`
   - Shows list of available templates with descriptions
   - User selects template
   - Wizard guides user through filling placeholders:
     - "Select circuit" → file picker or circuit builder
     - "Select backend" → backend selector
     - "Parameter values" → input range or list
   - Preview generated plan before execution
   - Command: `plan from template backend_comparison`

4. **Implement LLM-Assisted Plan Generation:**
   - Method: `generate_plan_from_description(description: str) -> ExecutionPlan`
   - User provides natural language description:
     - "Compare LRET and Cirq on a 10-qubit GHZ circuit with 1000 shots"
     - "Sweep rotation angle from 0 to π in 20 steps on Qiskit"
   - Send to LLM with prompt:
     ```
     Generate a quantum simulation execution plan based on this description: {{description}}
     
     Available stage types: CIRCUIT_EXECUTION, PARAMETER_SWEEP, MULTI_BACKEND_COMPARISON, 
     RESULT_ANALYSIS, CHECKPOINT, DATA_EXPORT, VISUALIZATION
     
     Available backends: {{backend_list}}
     
     Return a JSON plan with stages, dependencies, and inputs.
     ```
   - Parse LLM response as JSON
   - Validate generated plan
   - Present to user for approval before execution

5. **Implement Plan Optimization Suggestions:**
   - Method: `suggest_optimizations(plan) -> List[Suggestion]`
   - Analyze plan for improvement opportunities:
     - Parallelizable stages currently sequential
     - Redundant executions
     - Resource-inefficient backend choices
     - Missing checkpoints in long plans
     - Missing analysis or export stages
   - Use LLM to generate suggestions:
     - "Stage 3 and 4 can run in parallel since they have no dependencies"
     - "Consider adding a checkpoint after stage 5 (long-running)"
   - Display suggestions to user, allow one-click application

6. **Implement Interactive Plan Builder:**
   - Create TUI screen: `InteractivePlanBuilderScreen`
   - Visual plan editor:
     - Add stages with drag-and-drop or keyboard
     - Draw dependencies as arrows between stages
     - Edit stage parameters inline
     - Validate plan in real-time
     - Preview execution timeline
   - Save custom plans for reuse
   - Export/import plans as JSON

7. **Implement Plan Library:**
   - Store user-created plans in database
   - Table: plans (plan_id, name, description, plan_json, created_at, last_used)
   - Command: `save plan <name>` - save current plan
   - Command: `load plan <name>` - load saved plan
   - Command: `list plans` - show all saved plans
   - Allow tagging plans for organization

8. **Testing Plan Creation:**
   - Test template loading and instantiation
   - Test LLM plan generation with various descriptions
   - Verify generated plans are valid and executable
   - Test optimization suggestions improve plans
   - Test interactive builder creates correct plans
   - Test plan library save/load functionality

### Step 8.5: Implement Plan Visualization and Reporting

**Objective:** Create comprehensive visualization of plan structure and execution, plus automated reporting.

**Detailed Instructions:**

1. **Implement Plan Visualization:**
   - In `proxima/planning/visualizer.py`, define `PlanVisualizer` class
   - Method: `visualize_plan_graph(plan) -> str` (ASCII art or return path to image)
   - Generate visual representation of plan:
     - Nodes represent stages
     - Edges represent dependencies
     - Color-code by status (pending, running, completed, failed)
     - Size by estimated duration
   - Use graphviz library for rendering to image
   - ASCII fallback for terminal-only display
   - Command: `visualize plan` - display in UI or open image

2. **Implement Execution Timeline View:**
   - Create Gantt chart showing stage execution over time
   - X-axis: time
   - Y-axis: stages
   - Bars show: estimated duration (translucent), actual duration (solid)
   - Indicate parallel execution (overlapping bars)
   - Show current execution point
   - Update in real-time during execution
   - Use Textual widgets to render in TUI

3. **Implement Stage Detail View:**
   - TUI screen showing detailed info for specific stage
   - Display:
     - Stage name and description
     - Type and executor
     - Inputs and outputs
     - Dependencies and dependents
     - Estimated vs actual duration
     - Resource usage
     - Logs and error messages (if any)
   - Navigate from plan view: select stage, press Enter for details

4. **Implement Plan Execution Report:**
   - Method: `generate_execution_report(plan) -> Report`
   - Report dataclass contains:
     - Summary: total duration, stages completed/failed, overall status
     - Stage-by-stage breakdown with timings
     - Resource usage statistics
     - Key results and outputs
     - Visualizations (embedded charts)
     - Recommendations for future executions
   - Format: structured dict convertible to JSON, Markdown, or HTML

5. **Implement Automated Report Generation:**
   - After plan completion, optionally auto-generate report
   - Config: `planning.auto_generate_report` (default true)
   - Save report to `reports/plan_{plan_id}_{timestamp}.md`
   - Include:
     - Plan overview and objectives
     - Execution timeline with timestamps
     - Stage outputs (results, analysis, visualizations)
     - Performance metrics
     - Errors encountered and resolutions
   - Use LLM to generate natural language summary (with consent)

6. **Implement Report Templates:**
   - Different report styles for different audiences:
     - **Technical**: full details, all data, verbose
     - **Executive**: high-level summary, key findings, minimal detail
     - **Presentation**: focus on visualizations, brief text
   - User selects template when generating report
   - Templates stored as Jinja2 files in `proxima/planning/report_templates/`

7. **Implement Report Export:**
   - Export report to multiple formats:
     - Markdown (`.md`)
     - HTML (with embedded CSS and charts)
     - PDF (using weasyprint or similar)
     - JSON (structured data)
   - Command: `export report <plan_id> --format html --output report.html`
   - Embed images and data inline for self-contained reports

8. **Implement Plan Comparison View:**
   - Compare multiple plan executions:
     - Same plan run at different times
     - Different plans for same task
   - Show side-by-side:
     - Duration comparison
     - Resource usage comparison
     - Result accuracy comparison
     - Cost comparison (if applicable)
   - Identify performance regressions or improvements
   - Command: `compare plans <plan_id_1> <plan_id_2>`

9. **Testing Visualization and Reporting:**
   - Test plan graph generation for various plan structures
   - Test timeline view updates during execution
   - Test report generation includes all relevant information
   - Test report export to all supported formats
   - Test plan comparison correctly identifies differences
   - Visual QA: ensure visualizations are clear and informative

### Phase 8 Implementation Complete

**Artifacts Produced:**
- Plan and Stage abstractions with full type system
- Dependency graph with topological sorting
- Stage executors for 8+ stage types
- Plan orchestrator with parallel execution support
- Pause/resume functionality with checkpointing
- Plan templates and LLM-assisted plan generation
- Interactive plan builder UI
- Plan visualization (graph, timeline, details)
- Automated report generation with multiple formats
- Plan comparison functionality

**Verification:**
- Simple plans execute correctly in order
- Complex plans with dependencies execute in correct order
- Parallel stages execute concurrently when possible
- Pause and resume works without data loss
- Error handling follows configured policy
- Plan templates instantiate correctly
- LLM generates valid plans from descriptions
- Visualizations accurately represent plan structure
- Reports contain comprehensive execution information
- Plan comparison identifies differences
- Tests pass with coverage >80%

---

## PHASE 9: RESULT INTERPRETATION & INSIGHTS

### Step 9.1: Design Result Parser and Data Extraction

**Objective:** Create robust parsers for extracting structured data from quantum simulation results in various formats (CSV, XLSX, JSON).

**Detailed Instructions:**

1. **Define Result Data Models:**
   - In `proxima/results/types.py`, create dataclasses for structured result representation:
     - `MeasurementData`:
       - `bitstrings` (List[str]) - e.g., ["00", "01", "10", "11"]
       - `counts` (List[int]) - occurrences of each bitstring
       - `probabilities` (List[float]) - normalized probabilities
       - `total_shots` (int)
     - `StatevectorData`:
       - `amplitudes` (ndarray) - complex amplitudes
       - `basis_states` (List[str]) - corresponding basis states
       - `probabilities` (ndarray) - |amplitude|^2
     - `DensityMatrixData`:
       - `matrix` (ndarray) - full density matrix
       - `dimension` (int)
       - `trace` (float) - should be 1.0
       - `purity` (float) - Tr(ρ²)
     - `ExecutionMetadata`:
       - `backend_id` (str)
       - `execution_time` (float)
       - `memory_used` (float)
       - `timestamp` (datetime)
       - `circuit_info` (dict)

2. **Implement CSV Parser:**
   - In `proxima/results/parsers/csv_parser.py`, define `CSVResultParser` class
   - Method: `parse(filepath) -> ParsedResult`
   - CSV format detection:
     - Identify columns: bitstring/state, count, probability
     - Detect header row vs no header
     - Handle different delimiters (comma, tab, semicolon)
   - Use pandas library for robust parsing:
     - `df = pd.read_csv(filepath, delimiter=auto_detect)`
     - Handle various column names: "state", "bitstring", "measurement", etc.
     - Handle count columns: "count", "counts", "frequency", "shots"
   - Validation:
     - Verify bitstrings are valid binary strings
     - Check counts are non-negative integers
     - Verify probabilities sum to ~1.0 (within tolerance)
   - Convert to MeasurementData format
   - Handle errors: malformed CSV, missing columns, invalid data

3. **Implement XLSX Parser:**
   - In `proxima/results/parsers/xlsx_parser.py`, define `XLSXResultParser` class
   - Use openpyxl or pandas for Excel file parsing
   - Method: `parse(filepath, sheet_name=None) -> ParsedResult`
   - Multi-sheet support:
     - If sheet_name not specified, try to auto-detect data sheet
     - Look for sheets named "Results", "Measurements", "Data", etc.
     - Allow user to specify sheet: `parse(file, sheet_name="Sheet2")`
   - Similar column detection as CSV
   - Additional features:
     - Handle Excel formulas (evaluate to values)
     - Skip empty rows and columns
     - Handle merged cells
   - Extract metadata from separate metadata sheet if present

4. **Implement JSON Parser:**
   - In `proxima/results/parsers/json_parser.py`, define `JSONResultParser` class
   - Handle multiple JSON schemas:
     - Qiskit result format
     - Cirq result format
     - Custom Proxima format
     - Generic key-value format
   - Method: `parse(filepath) -> ParsedResult`
   - Schema detection:
     - Look for known keys: "counts", "statevector", "density_matrix"
     - Try multiple schemas until one succeeds
   - Recursive parsing for nested structures
   - Extract both data and metadata from JSON
   - Handle large JSON files (streaming parser for files >100MB)

5. **Implement Structured CSV Parser for LRET Output:**
   - LRET may produce structured CSV with multiple sections
   - In `proxima/results/parsers/structured_csv_parser.py`, define `StructuredCSVParser`
   - Handle sections like:
     - `[METADATA]` - execution info
     - `[MEASUREMENTS]` - measurement results
     - `[STATEVECTOR]` - statevector components
     - `[FIDELITY]` - fidelity metrics
   - Parse each section independently
   - Combine into comprehensive ParsedResult
   - This is backend-specific but provides richer data

6. **Implement Parser Registry:**
   - In `proxima/results/parser_registry.py`, define `ParserRegistry` singleton
   - Auto-detect file format from extension and content
   - Method: `get_parser(filepath) -> ResultParser`
   - Try parsers in order until one succeeds
   - Method: `parse_result_file(filepath) -> ParsedResult`
   - Unified interface regardless of format

7. **Implement Data Validation and Normalization:**
   - In `proxima/results/validator.py`, define `ResultValidator` class
   - Method: `validate(parsed_result) -> ValidationResult`
   - Checks:
     - Physical validity (probabilities in [0,1], sum to 1)
     - Data consistency (counts match probabilities)
     - Completeness (required fields present)
     - Format correctness (bitstrings match qubit count)
   - Normalization:
     - Re-normalize probabilities if slightly off (rounding errors)
     - Convert between representations (counts ↔ probabilities)
     - Standardize bitstring format (big-endian)

8. **Testing Parsers:**
   - Create test files in all supported formats
   - Test with valid and invalid data
   - Test with edge cases: empty results, single measurement, very large files
   - Test with real output from LRET, Cirq, Qiskit
   - Test parser auto-detection accuracy
   - Test validation catches errors

### Step 9.2: Implement Statistical Analysis

**Objective:** Compute statistical metrics and properties from quantum measurement results.

**Detailed Instructions:**

1. **Implement Distribution Statistics:**
   - In `proxima/results/analysis/statistics.py`, define `StatisticalAnalyzer` class
   - Method: `compute_basic_stats(measurement_data) -> BasicStatistics`
   - BasicStatistics dataclass:
     - `mean_value` (float) - if states can be interpreted as numbers
     - `variance` (float)
     - `standard_deviation` (float)
     - `entropy` (float) - Shannon entropy of distribution
     - `most_probable_state` (str)
     - `most_probable_probability` (float)
     - `unique_states_observed` (int)
   - Shannon entropy: H = -Σ p_i log₂(p_i)
     - High entropy: uniform distribution (maximally uncertain)
     - Low entropy: peaked distribution (more certain)

2. **Implement Purity and Entanglement Metrics:**
   - For density matrix results:
   - Method: `compute_purity(density_matrix) -> float`
     - Purity: Tr(ρ²)
     - Pure state: purity = 1
     - Mixed state: purity < 1
   - Method: `compute_von_neumann_entropy(density_matrix) -> float`
     - S = -Tr(ρ log ρ)
     - Measure of mixedness
   - Method: `estimate_entanglement(density_matrix, qubit_partition) -> float`
     - Compute reduced density matrix for subsystem
     - Measure entanglement via entropy of entanglement
     - Requires specifying partition (which qubits vs which qubits)

3. **Implement Fidelity Calculations:**
   - In `proxima/results/analysis/fidelity.py`, define `FidelityAnalyzer` class
   - Method: `compute_state_fidelity(actual_state, expected_state) -> float`
   - State fidelity between two states:
     - For pure states: F = |⟨ψ|φ⟩|²
     - For mixed states: F = Tr(√(√ρ σ √ρ))
   - Use numpy and scipy for matrix operations
   - Method: `compute_measurement_fidelity(actual_counts, expected_counts) -> float`
     - Classical fidelity between two probability distributions
     - F = Σ √(p_i q_i)
   - Method: `compute_process_fidelity(actual_process, ideal_process) -> float`
     - For process tomography results

4. **Implement Error Analysis:**
   - In `proxima/results/analysis/errors.py`, define `ErrorAnalyzer` class
   - Method: `estimate_sampling_error(counts, shots) -> dict`
   - Statistical uncertainty due to finite sampling:
     - For each state: σ = √(p(1-p)/N)
     - Return confidence intervals (95%)
   - Method: `detect_outliers(measurement_data) -> List[str]`
     - Identify unexpectedly frequent/rare outcomes
     - Use statistical tests (Z-score, IQR method)
   - Method: `estimate_readout_error(counts, expected_counts) -> ReadoutErrorModel`
     - Compare observed vs expected to infer readout errors
     - Build confusion matrix

5. **Implement Distribution Comparison:**
   - Method: `compare_distributions(dist1, dist2) -> ComparisonMetrics`
   - Metrics:
     - **Total Variation Distance**: TVD = 0.5 * Σ |p_i - q_i|
     - **Kullback-Leibler Divergence**: KL(P||Q) = Σ p_i log(p_i/q_i)
     - **Jensen-Shannon Divergence**: symmetric version of KL
     - **Hellinger Distance**: H = √(1 - Σ √(p_i q_i))
   - Used for comparing results across backends or runs
   - ComparisonMetrics dataclass with all computed metrics

6. **Implement Correlation Analysis:**
   - Method: `analyze_bit_correlations(measurement_data) -> CorrelationMatrix`
   - Compute correlations between qubit measurements
   - Build correlation matrix: C[i][j] = correlation between qubit i and j
   - Identify strongly correlated qubits (may indicate entanglement)
   - Visualize as heatmap

7. **Implement Hypothesis Testing:**
   - Method: `test_uniformity(measurement_data) -> TestResult`
   - Chi-squared test: is distribution uniform?
   - Method: `test_distribution_match(observed, expected) -> TestResult`
   - Chi-squared goodness-of-fit test
   - Return p-value and conclusion (reject/fail to reject)
   - TestResult dataclass: `statistic`, `p_value`, `conclusion`, `confidence_level`

8. **Testing Statistical Analysis:**
   - Test with known distributions (uniform, peaked, Bell state)
   - Verify fidelity calculations with analytical examples
   - Test entropy calculations
   - Test error estimation produces reasonable bounds
   - Test hypothesis tests correctly identify matches/mismatches

### Step 9.3: Implement LLM-Powered Result Interpretation

**Objective:** Use LLM to generate natural language insights, explanations, and recommendations from quantum simulation results.

**Detailed Instructions:**

1. **Create Result Summarizer:**
   - In `proxima/results/interpretation/summarizer.py`, define `ResultSummarizer` class
   - Method: `summarize_results(parsed_result, statistics) -> str`
   - Generate human-readable summary of results
   - Include:
     - Most probable outcomes
     - Distribution characteristics (uniform, peaked, bimodal)
     - Key statistics (entropy, fidelity)
     - Notable observations
   - Length: 3-5 sentences, concise but informative

2. **Design Result Interpretation Prompt Template:**
   - In `proxima/intelligence/prompts/result_interpretation.jinja2`:
   - Template structure:
     ```
     Analyze these quantum simulation results and provide insights:
     
     Circuit: {{ circuit_description }}
     Backend: {{ backend_name }}
     Shots: {{ shots }}
     
     Measurement Results:
     {% for state, prob in top_states %}
     - |{{ state }}⟩: {{ prob }}
     {% endfor %}
     
     Statistics:
     - Entropy: {{ entropy }}
     - Most probable: |{{ most_probable }}⟩ ({{ max_prob }})
     {% if fidelity %}
     - Fidelity to ideal: {{ fidelity }}
     {% endif %}
     
     Provide:
     1. Interpretation of the measurement distribution
     2. Physical insights (e.g., superposition, entanglement, interference)
     3. Assessment of result quality
     4. Potential error sources if fidelity is low
     5. Recommendations for further analysis or circuit modifications
     
     Be concise but technically accurate. Target audience: quantum computing practitioners.
     ```

3. **Implement LLM Interpretation Generator:**
   - In `proxima/results/interpretation/llm_interpreter.py`, define `LLMInterpreter` class
   - Method: `interpret_results(parsed_result, statistics, context) -> Interpretation`
   - Interpretation dataclass:
     - `summary` (str) - brief overview
     - `detailed_analysis` (str) - in-depth interpretation
     - `insights` (List[str]) - key findings
     - `recommendations` (List[str]) - suggested actions
     - `confidence` (float) - LLM's confidence in interpretation
   - Flow:
     1. Prepare result data for LLM
     2. Render prompt template with data
     3. Request LLM consent (purpose: "Result interpretation")
     4. Call LLM Hub with interpretation task
     5. Parse LLM response
     6. Structure into Interpretation object
   - Use medium-quality model (GPT-4, Claude Sonnet, or good local model)

4. **Implement Anomaly Detection and Explanation:**
   - Method: `detect_and_explain_anomalies(results, expected_behavior) -> List[Anomaly]`
   - Anomaly dataclass:
     - `description` (str) - what's anomalous
     - `severity` (str) - "info", "warning", "critical"
     - `possible_causes` (List[str])
     - `recommendations` (List[str])
   - Types of anomalies:
     - Unexpected peak in distribution
     - Missing expected states
     - Fidelity lower than expected
     - Suspicious correlations
   - Use LLM to generate explanations for detected anomalies
   - Prompt: "This quantum circuit was expected to produce {{expected}}, but we observed {{actual}}. What could explain this discrepancy?"

5. **Implement Comparative Interpretation:**
   - Method: `interpret_comparison(results_list, backends) -> ComparativeInterpretation`
   - When multiple backend results available:
     - Compare distributions
     - Identify agreements and discrepancies
     - Explain why backends might differ
   - LLM prompt:
     ```
     Compare these results from different quantum simulators:
     
     LRET: {{ lret_distribution }}
     Cirq: {{ cirq_distribution }}
     Qiskit: {{ qiskit_distribution }}
     
     Analysis:
     - Which results are most similar?
     - What causes the differences?
     - Which result is likely most accurate?
     - What do the differences tell us about backend characteristics?
     ```

6. **Implement Insight Extraction:**
   - Method: `extract_insights(results, circuit_type) -> List[Insight]`
   - Insight dataclass:
     - `category` (str) - "entanglement", "interference", "noise", "performance"
     - `description` (str)
     - `supporting_evidence` (dict) - metrics that support this insight
     - `implications` (str) - what this means for the user
   - Use LLM to identify non-obvious patterns and insights
   - Examples:
     - "Results show Bell state entanglement between qubits 0 and 1"
     - "High entropy suggests decoherence effects are significant"
     - "Interference pattern visible in outcome distribution"

7. **Implement Recommendation Engine:**
   - Method: `generate_recommendations(results, circuit, execution_context) -> List[Recommendation]`
   - Recommendation dataclass:
     - `title` (str) - brief recommendation
     - `description` (str) - detailed explanation
     - `priority` (str) - "high", "medium", "low"
     - `actionable` (bool) - can user act on this?
     - `action_command` (Optional[str]) - Proxima command to implement
   - Types of recommendations:
     - Circuit optimization: "Reduce circuit depth to improve fidelity"
     - Backend selection: "Try statevector backend for this noiseless circuit"
     - Parameter tuning: "Increase shots to reduce sampling error"
     - Further analysis: "Run parameter sweep to explore sensitivity"
   - Use LLM for context-aware recommendations

8. **Implement Explanation Caching:**
   - Cache LLM interpretations to avoid redundant API calls
   - Key: hash of (result_data, statistics, context)
   - Cache TTL: 7 days (results don't change over time)
   - Store in local cache directory

9. **Testing LLM Interpretation:**
   - Test with known quantum states (Bell, GHZ, random)
   - Verify interpretations are physically accurate
   - Test anomaly detection catches real anomalies
   - Test comparative interpretation identifies differences
   - Test recommendations are sensible and actionable
   - User testing: are explanations helpful to non-experts?

### Step 9.4: Implement Visualization Generation

**Objective:** Create informative visualizations of quantum simulation results with various plot types.

**Detailed Instructions:**

1. **Setup Visualization Infrastructure:**
   - In `proxima/results/visualization/`, create visualization module
   - Use matplotlib for static plots
   - Use plotly for interactive plots (optional)
   - Define `VisualizationConfig` dataclass:
     - `style` (str) - "publication", "presentation", "notebook"
     - `interactive` (bool) - use plotly vs matplotlib
     - `dpi` (int) - resolution for static plots
     - `color_scheme` (str) - "default", "colorblind", "grayscale"
     - `show_grid` (bool)
     - `font_size` (int)

2. **Implement Measurement Histogram:**
   - In `proxima/results/visualization/histogram.py`, define `MeasurementHistogram` class
   - Method: `plot(measurement_data, options) -> Figure`
   - X-axis: basis states (bitstrings)
   - Y-axis: probability or count
   - Features:
     - Sort by probability (highest first) or by bitstring value
     - Limit to top N states (e.g., top 10)
     - Color-code by expected vs unexpected
     - Error bars for sampling uncertainty
     - Horizontal line showing uniform distribution for reference
   - Save to file or return matplotlib Figure

3. **Implement Statevector Visualization:**
   - In `proxima/results/visualization/statevector.py`, define `StatevectorVisualizer` class
   - Method: `plot_amplitudes(statevector) -> Figure`
   - Bar chart showing real and imaginary parts of amplitudes
   - Separate subplot for magnitudes
   - Separate subplot for phases
   - Method: `plot_bloch_sphere(statevector, qubits)` - for single/two qubit states
   - Use qutip library if available for Bloch sphere rendering
   - Fallback: 2D projection of Bloch vector

4. **Implement Density Matrix Visualization:**
   - In `proxima/results/visualization/density_matrix.py`, define `DensityMatrixVisualizer` class
   - Method: `plot_real_imaginary(density_matrix) -> Figure`
   - Two subplots: real part and imaginary part
   - Heatmap representation with colorbar
   - Method: `plot_3d_bars(density_matrix) -> Figure`
   - 3D bar chart representation (city skyline view)
   - Useful for small density matrices (up to 16x16)
   - Method: `plot_hinton_diagram(density_matrix) -> Figure`
   - Hinton diagram: square sizes represent magnitudes
   - Classic quantum visualization style

5. **Implement Fidelity Visualization:**
   - In `proxima/results/visualization/fidelity.py`, define `FidelityVisualizer` class
   - Method: `plot_fidelity_comparison(fidelities, labels) -> Figure`
   - Bar chart comparing fidelities across different executions/backends
   - Reference line at fidelity = 1.0 (perfect)
   - Color-code: green (>0.95), yellow (0.9-0.95), red (<0.9)
   - Method: `plot_fidelity_vs_noise(fidelity_data, noise_levels) -> Figure`
   - Line plot showing how fidelity degrades with noise

6. **Implement Correlation Heatmap:**
   - In `proxima/results/visualization/correlation.py`, define `CorrelationVisualizer` class
   - Method: `plot_bit_correlations(correlation_matrix, qubit_labels) -> Figure`
   - Heatmap of qubit correlations
   - Color scale: -1 (anti-correlated) to +1 (correlated)
   - Annotate cells with correlation values
   - Useful for identifying entangled qubit pairs

7. **Implement Time Series Visualization:**
   - For parameter sweeps or time evolution
   - In `proxima/results/visualization/timeseries.py`, define `TimeSeriesVisualizer` class
   - Method: `plot_evolution(measurement_sequences, parameter_values) -> Figure`
   - X-axis: parameter value or time step
   - Y-axis: probability
   - Multiple lines: different basis states
   - Shows how distribution evolves with parameter

8. **Implement Comparison Visualizations:**
   - In `proxima/results/visualization/comparison.py`, define `ComparisonVisualizer` class
   - Method: `plot_multi_backend_comparison(results_by_backend) -> Figure`
   - Side-by-side or overlaid histograms
   - Different colors per backend
   - Legend showing backend names
   - Method: `plot_difference_heatmap(result1, result2) -> Figure`
   - Heatmap showing probability difference for each state
   - Positive (green): higher in result1, Negative (red): higher in result2

9. **Implement Automated Report Figures:**
   - Method: `generate_standard_report_plots(results) -> List[Figure]`
   - Automatically generate comprehensive set of visualizations:
     - Measurement histogram
     - Statistics summary (box with key metrics)
     - Correlation heatmap (if multi-qubit)
     - Fidelity comparison (if multiple results)
   - Used for automated report generation
   - Save all figures to report directory

10. **Testing Visualization:**
    - Generate visualizations for various result types
    - Visual QA: ensure plots are clear and informative
    - Test with different configurations (styles, color schemes)
    - Test interactive plots (if implemented)
    - Test all plot types render without errors
    - Test plot export to various formats (PNG, SVG, PDF)

### Step 9.5: Integration and Result Workflow

**Objective:** Integrate all result interpretation components into cohesive workflow accessible from UI and commands.

**Detailed Instructions:**

1. **Create Result Interpreter Orchestrator:**
   - In `proxima/results/interpreter.py`, define `ResultInterpreter` singleton
   - Coordinates all interpretation components
   - Method: `interpret(result_source, options) -> InterpretationReport`
   - InterpretationReport dataclass:
     - `parsed_data` (ParsedResult)
     - `statistics` (BasicStatistics)
     - `llm_interpretation` (Interpretation)
     - `visualizations` (List[Figure])
     - `anomalies` (List[Anomaly])
     - `recommendations` (List[Recommendation])

2. **Implement Interpretation Pipeline:**
   - Method: `run_interpretation_pipeline(result_file, config) -> InterpretationReport`
   - Pipeline steps:
     1. Parse result file (auto-detect format)
     2. Validate parsed data
     3. Compute statistical metrics
     4. Generate visualizations
     5. Request LLM interpretation (with consent)
     6. Detect anomalies
     7. Generate recommendations
     8. Combine into report
   - Each step logs progress
   - Configurable: can skip LLM step or visualization step

3. **Implement Result Viewer TUI Screen:**
   - Create `ResultViewerScreen(Screen)` in `proxima/ui/screens/result_viewer.py`
   - Layout:
     - Top: result summary (shots, backend, timestamp)
     - Left panel: measurement histogram (ASCII art or Unicode)
     - Right panel: statistics and metrics
     - Bottom: interpretation and insights (scrollable)
   - Keyboard shortcuts:
     - v: toggle visualization view (show plots in separate window or inline)
     - i: show full LLM interpretation
     - r: show recommendations
     - e: export results to file
     - c: compare with another result

4. **Implement Command Interface:**
   - Command: `interpret <result_file>` - run full interpretation pipeline
   - Command: `analyze <result_file>` - statistical analysis only (no LLM)
   - Command: `visualize <result_file>` - generate plots only
   - Command: `compare results <file1> <file2>` - compare two results
   - Options:
     - `--skip-llm` - skip LLM interpretation
     - `--export-report <path>` - save report to file
     - `--format <format>` - output format (text, markdown, html)

5. **Implement Auto-Interpretation After Execution:**
   - After circuit execution completes, offer to interpret results
   - Config: `results.auto_interpret` (default true)
   - If enabled:
     - Automatically run interpretation pipeline
     - Display results in Result Viewer
     - Save interpretation to session history
   - User can disable for batch jobs where interpretation not needed

6. **Implement Result Comparison Workflow:**
   - Method: `compare_results(result_list, comparison_type) -> ComparisonReport`
   - Comparison types:
     - "backends" - compare same circuit on different backends
     - "parameters" - compare parameter sweep results
     - "temporal" - compare same circuit over time (detect drift)
   - ComparisonReport includes:
     - Distribution comparisons (metrics like TVD, KL divergence)
     - Statistical summaries for each result
     - Visualizations showing differences
     - LLM interpretation of differences
     - Recommendations based on comparison

7. **Implement Result History and Lookup:**
   - Store interpreted results in database
   - Table: `interpreted_results` with columns:
     - id, execution_id, result_file_path, interpretation_json, timestamp
   - Command: `history results` - show recent interpreted results
   - Command: `recall result <id>` - load previous interpretation
   - Quick access to past interpretations without re-running LLM

8. **Implement Export Functionality:**
   - Method: `export_interpretation(report, format, filepath)`
   - Export formats:
     - **Text**: plain text summary
     - **Markdown**: formatted report with embedded images
     - **HTML**: interactive report with plotly charts
     - **PDF**: publication-ready report
     - **JSON**: structured data for programmatic access
   - Include all components: data, statistics, interpretation, plots
   - Self-contained (embed images as base64 in HTML/PDF)

9. **Implement Batch Interpretation:**
   - For interpreting multiple results at once
   - Command: `interpret batch <directory>` - interpret all result files in directory
   - Generate comparative report across all results
   - Useful for parameter sweeps or multi-run experiments
   - Progress bar showing interpretation progress

10. **Testing Integration:**
    - End-to-end test: execute circuit → auto-interpret → view results
    - Test all interpretation pipeline steps
    - Test Result Viewer UI with various result types
    - Test command interface with different options
    - Test result comparison workflow
    - Test export to all formats
    - Test batch interpretation with multiple files

### Phase 9 Implementation Complete

**Artifacts Produced:**
- Result parsers for CSV, XLSX, JSON formats
- Statistical analysis with entropy, fidelity, correlation metrics
- LLM-powered interpretation with insights and recommendations
- Comprehensive visualization suite (histograms, density matrices, correlations, comparisons)
- Result Interpreter orchestrator with full pipeline
- Result Viewer TUI screen
- Command interface for interpretation
- Result comparison and history functionality
- Multi-format export (text, markdown, HTML, PDF, JSON)

**Verification:**
- All supported result formats parse correctly
- Statistical metrics are accurate (validated against analytical examples)
- LLM interpretations are physically accurate and helpful
- Visualizations are clear and informative
- Full interpretation pipeline runs without errors
- Result Viewer displays all information clearly
- Commands execute successfully
- Export produces valid output in all formats
- Tests pass with coverage >80%

---

## PHASE 10: MULTI-BACKEND COMPARISON

### Step 10.1: Design Comparison Framework

**Objective:** Create infrastructure for systematically comparing quantum simulation results across multiple backends with robust metrics and analysis.

**Detailed Instructions:**

1. **Define Comparison Task Model:**
   - In `proxima/comparison/types.py`, create dataclasses:
     - `ComparisonTask`:
       - `task_id` (str) - unique identifier
       - `circuit` (Circuit) - circuit to compare across backends
       - `backends` (List[str]) - backend IDs to compare
       - `shots` (int) - shots per backend
       - `options` (dict) - execution options (same for all backends)
       - `comparison_metrics` (List[str]) - which metrics to compute
       - `created_at` (datetime)
       - `status` (TaskStatus)
     - `BackendExecutionResult`:
       - `backend_id` (str)
       - `result` (BackendResult) - raw result
       - `execution_time` (float)
       - `memory_used` (float)
       - `success` (bool)
       - `error` (Optional[str])
     - `ComparisonResult`:
       - `task_id` (str)
       - `executions` (Dict[str, BackendExecutionResult]) - results by backend
       - `metrics` (ComparisonMetrics)
       - `analysis` (ComparisonAnalysis)
       - `timestamp` (datetime)

2. **Define Comparison Metrics:**
   - In `proxima/comparison/metrics.py`, create `ComparisonMetrics` dataclass:
     - **Result Agreement Metrics:**
       - `pairwise_fidelities` (Dict[Tuple[str, str], float]) - fidelity between each pair
       - `total_variation_distances` (Dict[Tuple[str, str], float])
       - `kl_divergences` (Dict[Tuple[str, str], float])
       - `consensus_result` (Optional[MeasurementData]) - if backends agree
       - `agreement_score` (float, 0-1) - overall agreement across all backends
     - **Performance Metrics:**
       - `execution_times` (Dict[str, float]) - time per backend
       - `speedup_ratios` (Dict[Tuple[str, str], float]) - relative speedups
       - `memory_usage` (Dict[str, float]) - memory per backend
       - `efficiency_scores` (Dict[str, float]) - time * memory product
     - **Quality Metrics:**
       - `statistical_errors` (Dict[str, float]) - estimated sampling errors
       - `fidelity_to_ideal` (Dict[str, float]) - if ideal result known
       - `noise_levels` (Dict[str, float]) - inferred noise from results

3. **Create Comparison Configuration:**
   - In config schema, add `comparison` section:
     - `parallel_execution` (bool, default true) - run backends in parallel
     - `timeout_per_backend` (float, default 300) - seconds
     - `require_all_success` (bool, default false) - abort if any backend fails
     - `default_metrics` (List[str]) - metrics to compute by default
     - `consensus_threshold` (float, default 0.95) - fidelity threshold for consensus

4. **Implement Comparison Task Builder:**
   - In `proxima/comparison/task_builder.py`, define `ComparisonTaskBuilder` class
   - Fluent API:
     - `create_task(circuit) -> ComparisonTaskBuilder`
     - `add_backend(backend_id) -> ComparisonTaskBuilder`
     - `add_backends(backend_ids) -> ComparisonTaskBuilder`
     - `set_shots(shots) -> ComparisonTaskBuilder`
     - `set_options(**options) -> ComparisonTaskBuilder`
     - `compare_metrics(metric_list) -> ComparisonTaskBuilder`
     - `build() -> ComparisonTask`
   - Validation:
     - At least 2 backends specified
     - All backends available
     - Circuit compatible with all backends

### Step 10.2: Implement Parallel Backend Execution

**Objective:** Execute same circuit on multiple backends simultaneously with proper resource management and error handling.

**Detailed Instructions:**

1. **Create Comparison Executor:**
   - In `proxima/comparison/executor.py`, define `ComparisonExecutor` class
   - Method: `execute_comparison(task: ComparisonTask) -> ComparisonResult`
   - Orchestrates execution across multiple backends
   - Tracks execution status for each backend

2. **Implement Parallel Execution Logic:**
   - Method: `execute_backends_parallel(task) -> Dict[str, BackendExecutionResult]`
   - Use ThreadPoolExecutor or asyncio for concurrent execution
   - Create separate execution thread/task for each backend
   - Maximum parallelism: min(num_backends, config.max_parallel_backends)
   - Each thread:
     1. Get backend from registry
     2. Check resources before execution
     3. Execute circuit with specified options
     4. Collect result and metadata
     5. Return BackendExecutionResult
   - Handle partial failures: continue with successful backends

3. **Implement Sequential Execution Fallback:**
   - Method: `execute_backends_sequential(task) -> Dict[str, BackendExecutionResult]`
   - Used when parallel_execution disabled or resource-constrained
   - Execute backends one at a time
   - Safer for memory-intensive circuits
   - Show progress: "Executing backend 2/5: Cirq DM"

4. **Implement Resource-Aware Scheduling:**
   - Before starting parallel execution, estimate total resource needs
   - Sum estimated memory for all backends
   - If total > available memory: reduce parallelism or use sequential
   - Dynamic scheduling: start backends as resources become available
   - Priority ordering: execute faster backends first for early results

5. **Implement Timeout Handling:**
   - Each backend execution has timeout (configurable per task)
   - Use timeout context or thread timeout
   - If backend exceeds timeout:
     - Terminate execution gracefully
     - Mark as failed with timeout error
     - Continue with other backends
   - Display warning: "Backend X timed out after 300s"

6. **Implement Progress Tracking:**
   - Track execution status for each backend:
     - PENDING, RUNNING, COMPLETED, FAILED, TIMED_OUT
   - Emit progress events for UI updates
   - Show live progress: "3/5 backends completed, 2 running"
   - Estimate time remaining based on slowest backend

7. **Implement Result Collection and Normalization:**
   - As each backend completes, collect result
   - Normalize all results to common format
   - Handle different result structures (some backends return more data)
   - Store normalized results in ComparisonResult

8. **Testing Parallel Execution:**
   - Test with 2-5 backends
   - Test with fast and slow backends (timeout)
   - Test with intentional backend failures
   - Test resource-aware scheduling prevents overload
   - Test parallel execution is faster than sequential
   - Test progress tracking updates correctly

### Step 10.3: Implement Comparison Metrics Computation

**Objective:** Compute comprehensive metrics comparing results across backends.

**Detailed Instructions:**

1. **Implement Pairwise Comparisons:**
   - In `proxima/comparison/metrics_calculator.py`, define `MetricsCalculator` class
   - Method: `compute_pairwise_fidelities(results) -> Dict[Tuple[str, str], float]`
   - For each pair of backends:
     - Compute classical fidelity between measurement distributions
     - F = Σ √(p_i q_i) where p, q are probability distributions
   - Store in dictionary keyed by backend pair: (backend_a, backend_b)
   - Method: `compute_pairwise_tvd(results) -> Dict[Tuple[str, str], float]`
   - Total Variation Distance for each pair

2. **Implement Consensus Detection:**
   - Method: `detect_consensus(results, threshold) -> Optional[ConsensusResult]`
   - Check if all backends agree within threshold
   - Compute average fidelity across all pairs
   - If average > threshold: consensus exists
   - ConsensusResult:
     - `consensus_distribution` (MeasurementData) - average distribution
     - `agreement_score` (float)
     - `outlier_backends` (List[str]) - backends that disagree
   - If no consensus: identify which backends form clusters (agreeing subsets)

3. **Implement Performance Comparisons:**
   - Method: `compute_speedup_ratios(execution_times) -> Dict[Tuple[str, str], float]`
   - For each pair: speedup = time_b / time_a
   - Identify fastest backend
   - Method: `compute_efficiency_scores(times, memory) -> Dict[str, float]`
   - Efficiency = 1 / (time * memory)
   - Balances speed and memory usage

4. **Implement Statistical Significance Testing:**
   - Method: `test_distribution_differences(result_a, result_b, shots) -> TestResult`
   - Chi-squared test: are distributions significantly different?
   - Account for sampling error
   - Return p-value and conclusion
   - Used to determine if differences are meaningful or just noise

5. **Implement Quality Assessment:**
   - If ideal/expected result provided:
     - Method: `assess_backend_accuracy(results, ideal) -> Dict[str, float]`
     - Compute fidelity to ideal for each backend
     - Rank backends by accuracy
   - If no ideal result:
     - Method: `infer_ground_truth(results) -> MeasurementData`
     - Use majority voting or weighted average
     - Backends with high agreement have more weight

6. **Implement Divergence Analysis:**
   - Method: `analyze_divergences(results) -> DivergenceAnalysis`
   - Identify where backends disagree most:
     - Which basis states show largest probability differences?
     - Which backend pairs are most divergent?
   - DivergenceAnalysis dataclass:
     - `max_divergence_states` (List[str]) - states with largest disagreement
     - `most_divergent_pair` (Tuple[str, str])
     - `divergence_causes` (List[str]) - potential reasons (noise models, precision, etc.)

7. **Implement Metric Aggregation:**
   - Method: `aggregate_metrics(pairwise_metrics) -> AggregateMetrics`
   - Compute summary statistics across all pairs:
     - Mean, median, std dev of fidelities
     - Best and worst agreement pairs
     - Overall agreement score
   - AggregateMetrics used for high-level summary

8. **Testing Metrics Computation:**
   - Test with identical results (should show perfect agreement)
   - Test with completely different results (low fidelity)
   - Test with subtle differences (sampling error vs real difference)
   - Verify statistical tests correctly identify significance
   - Test consensus detection finds clusters

### Step 10.4: Implement Comparison Analysis and Insights

**Objective:** Generate deep analysis and LLM-powered insights explaining comparison results.

**Detailed Instructions:**

1. **Create Comparison Analyzer:**
   - In `proxima/comparison/analyzer.py`, define `ComparisonAnalyzer` class
   - Method: `analyze_comparison(comparison_result) -> ComparisonAnalysis`
   - ComparisonAnalysis dataclass:
     - `summary` (str) - overall summary
     - `backend_rankings` (Dict[str, Dict]) - rankings by various criteria
     - `key_findings` (List[str]) - important observations
     - `discrepancy_explanations` (List[str]) - why backends differ
     - `recommendations` (List[str]) - which backend to use when

2. **Implement Backend Ranking:**
   - Method: `rank_backends(results, metrics, criteria) -> Dict[str, Ranking]`
   - Ranking criteria:
     - "accuracy" - closest to ideal/consensus
     - "speed" - fastest execution
     - "efficiency" - best time/memory trade-off
     - "reliability" - most consistent with others
   - Ranking dataclass:
     - `rank` (int) - 1 is best
     - `score` (float)
     - `rationale` (str) - why this rank
   - Multi-criteria ranking: weighted average

3. **Implement LLM-Powered Comparison Explanation:**
   - Method: `explain_comparison(comparison_result) -> str`
   - Prompt template in `proxima/intelligence/prompts/backend_comparison.jinja2`:
     ```
     Compare quantum simulation results across multiple backends:
     
     Circuit: {{ circuit_description }}
     Backends tested: {{ backend_list }}
     
     Results Summary:
     {% for backend, result in results.items() %}
     {{ backend }}:
     - Top state: |{{ result.top_state }}⟩ ({{ result.top_prob }})
     - Execution time: {{ result.time }}s
     - Memory: {{ result.memory }}MB
     {% endfor %}
     
     Agreement Metrics:
     - Average fidelity: {{ avg_fidelity }}
     - Consensus: {{ consensus_status }}
     
     Performance:
     - Fastest: {{ fastest_backend }} ({{ fastest_time }}s)
     - Most memory-efficient: {{ efficient_backend }}
     
     Provide analysis:
     1. Degree of agreement between backends
     2. Explanation for any significant discrepancies
     3. Performance trade-offs observed
     4. Recommendations for which backend to use in different scenarios
     5. Confidence in results based on cross-backend validation
     ```
   - Request LLM consent
   - Generate comprehensive explanation

4. **Implement Discrepancy Root Cause Analysis:**
   - Method: `diagnose_discrepancies(comparison_result) -> List[Diagnosis]`
   - For backends that disagree significantly:
     - Identify potential causes:
       - Different noise models (DM vs SV)
       - Numerical precision differences
       - Different optimization/transpilation
       - Backend bugs or issues
       - Different qubit ordering/endianness
   - Diagnosis dataclass:
     - `backends_involved` (List[str])
     - `discrepancy_type` (str) - "noise", "precision", "bug", "ordering"
     - `evidence` (str) - what suggests this cause
     - `severity` (str)
     - `recommendation` (str)
   - Use pattern matching and heuristics
   - Optionally use LLM for complex diagnoses

5. **Implement Confidence Scoring:**
   - Method: `compute_result_confidence(comparison_result) -> ConfidenceScore`
   - ConfidenceScore dataclass:
     - `overall_confidence` (float, 0-1)
     - `consensus_confidence` (float) - how well backends agree
     - `execution_confidence` (float) - all executions successful?
     - `quality_confidence` (float) - statistical quality of results
     - `explanation` (str) - what affects confidence
   - High confidence: all backends agree, all successful, low sampling error
   - Low confidence: discrepancies, failures, high sampling error

6. **Implement Recommendation Generation:**
   - Method: `generate_backend_recommendations(comparison_result) -> List[Recommendation]`
   - Recommendations for future use:
     - "Use {backend} for best speed-accuracy balance"
     - "Avoid {backend} for this circuit type (poor accuracy)"
     - "Consider {backend} with GPU for large circuits"
     - "Results are backend-independent (high confidence)"
   - Actionable recommendations with justifications

7. **Implement Historical Comparison Trends:**
   - Track comparison results over time
   - Method: `analyze_trends(historical_comparisons) -> TrendAnalysis`
   - Detect:
     - Backend performance drift (getting faster/slower)
     - Accuracy changes (after backend updates)
     - New discrepancy patterns
   - TrendAnalysis useful for long-term validation

8. **Testing Comparison Analysis:**
   - Test with various agreement scenarios (high, medium, low)
   - Test LLM explanations are accurate and helpful
   - Test discrepancy diagnosis identifies correct causes
   - Test confidence scoring reflects actual reliability
   - Test recommendations are sensible

### Step 10.5: Implement Comparison Visualization and Reporting

**Objective:** Create comprehensive visualizations and reports for multi-backend comparisons.

**Detailed Instructions:**

1. **Implement Comparative Histogram:**
   - In `proxima/comparison/visualization/comparative_plots.py`, define plotting functions
   - Function: `plot_comparative_histogram(results_by_backend) -> Figure`
   - Grouped bar chart:
     - X-axis: basis states (top N most probable)
     - Y-axis: probability
     - Bars grouped by state, colored by backend
     - Legend showing backend names
   - Side-by-side subplots alternative: one subplot per backend

2. **Implement Agreement Matrix Heatmap:**
   - Function: `plot_agreement_matrix(pairwise_fidelities) -> Figure`
   - Heatmap showing fidelity between each backend pair
   - Rows and columns: backend names
   - Cell color: fidelity value (green high, red low)
   - Annotate cells with numeric fidelities
   - Diagonal is always 1.0 (backend vs itself)
   - Symmetric matrix

3. **Implement Performance Comparison Charts:**
   - Function: `plot_performance_comparison(execution_times, memory_usage) -> Figure`
   - Two subplots:
     - Execution time bar chart
     - Memory usage bar chart
   - Backends sorted by metric value
   - Reference line showing average
   - Color-code: green (best), yellow (medium), red (worst)

4. **Implement Distribution Difference Plots:**
   - Function: `plot_distribution_differences(result_a, result_b) -> Figure`
   - Bar chart showing probability difference for each state
   - Positive bars: higher in result_a
   - Negative bars: higher in result_b
   - Useful for detailed comparison of two specific backends

5. **Implement Radar Chart for Multi-Dimensional Comparison:**
   - Function: `plot_radar_comparison(backend_scores, dimensions) -> Figure`
   - Dimensions: accuracy, speed, memory efficiency, reliability
   - One trace per backend
   - Visual representation of trade-offs
   - Ideal backend would fill entire radar

6. **Implement Comparison Dashboard:**
   - Create TUI screen: `ComparisonDashboardScreen`
   - Layout with multiple panels:
     - Top: summary metrics (agreement score, fastest, most accurate)
     - Left: backend list with status indicators
     - Center: current visualization (switchable)
     - Right: analysis text and recommendations
     - Bottom: command bar
   - Keyboard shortcuts:
     - 1-5: switch between visualization types
     - a: show full analysis
     - r: show recommendations
     - e: export comparison report
     - d: show detailed metrics table

7. **Implement Comparison Report Generation:**
   - Method: `generate_comparison_report(comparison_result) -> Report`
   - Report structure:
     - **Executive Summary**: agreement score, key findings, recommendation
     - **Execution Details**: table of execution times, memory, status per backend
     - **Result Comparison**: comparative histogram, agreement matrix
     - **Performance Analysis**: speed and efficiency comparison
     - **Quality Assessment**: accuracy rankings, confidence scores
     - **Discrepancy Analysis**: explanation of disagreements
     - **Recommendations**: which backend for which scenario
     - **Appendix**: full metrics table, raw data
   - Use LLM to generate narrative sections (with consent)

8. **Implement Interactive Comparison Explorer:**
   - For detailed exploration of comparison results
   - Features:
     - Select two backends: show detailed difference plot
     - Filter states: show only discrepant states
     - Adjust agreement threshold: see how consensus changes
     - Export selected data
   - Command: `explore comparison <comparison_id>`

9. **Implement Batch Comparison Reporting:**
   - When comparing many circuits across backends
   - Method: `generate_batch_comparison_report(comparisons) -> BatchReport`
   - Aggregate findings across all circuits:
     - Which backend is consistently best?
     - Which circuits show most backend sensitivity?
     - Overall reliability rankings
   - Include summary visualizations

10. **Testing Visualization and Reporting:**
    - Generate all visualization types with sample data
    - Visual QA: ensure plots are clear and informative
    - Test comparison dashboard with various results
    - Test report generation includes all sections
    - Test interactive explorer allows detailed investigation
    - Test batch reporting aggregates correctly

### Step 10.6: Integration and Workflow

**Objective:** Integrate multi-backend comparison throughout Proxima with seamless user experience.

**Detailed Instructions:**

1. **Implement Comparison Command Interface:**
   - Command: `compare backends <backend1> <backend2> ... <backendN>`
     - Prompts for circuit selection
     - Runs comparison
     - Displays results
   - Command: `compare backends --all`
     - Compares all available backends
   - Command: `compare backends --fast`
     - Compares only fast statevector backends
   - Options:
     - `--shots N` - shots per backend
     - `--parallel` / `--sequential` - execution mode
     - `--export <path>` - save report to file

2. **Implement Quick Comparison from Execution:**
   - After circuit execution, offer to run comparison
   - UI prompt: "Compare with other backends? (y/n)"
   - If yes: show backend selection menu
   - Run comparison with same circuit and options

3. **Implement Comparison Templates:**
   - Predefined comparison scenarios:
     - "Speed Test": compare execution times across all backends
     - "Accuracy Test": compare against known ideal result
     - "DM vs SV": compare density matrix vs statevector backends
     - "Local vs Remote": compare local and remote options
   - Command: `compare template speed-test`

4. **Implement Comparison History:**
   - Store all comparison results in database
   - Table: `backend_comparisons` with columns:
     - id, circuit_hash, backends_json, results_json, timestamp
   - Command: `history comparisons` - list past comparisons
   - Command: `recall comparison <id>` - load previous comparison

5. **Implement Smart Backend Selection from Comparison:**
   - After comparison, user can select preferred backend for future runs
   - Command: `pin backend <backend>` based on comparison results
   - Automatically update backend preference in config

6. **Implement Continuous Validation:**
   - Option to periodically run comparisons on sample circuits
   - Detect backend regression or divergence over time
   - Config: `comparison.continuous_validation.enabled` (default false)
   - Config: `comparison.continuous_validation.interval_hours` (default 24)
   - Alert user if discrepancies increase beyond threshold

7. **Implement Comparison in Planning:**
   - Add comparison as stage type in planning engine
   - Stage: MULTI_BACKEND_COMPARISON automatically handled by comparison executor
   - Integrates comparison into multi-step workflows

8. **Implement Export and Sharing:**
   - Export comparison results to shareable formats
   - Command: `export comparison <id> --format <format>`
   - Formats: JSON (data), HTML (interactive), PDF (report)
   - Include reproducibility info (circuit, backends, versions)

9. **Testing Integration:**
   - End-to-end test: run comparison from command → view dashboard → export report
   - Test comparison history save/recall
   - Test integration with planning engine
   - Test continuous validation runs periodically
   - Test all command options work correctly

### Phase 10 Implementation Complete

**Artifacts Produced:**
- Comparison task model and builder
- Parallel backend executor with resource-aware scheduling
- Comprehensive metrics calculator (fidelity, TVD, performance)
- Consensus detection and outlier identification
- LLM-powered comparison analysis with explanations
- Discrepancy diagnosis and confidence scoring
- Comparative visualizations (histograms, heatmaps, radar charts)
- Comparison dashboard TUI screen
- Automated report generation
- Command interface for comparisons
- Comparison history and templates
- Integration with planning engine

**Verification:**
- Parallel execution runs multiple backends correctly
- Metrics accurately quantify agreement and performance
- Consensus detection identifies when backends agree
- LLM explanations are accurate and helpful
- Visualizations clearly show comparisons
- Comparison dashboard displays all information
- Reports are comprehensive and well-formatted
- Commands execute successfully
- History tracking works correctly
- Tests pass with coverage >80%

---

## PHASE 11: PAUSE/RESUME/ROLLBACK

### Step 11.1: Design Checkpoint System Architecture

**Objective:** Create a robust checkpoint system that captures complete execution state for pause/resume and rollback capabilities.

**Detailed Instructions:**

1. **Define Checkpoint Data Model:**
   - In `proxima/checkpoint/types.py`, create dataclasses:
     - `Checkpoint`:
       - `checkpoint_id` (str) - unique identifier (UUID)
       - `execution_id` (str) - which execution this checkpoint belongs to
       - `checkpoint_type` (CheckpointType enum) - MANUAL, AUTO, PRE_STAGE, POST_STAGE, EMERGENCY
       - `timestamp` (datetime) - when checkpoint was created
       - `state_snapshot` (ExecutionState) - complete execution state
       - `description` (str) - user or system description
       - `file_path` (Path) - where checkpoint data is stored
       - `size_bytes` (int) - checkpoint file size
       - `compression` (str) - compression method used
     - `ExecutionState`:
       - `execution_metadata` (dict) - execution info (backend, circuit, options)
       - `stage_states` (Dict[str, StageState]) - state of each stage in plan
       - `variable_context` (dict) - all runtime variables
       - `result_data` (dict) - results produced so far
       - `resource_state` (ResourceSnapshot) - memory, CPU at checkpoint time
       - `ui_state` (dict) - UI state for seamless resume
     - `CheckpointMetadata`:
       - `checkpoint_id` (str)
       - `timestamp` (datetime)
       - `execution_name` (str)
       - `progress_percentage` (float)
       - `can_resume` (bool) - is checkpoint valid for resume?
       - `dependencies` (List[str]) - files/resources needed for resume

2. **Define Checkpoint Types and Triggers:**
   - CheckpointType enum:
     - `MANUAL` - user explicitly requested checkpoint
     - `AUTO` - automatic checkpoint based on time/stage interval
     - `PRE_STAGE` - before important stage execution
     - `POST_STAGE` - after stage completion
     - `EMERGENCY` - system emergency (crash, resource exhaustion)
   - Checkpoint triggers configuration:
     - `checkpoint.auto_interval_minutes` (default 10) - time-based auto checkpoint
     - `checkpoint.auto_on_stage_completion` (default true) - checkpoint after each stage
     - `checkpoint.checkpoint_before_risky_stages` (default true) - checkpoint before long/risky stages

3. **Design Checkpoint Storage Strategy:**
   - Store checkpoints in `data/checkpoints/{execution_id}/`
   - File naming: `checkpoint_{checkpoint_id}_{timestamp}.pkl` or `.json`
   - Compression options: none, gzip, bz2, lzma
   - Config: `checkpoint.compression` (default "gzip")
   - Large data (arrays, matrices) stored separately: `checkpoint_{id}_data.npz`
   - Metadata index: `checkpoints_index.json` for quick lookup

4. **Implement Checkpoint Size Management:**
   - Limit number of checkpoints per execution
   - Config: `checkpoint.max_checkpoints_per_execution` (default 10)
   - Retention policy: keep manual checkpoints, prune old auto checkpoints
   - Size limits: warn if checkpoint > 100MB
   - Cleanup old checkpoints: command `cleanup checkpoints --older-than 30d`

5. **Define Rollback Points:**
   - Rollback point: special checkpoint that can be restored to undo changes
   - Create rollback point before:
     - Starting execution
     - Applying configuration changes
     - Running destructive operations
   - RollbackPoint extends Checkpoint with additional metadata:
     - `rollback_reason` (str) - why this rollback point exists
     - `affected_components` (List[str]) - what would be reverted

### Step 11.2: Implement Checkpoint Creation

**Objective:** Implement comprehensive state capture for creating checkpoints.

**Detailed Instructions:**

1. **Create Checkpoint Manager:**
   - In `proxima/checkpoint/manager.py`, define `CheckpointManager` singleton
   - Responsibilities:
     - Create checkpoints
     - List available checkpoints
     - Load checkpoint data
     - Delete checkpoints
     - Validate checkpoint integrity
   - Dependencies: serialization, compression, file system access

2. **Implement State Serialization:**
   - In `proxima/checkpoint/serializer.py`, define `StateSerializer` class
   - Method: `serialize_state(execution_state) -> bytes`
   - Serialization strategy:
     - Use pickle for Python objects (fastest, most complete)
     - Alternative: JSON for human-readable checkpoints (config option)
     - Custom serializers for special types:
       - Numpy arrays → save to .npz file separately
       - Large dataframes → parquet format
       - Circuit objects → to QASM or internal format
   - Handle non-serializable objects:
     - Skip transient objects (locks, file handles)
     - Convert to serializable representations
     - Store references instead of full objects where appropriate

3. **Implement Execution State Capture:**
   - Method: `capture_execution_state() -> ExecutionState`
   - Capture from execution controller:
     - Current execution metadata (backend, circuit, shots, options)
     - Execution timeline (start time, elapsed time)
     - Current stage and progress
   - Capture from plan orchestrator (if in plan execution):
     - All stage states (pending, running, completed)
     - Stage outputs (results from completed stages)
     - Dependency graph state
   - Capture from execution context:
     - All variables in context
     - Global state
   - Capture from result interpreter:
     - Parsed results
     - Analysis results
     - Interpretations

4. **Implement Resource State Snapshot:**
   - Method: `capture_resource_state() -> ResourceSnapshot`
   - Capture from resource monitor:
     - Current memory usage
     - Current CPU usage
     - GPU state (if using GPU)
     - Open file handles count
   - Store resource limits that were in effect
   - Used for validating resume feasibility

5. **Implement UI State Capture:**
   - Method: `capture_ui_state() -> dict`
   - Capture from TUI:
     - Current screen/view
     - Scroll positions
     - Expanded/collapsed sections
     - User preferences
   - Allows seamless UI restoration on resume
   - Optional: config `checkpoint.include_ui_state` (default false)

6. **Implement Checkpoint Creation Flow:**
   - Method: `create_checkpoint(checkpoint_type, description) -> Checkpoint`
   - Steps:
     1. Generate unique checkpoint ID
     2. Capture execution state
     3. Capture resource state
     4. Capture UI state (if configured)
     5. Serialize state data
     6. Compress if configured
     7. Write to checkpoint file
     8. Save separate data files (numpy arrays, etc.)
     9. Create checkpoint metadata
     10. Update checkpoint index
     11. Enforce retention policy (delete old checkpoints if over limit)
     12. Verify checkpoint integrity
     13. Return Checkpoint object
   - Handle errors: if checkpoint creation fails, log error but don't crash execution

7. **Implement Incremental Checkpointing:**
   - Optimization: only save changed state since last checkpoint
   - Method: `create_incremental_checkpoint(base_checkpoint_id) -> Checkpoint`
   - Store diff/delta instead of full state
   - Significantly reduces checkpoint size for large executions
   - Requires base checkpoint for restoration

8. **Implement Automatic Checkpoint Scheduling:**
   - In execution controller, schedule periodic checkpoints
   - Use timer or hook into event loop
   - Create checkpoint every N minutes (configurable)
   - Create checkpoint after each stage completion (if enabled)
   - Skip checkpoint if execution in critical section (configurable)

9. **Implement Emergency Checkpoint:**
   - Method: `create_emergency_checkpoint(reason) -> Checkpoint`
   - Called when:
     - Unhandled exception occurs
     - Resource exhaustion detected
     - User requests emergency stop
     - System shutdown signal received
   - Higher priority: written synchronously, no compression delay
   - Marked as EMERGENCY type for easy identification

10. **Testing Checkpoint Creation:**
    - Test checkpoint creation during various execution states
    - Test serialization of complex state (nested objects, numpy arrays)
    - Test compression reduces size without data loss
    - Test incremental checkpointing produces correct diffs
    - Test emergency checkpoint completes quickly
    - Test checkpoint index updated correctly

### Step 11.3: Implement Resume Functionality

**Objective:** Enable resuming execution from saved checkpoints with state restoration.

**Detailed Instructions:**

1. **Implement Checkpoint Loading:**
   - Method: `load_checkpoint(checkpoint_id) -> Checkpoint`
   - Steps:
     1. Locate checkpoint file from ID
     2. Verify checkpoint file integrity (checksum)
     3. Decompress if needed
     4. Deserialize state data
     5. Load separate data files (numpy arrays, etc.)
     6. Validate checkpoint structure
     7. Return Checkpoint object with loaded state
   - Handle errors: corrupted checkpoint, missing files, version mismatch

2. **Implement Checkpoint Validation:**
   - Method: `validate_checkpoint(checkpoint) -> ValidationResult`
   - Checks:
     - Checkpoint file exists and is readable
     - Checksum matches (detect corruption)
     - Required dependency files present
     - Checkpoint version compatible with current Proxima version
     - Required backends still available
     - Circuit files still accessible
   - ValidationResult:
     - `is_valid` (bool)
     - `errors` (List[str]) - blocking issues
     - `warnings` (List[str]) - potential issues but resumable
   - Display validation results to user before resume

3. **Implement State Restoration:**
   - Method: `restore_execution_state(execution_state) -> bool`
   - Restore to execution controller:
     - Set execution metadata
     - Restore timeline state
     - Set current stage/progress
   - Restore to plan orchestrator (if plan execution):
     - Restore stage states
     - Repopulate execution context with stage outputs
     - Rebuild dependency graph
     - Mark completed stages as completed
   - Restore to result interpreter:
     - Load parsed results
     - Load analysis results
   - Restore to backend registry:
     - Verify backends still available
     - Warn if backend configuration changed

4. **Implement Resume Decision Logic:**
   - Method: `can_resume(checkpoint) -> ResumeDecision`
   - ResumeDecision dataclass:
     - `can_resume` (bool)
     - `resume_point` (str) - where execution will resume (stage name, line number)
     - `blocking_issues` (List[str])
     - `warnings` (List[str])
     - `required_actions` (List[str]) - user must do before resume (e.g., "start GPU backend")
   - Check conditions:
     - All required backends available
     - Sufficient resources (memory, disk space)
     - No conflicting execution currently running
     - Circuit and dependencies accessible
   - Idempotency check: has this stage already completed? Skip if yes

5. **Implement Resume Flow:**
   - Method: `resume_execution(checkpoint_id) -> ResumeResult`
   - Steps:
     1. Load checkpoint
     2. Validate checkpoint
     3. Check if can resume
     4. Display resume plan to user:
        - "Will resume from stage X"
        - "Already completed: stages A, B, C"
        - "Remaining: stages D, E, F"
     5. Request user confirmation
     6. Restore execution state
     7. Restore resource monitor state
     8. Restore UI state (if captured)
     9. Update execution status to RESUMED
     10. Continue execution from resume point
   - ResumeResult:
     - `success` (bool)
     - `resumed_from` (str) - description of resume point
     - `error` (Optional[str])

6. **Implement Smart Resume Point Detection:**
   - Method: `determine_resume_point(execution_state) -> ResumePoint`
   - Analyze checkpoint state to find best resume point:
     - If in middle of stage execution: restart stage (safest)
     - If stage just completed: resume from next stage
     - If waiting for user input: resume at that point
   - ResumePoint dataclass:
     - `stage_id` (Optional[str])
     - `step_index` (Optional[int])
     - `description` (str)

7. **Implement Resume with Modifications:**
   - Allow user to modify execution before resuming
   - Method: `resume_with_modifications(checkpoint_id, modifications) -> ResumeResult`
   - Possible modifications:
     - Change backend for remaining stages
     - Skip certain stages
     - Modify stage parameters
     - Add new stages
   - Validate modifications don't break dependencies
   - Apply modifications to restored state before resuming

8. **Implement Resource Compatibility Check:**
   - Method: `check_resource_compatibility(checkpoint) -> CompatibilityResult`
   - Compare checkpoint's resource state with current system:
     - Was using 16GB RAM, now only 8GB available: warn user
     - Was using GPU, GPU no longer available: error or fallback to CPU
     - Was using specific backend version, now different: warn about potential differences
   - CompatibilityResult includes recommendations for resolving issues

9. **Implement UI Restoration:**
   - Method: `restore_ui_state(ui_state) -> bool`
   - If UI state was captured in checkpoint:
     - Restore screen/view
     - Restore scroll positions
     - Restore expanded sections
     - Show resume notification
   - Seamless experience: user sees exactly what they saw before pause

10. **Testing Resume Functionality:**
    - Test resume from various checkpoint types
    - Test resume with missing dependencies (should fail gracefully)
    - Test resume with modified backend configuration
    - Test resume continues execution correctly from checkpoint
    - Test resume with insufficient resources (should warn/fail)
    - Test UI restoration matches pre-checkpoint state

### Step 11.4: Implement Rollback Capability

**Objective:** Enable rolling back execution to previous state, undoing changes.

**Detailed Instructions:**

1. **Define Rollback Semantics:**
   - In `proxima/checkpoint/rollback.py`, define rollback behavior
   - Rollback restores state to a previous checkpoint
   - Reverses changes made after checkpoint:
     - Delete results produced after checkpoint
     - Reset stage states to checkpoint values
     - Restore configuration to checkpoint state
     - Clear cached data from rolled-back stages
   - Rollback types:
     - **Soft rollback**: reset state but keep result files
     - **Hard rollback**: delete results and reset state
     - **Selective rollback**: rollback specific stages only

2. **Implement Rollback Manager:**
   - In `proxima/checkpoint/rollback_manager.py`, define `RollbackManager` class
   - Method: `create_rollback_point(description) -> RollbackPoint`
   - Create checkpoint specifically for rollback
   - Store additional metadata for rollback:
     - Files created since this point
     - Configuration changes since this point
     - Database entries created since this point

3. **Implement Rollback Impact Analysis:**
   - Method: `analyze_rollback_impact(target_checkpoint) -> RollbackImpact`
   - RollbackImpact dataclass:
     - `stages_affected` (List[str]) - stages that will be undone
     - `results_deleted` (List[str]) - result files that will be deleted
     - `state_changes` (List[str]) - state modifications that will be reverted
     - `data_loss_risk` (str) - "none", "low", "medium", "high"
     - `reversible` (bool) - can this rollback be undone?
   - Display impact analysis to user before rollback

4. **Implement Rollback Execution:**
   - Method: `rollback_to_checkpoint(checkpoint_id, rollback_type) -> RollbackResult`
   - Steps:
     1. Load target checkpoint
     2. Analyze rollback impact
     3. Request user confirmation (show impact)
     4. Create safety checkpoint (in case user wants to undo rollback)
     5. Restore state from target checkpoint
     6. Clean up rolled-back data:
        - Delete result files (if hard rollback)
        - Delete database entries for rolled-back stages
        - Clear cached interpretations
     7. Update execution status to ROLLED_BACK
     8. Log rollback action with timestamp and reason
   - RollbackResult:
     - `success` (bool)
     - `rolled_back_to` (datetime)
     - `stages_reverted` (List[str])
     - `error` (Optional[str])

5. **Implement Selective Rollback:**
   - Method: `rollback_stages(stage_ids, checkpoint_id) -> RollbackResult`
   - Rollback specific stages without affecting others
   - Use case: stage produced incorrect results, want to re-run just that stage
   - Steps:
     1. Identify which stages depend on target stages (dependency graph)
     2. Mark target and dependent stages for rollback
     3. Restore state for those stages only
     4. Leave other stages intact
   - More complex but preserves more work

6. **Implement Rollback History:**
   - Track all rollback operations
   - Table: `rollback_history` with columns:
     - id, execution_id, target_checkpoint_id, rollback_type, timestamp, reason, user
   - Useful for auditing and understanding execution history
   - Command: `history rollbacks` - show rollback history

7. **Implement Rollback Verification:**
   - Method: `verify_rollback(checkpoint_before, checkpoint_after) -> bool`
   - After rollback, verify state matches target checkpoint
   - Compare:
     - Stage states
     - Variable context
     - File system state
   - Alert if verification fails (corruption?)

8. **Implement Undo Rollback:**
   - Method: `undo_rollback(rollback_id) -> bool`
   - Restore to state before rollback (using safety checkpoint)
   - Only possible if safety checkpoint was created
   - Use case: user rolled back by mistake
   - Command: `undo rollback`

9. **Implement Configuration Rollback:**
   - Separate from execution rollback: restore configuration settings
   - Method: `rollback_configuration(timestamp) -> bool`
   - Restore config file to previous state
   - Config changes are versioned in `data/config_history/`
   - Use case: user changed settings, wants to revert

10. **Testing Rollback:**
    - Test soft rollback preserves result files
    - Test hard rollback deletes results
    - Test selective rollback only affects target stages
    - Test rollback impact analysis correctly predicts changes
    - Test undo rollback restores pre-rollback state
    - Test rollback with dependencies correctly identifies affected stages

### Step 11.5: Implement Checkpoint Management and UI

**Objective:** Provide user interface and commands for managing checkpoints, resume, and rollback.

**Detailed Instructions:**

1. **Implement Checkpoint List View:**
   - Create TUI screen: `CheckpointManagerScreen`
   - Display table of all checkpoints:
     - Checkpoint ID (truncated)
     - Type (icon/color-coded)
     - Timestamp
     - Execution name
     - Progress percentage
     - Size
     - Description
   - Sort options: by time, by size, by type
   - Filter: by execution, by type, by date range

2. **Implement Checkpoint Detail View:**
   - Show detailed info for selected checkpoint:
     - Full metadata
     - Execution state summary (which stage, progress)
     - Resource state at checkpoint time
     - Can resume? (validation result)
     - Warnings/issues
     - Files included in checkpoint
   - Actions:
     - Resume from this checkpoint
     - Rollback to this checkpoint
     - Delete checkpoint
     - Export checkpoint

3. **Implement Checkpoint Browser:**
   - Navigate checkpoint history visually
   - Timeline view: checkpoints on timeline
   - Tree view: checkpoint hierarchy (incremental checkpoints)
   - Compare checkpoints: show state differences
   - Keyboard shortcuts:
     - Enter: view checkpoint details
     - r: resume from checkpoint
     - d: delete checkpoint
     - c: compare with another checkpoint

4. **Implement Command Interface:**
   - Command: `checkpoint create [description]` - manual checkpoint
   - Command: `checkpoint list` - list all checkpoints
   - Command: `checkpoint info <id>` - show checkpoint details
   - Command: `checkpoint delete <id>` - delete checkpoint
   - Command: `checkpoint cleanup` - delete old/invalid checkpoints
   - Command: `resume <checkpoint_id>` - resume from checkpoint
   - Command: `rollback <checkpoint_id>` - rollback to checkpoint
   - Options:
     - `--hard` / `--soft` for rollback type
     - `--force` to skip confirmation
     - `--dry-run` to preview without executing

5. **Implement Checkpoint Auto-Management:**
   - Background task: monitor checkpoint storage
   - Auto-cleanup:
     - Delete checkpoints older than retention period
     - Keep manual checkpoints indefinitely (or per config)
     - Delete checkpoints for deleted executions
     - Compress old checkpoints to save space
   - Config:
     - `checkpoint.retention_days` (default 30)
     - `checkpoint.keep_manual_forever` (default true)
     - `checkpoint.auto_compress_after_days` (default 7)

6. **Implement Checkpoint Export/Import:**
   - Method: `export_checkpoint(checkpoint_id, filepath) -> bool`
   - Package checkpoint with all dependencies
   - Create portable checkpoint archive (.tar.gz or .zip)
   - Include:
     - Checkpoint data
     - Circuit files
     - Result files
     - Metadata
     - README with restore instructions
   - Method: `import_checkpoint(filepath) -> Checkpoint`
   - Unpack checkpoint archive
   - Validate and register checkpoint
   - Use case: share checkpoints between machines or users

7. **Implement Pause Command:**
   - Command: `pause` - pause current execution
   - Flow:
     1. Signal execution controller to pause
     2. Wait for current stage to finish (graceful)
     3. Create checkpoint automatically
     4. Update status to PAUSED
     5. Display: "Execution paused. Resume with: resume <checkpoint_id>"
   - Urgent pause: `pause --immediate` stops mid-stage (less safe)

8. **Implement Auto-Resume on Restart:**
   - If Proxima exits during execution, offer to resume on next start
   - Check for recent unfinished executions on startup
   - Display: "Found paused execution from 2 hours ago. Resume? (y/n)"
   - If yes: load most recent checkpoint and resume
   - Config: `checkpoint.auto_resume_on_restart` (default true)

9. **Implement Checkpoint Notifications:**
   - Notify user when checkpoint created (optional notification)
   - Notify if checkpoint creation fails
   - Notify when approaching storage limits
   - Notify if checkpoint validation fails (corruption)
   - Config: `checkpoint.notifications_enabled` (default true)

10. **Implement Checkpoint Analytics:**
    - Track checkpoint usage statistics:
      - How often checkpoints used
      - How often resume used
      - Average checkpoint size
      - Checkpoint creation success rate
    - Display: `checkpoint stats` command
    - Helps understand checkpoint effectiveness

11. **Testing Checkpoint Management:**
    - Test checkpoint list/detail views display correctly
    - Test all commands execute successfully
    - Test pause command creates checkpoint and pauses
    - Test auto-resume on restart works correctly
    - Test checkpoint export/import preserves data
    - Test auto-cleanup removes old checkpoints
    - Test checkpoint browser navigation

### Step 11.6: Integration and Advanced Features

**Objective:** Integrate checkpoint system throughout Proxima and add advanced checkpoint features.

**Detailed Instructions:**

1. **Integrate with Execution Controller:**
   - Hook checkpoint creation into execution flow:
     - Before starting execution: create initial checkpoint (rollback point)
     - After stage completion: auto-checkpoint if configured
     - On pause request: checkpoint and pause
     - On error: emergency checkpoint
   - Resume flow integrated: load checkpoint, restore state, continue

2. **Integrate with Planning Engine:**
   - Plan stages can include checkpoint stages
   - StageType.CHECKPOINT automatically creates checkpoint
   - Dependencies: subsequent stages depend on checkpoint stage
   - Checkpoint named based on stage name for easy identification

3. **Implement Checkpoint-Based Debugging:**
   - Method: `debug_from_checkpoint(checkpoint_id) -> DebugSession`
   - Load checkpoint and enter debug mode
   - Inspect state: variables, stage outputs, resource usage
   - Step through remaining stages
   - Modify state and resume
   - Useful for investigating failures

4. **Implement Checkpoint Diff:**
   - Method: `diff_checkpoints(checkpoint_id_1, checkpoint_id_2) -> CheckpointDiff`
   - Compare two checkpoints:
     - What stages completed between them?
     - What variables changed?
     - What results produced?
   - CheckpointDiff dataclass with detailed differences
   - Display: `checkpoint diff <id1> <id2>`

5. **Implement Checkpoint Merging:**
   - Advanced: merge states from multiple execution branches
   - Use case: ran same circuit with different backends, want to merge results
   - Method: `merge_checkpoints(checkpoint_ids) -> MergedCheckpoint`
   - Combines results, handles conflicts
   - Experimental feature

6. **Implement Cloud Checkpoint Storage:**
   - Option to store checkpoints in cloud storage (S3, Azure Blob, GCS)
   - Config: `checkpoint.storage.type` (local, s3, azure, gcs)
   - Config: `checkpoint.storage.bucket` (bucket/container name)
   - Benefits:
     - Backup (don't lose checkpoints if local disk fails)
     - Share checkpoints across machines
     - Resume on different machine
   - Use boto3 (S3), azure-storage-blob, google-cloud-storage libraries

7. **Implement Checkpoint Integrity Monitoring:**
   - Background task: periodically verify checkpoint integrity
   - Compute checksums and verify files
   - Alert if corruption detected
   - Offer to recreate checkpoint from redundant data
   - Config: `checkpoint.integrity_check_interval_hours` (default 24)

8. **Implement Checkpoint Policies:**
   - Define policies for when to checkpoint:
     - Time-based: every N minutes
     - Stage-based: after specific stages
     - Resource-based: when memory usage high (preemptive)
     - Event-based: before risky operations
   - User can define custom policies
   - Policy engine evaluates conditions and triggers checkpoints

9. **Implement Checkpoint Compression Strategies:**
   - Different compression for different data:
     - Text data: gzip (fast, good compression)
     - Numpy arrays: blosc (very fast, good for numerical data)
     - Large files: lzma (slow but best compression)
   - Auto-select compression based on data type and size
   - Config: `checkpoint.compression_strategy` (fast, balanced, maximum)

10. **Testing Integration:**
    - End-to-end: execute → pause → resume → complete
    - Test checkpoint created at correct times
    - Test resume continues seamlessly
    - Test rollback undoes changes correctly
    - Test checkpoint debugging allows state inspection
    - Test cloud storage upload/download (if configured)
    - Test integrity monitoring detects corruption

### Phase 11 Implementation Complete

**Artifacts Produced:**
- Checkpoint data model with full state capture
- Checkpoint manager with create/load/delete operations
- State serialization with compression support
- Resume functionality with validation and compatibility checks
- Rollback capability (soft, hard, selective)
- Checkpoint management UI (list, detail, browser)
- Command interface (checkpoint, resume, rollback, pause)
- Auto-checkpoint scheduling and management
- Export/import for portable checkpoints
- Cloud storage integration
- Checkpoint debugging and diffing tools

**Verification:**
- Checkpoints capture complete execution state
- Resume restores state and continues correctly
- Rollback reverts to previous state successfully
- Pause/resume cycle works seamlessly
- Checkpoint management UI displays all information
- Commands execute successfully
- Auto-checkpoint creates checkpoints at correct times
- Export/import preserves checkpoint data
- Cloud storage (if configured) syncs checkpoints
- Tests pass with coverage >80%

---

## PHASE 12: AGENT INSTRUCTION SYSTEM

### Step 12.1: Design Agent Instruction Parser

**Objective:** Create a system to parse and interpret instructions from proxima_agent.md files, enabling automated agent workflows.

**Detailed Instructions:**

1. **Define Agent Instruction Format:**
   - In `proxima/agent/instruction_format.md`, document instruction format specification:
     - Instructions in Markdown format
     - Structured sections with headers
     - YAML frontmatter for metadata
     - Code blocks for circuits and configurations
     - Special directives: `@proxima:command`, `@proxima:condition`, `@proxima:loop`
   - Example structure:
     ```markdown
     ---
     agent: proxima
     version: 1.0
     ---
     
     # Task: Benchmark Quantum Backends
     
     ## Objective
     Compare all available backends on standard test circuits.
     
     ## Instructions
     1. @proxima:foreach backend in available_backends
        - Execute bell_state circuit with 1000 shots
        - Record execution time and fidelity
     2. @proxima:compare_results
     3. @proxima:generate_report format=markdown
     ```

2. **Create Instruction AST (Abstract Syntax Tree):**
   - In `proxima/agent/ast.py`, define AST node types:
     - `InstructionDocument` - root node
       - `metadata` (dict) - from YAML frontmatter
       - `sections` (List[Section]) - instruction sections
     - `Section`:
       - `heading` (str) - section title
       - `level` (int) - heading level (1-6)
       - `content` (List[ContentNode]) - section content
     - `ContentNode` (abstract base):
       - `TextNode` - plain text
       - `CommandNode` - Proxima command directive
       - `ConditionNode` - conditional execution
       - `LoopNode` - loop directive
       - `CodeBlockNode` - code snippet
       - `VariableNode` - variable reference

3. **Implement Markdown Parser:**
   - In `proxima/agent/parser.py`, define `InstructionParser` class
   - Method: `parse(filepath) -> InstructionDocument`
   - Use python-markdown or mistune library for Markdown parsing
   - Steps:
     1. Read file content
     2. Extract YAML frontmatter (if present)
     3. Parse Markdown to tokens
     4. Build AST from tokens
     5. Identify Proxima directives (lines starting with @proxima:)
     6. Parse directive syntax
     7. Return structured InstructionDocument
   - Handle syntax errors gracefully with helpful error messages

4. **Implement Directive Parser:**
   - In `proxima/agent/directive_parser.py`, define `DirectiveParser` class
   - Parse Proxima-specific directives:
     - `@proxima:command <command_string>` - execute Proxima command
     - `@proxima:if <condition>` - conditional execution
     - `@proxima:foreach <var> in <collection>` - loop over items
     - `@proxima:set <variable> = <value>` - set variable
     - `@proxima:require <condition>` - assert requirement
     - `@proxima:llm <prompt>` - invoke LLM
   - Method: `parse_directive(line) -> DirectiveNode`
   - Return appropriate AST node based on directive type

5. **Implement Expression Parser:**
   - Parse expressions in directives:
     - Variables: `$backend`, `$results`, `$fidelity`
     - Operators: `==`, `!=`, `>`, `<`, `and`, `or`
     - Functions: `len($list)`, `max($values)`, `avg($numbers)`
   - In `proxima/agent/expression_parser.py`, define `ExpressionParser` class
   - Method: `parse(expression_string) -> ExpressionNode`
   - Build expression AST for later evaluation

6. **Implement Validation:**
   - Method: `validate_instruction_document(doc) -> ValidationResult`
   - Validation checks:
     - Required metadata present (agent: proxima)
     - No undefined variables referenced
     - Commands are valid Proxima commands
     - Loops have valid iterables
     - Conditions have valid syntax
   - ValidationResult with errors and warnings

### Step 12.2: Implement Instruction Interpreter

**Objective:** Execute parsed instructions, handling commands, conditions, loops, and variables.

**Detailed Instructions:**

1. **Create Instruction Interpreter:**
   - In `proxima/agent/interpreter.py`, define `InstructionInterpreter` class
   - Method: `interpret(instruction_doc, context) -> InterpretationResult`
   - InterpretationResult dataclass:
     - `success` (bool)
     - `output` (str) - execution log
     - `variables` (dict) - final variable state
     - `errors` (List[str])
   - Interpreter executes instructions sequentially

2. **Implement Execution Context:**
   - In `proxima/agent/context.py`, define `AgentContext` class
   - Context stores:
     - Variables: user-defined and system variables
     - Execution history: commands executed
     - Results: outputs from commands
     - State: current section, step
   - Method: `get_variable(name) -> Any`
   - Method: `set_variable(name, value)`
   - Method: `resolve_expression(expr) -> Any` - evaluate expression with current variables

3. **Implement Command Execution:**
   - Method: `execute_command(command_node, context) -> CommandResult`
   - Extract command string from node
   - Resolve variables in command string
   - Execute command via Proxima command dispatcher
   - Capture output and store in context
   - Handle command errors gracefully
   - CommandResult:
     - `success` (bool)
     - `output` (str)
     - `variables_set` (dict) - any variables the command set

4. **Implement Condition Evaluation:**
   - Method: `evaluate_condition(condition_node, context) -> bool`
   - Evaluate condition expression with current variable values
   - Support boolean operations: `and`, `or`, `not`
   - Support comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=`
   - Examples:
     - `$backend == "cirq-sv"` - check backend name
     - `$fidelity > 0.95` - check fidelity threshold
     - `$execution_time < 10 and $memory < 1000` - compound condition

5. **Implement Conditional Execution:**
   - Method: `execute_conditional(condition_node, context) -> None`
   - Structure:
     ```
     @proxima:if <condition>
       <instructions if true>
     @proxima:else
       <instructions if false>
     @proxima:endif
     ```
   - Evaluate condition
   - Execute appropriate branch
   - Skip else branch if condition true

6. **Implement Loop Execution:**
   - Method: `execute_loop(loop_node, context) -> None`
   - Loop types:
     - **Foreach**: `@proxima:foreach $item in $collection`
       - Iterate over collection
       - Set loop variable for each iteration
       - Execute loop body
     - **While**: `@proxima:while <condition>`
       - Evaluate condition
       - Execute body while condition true
       - Protect against infinite loops (max iterations)
   - Loop body executed multiple times with updated context

7. **Implement Variable Resolution:**
   - Method: `resolve_variables(text, context) -> str`
   - Replace variable references in text with their values
   - Syntax: `$variable_name` or `${variable_name}`
   - Examples:
     - `"Execute circuit on $backend"` → `"Execute circuit on cirq-sv"`
     - `"Shots: $shots"` → `"Shots: 1000"`
   - Handle undefined variables: error or use default value (configurable)

8. **Implement Built-in Functions:**
   - Provide utility functions for expressions:
     - `len($collection)` - length of list
     - `max($numbers)`, `min($numbers)` - max/min value
     - `avg($numbers)` - average
     - `sum($numbers)` - sum
     - `range(start, end)` - generate number range
     - `backends_available()` - get list of available backends
   - In context, register function handlers
   - Functions callable in expressions

9. **Implement LLM Directive:**
   - Directive: `@proxima:llm <prompt_template>`
   - Method: `execute_llm_directive(prompt_template, context) -> str`
   - Resolve variables in prompt template
   - Request LLM consent (purpose: "Agent instruction execution")
   - Call LLM via LLMHub
   - Store result in variable: `$llm_response`
   - Example:
     ```
     @proxima:llm "Analyze these results: $results. Provide recommendations."
     @proxima:set $recommendations = $llm_response
     ```

10. **Testing Interpreter:**
    - Test command execution executes commands correctly
    - Test conditional execution takes correct branch
    - Test loops iterate correct number of times
    - Test variable resolution substitutes values
    - Test built-in functions compute correctly
    - Test LLM directive invokes LLM and stores response

### Step 12.3: Implement Agent Automation Workflows

**Objective:** Enable complex automated workflows using instruction files.

**Detailed Instructions:**

1. **Create Workflow Templates:**
   - In `proxima/agent/templates/`, create common workflow templates:
     - `benchmark_backends.md` - compare all backends
     - `parameter_sweep.md` - sweep circuit parameter
     - `noise_study.md` - test various noise levels
     - `regression_test.md` - test for regressions
     - `nightly_validation.md` - comprehensive validation suite
   - Templates include placeholders for customization

2. **Implement Workflow Library:**
   - In `proxima/agent/workflow_library.py`, define `WorkflowLibrary` class
   - Method: `list_workflows() -> List[WorkflowInfo]`
   - Method: `load_workflow(name) -> InstructionDocument`
   - Method: `instantiate_workflow(name, parameters) -> InstructionDocument`
   - WorkflowInfo dataclass:
     - `name` (str)
     - `description` (str)
     - `parameters` (List[str]) - required parameters
     - `estimated_duration` (str)

3. **Implement Parameterized Workflows:**
   - Workflows can accept parameters
   - In YAML frontmatter:
     ```yaml
     parameters:
       - name: circuit_file
         type: path
         description: Circuit to benchmark
       - name: backends
         type: list
         default: ["all"]
     ```
   - Parameters resolved before execution
   - User provides parameter values via command or prompt

4. **Implement Workflow Scheduling:**
   - In `proxima/agent/scheduler.py`, define `WorkflowScheduler` class
   - Method: `schedule_workflow(workflow_name, schedule, parameters)`
   - Schedule formats:
     - Cron-style: `"0 2 * * *"` (daily at 2am)
     - Interval: `"every 6 hours"`
     - One-time: `"2026-01-08 10:00"`
   - Store scheduled workflows in database
   - Background task checks schedule and executes workflows
   - Config: `agent.enable_scheduler` (default false)

5. **Implement Workflow Monitoring:**
   - Track workflow execution:
     - Status (running, completed, failed)
     - Progress (current step)
     - Duration
     - Errors
   - Method: `get_workflow_status(execution_id) -> WorkflowStatus`
   - Real-time status updates in UI
   - Notifications on workflow completion/failure

6. **Implement Workflow Chaining:**
   - One workflow can trigger another
   - Directive: `@proxima:run_workflow <workflow_name> parameters=$params`
   - Workflows can pass data to subsequent workflows
   - Use case: benchmark → analyze → generate report (chain of 3 workflows)

7. **Implement Error Handling in Workflows:**
   - Directive: `@proxima:on_error <action>`
   - Actions:
     - `abort` - stop workflow execution
     - `continue` - log error and continue
     - `retry` - retry failed step N times
     - `fallback <alternative_command>` - execute alternative
   - Try-catch style error handling:
     ```
     @proxima:try
       @proxima:command execute circuit on qiskit-aer
     @proxima:catch
       @proxima:command execute circuit on cirq-sv
     @proxima:endtry
     ```

8. **Implement Workflow Versioning:**
   - Store workflow version in frontmatter
   - Track changes to workflows over time
   - Method: `get_workflow_versions(workflow_name) -> List[VersionInfo]`
   - Can execute specific version for reproducibility
   - Command: `agent run workflow benchmark@v1.2`

9. **Testing Workflows:**
   - Test each workflow template executes successfully
   - Test parameterized workflows with various parameter values
   - Test scheduled workflows execute at correct times
   - Test workflow chaining passes data correctly
   - Test error handling takes correct action on errors
   - Test workflow monitoring tracks status accurately

### Step 12.4: Implement LLM-Agent Collaboration

**Objective:** Enable deep integration between LLM and agent system for intelligent automation.

**Detailed Instructions:**

1. **Implement LLM Workflow Generation:**
   - Method: `generate_workflow_from_description(description) -> InstructionDocument`
   - User provides natural language description:
     - "Create a workflow that tests circuit X on all backends and generates a comparison report"
   - LLM generates proxima_agent.md file
   - Prompt template:
     ```
     Generate a Proxima agent instruction file for the following task: {{description}}
     
     Available Proxima directives:
     - @proxima:command <cmd>
     - @proxima:foreach $var in $collection
     - @proxima:if <condition>
     - @proxima:set $var = <value>
     - @proxima:llm <prompt>
     
     Available commands: {{command_list}}
     
     Return a complete agent instruction file in Markdown format with YAML frontmatter.
     ```
   - Parse LLM output
   - Validate generated workflow
   - Present to user for approval

2. **Implement Interactive Workflow Building:**
   - Method: `build_workflow_interactively() -> InstructionDocument`
   - Conversational workflow builder:
     1. LLM asks: "What do you want to accomplish?"
     2. User responds with goal
     3. LLM suggests steps
     4. User confirms or modifies
     5. LLM generates workflow
   - Iterative refinement until user satisfied
   - Multi-turn LLM conversation

3. **Implement Workflow Optimization:**
   - Method: `optimize_workflow(workflow) -> InstructionDocument`
   - LLM analyzes workflow and suggests improvements:
     - Parallelization opportunities
     - Redundant steps
     - More efficient commands
     - Better error handling
   - User reviews and applies optimizations
   - Prompt:
     ```
     Analyze this Proxima workflow and suggest optimizations:
     {{workflow_content}}
     
     Consider:
     - Steps that could run in parallel
     - Redundant operations
     - Error handling improvements
     - Resource efficiency
     ```

4. **Implement Intelligent Step Execution:**
   - When executing steps, LLM can provide guidance
   - Method: `execute_with_llm_guidance(step, context) -> ExecutionResult`
   - If step ambiguous or parameters unclear:
     - Ask LLM to interpret step
     - LLM provides concrete command
     - Execute command
   - If step fails:
     - Ask LLM for troubleshooting suggestions
     - Apply suggestions automatically (with consent)

5. **Implement Dynamic Workflow Adaptation:**
   - Method: `adapt_workflow(workflow, execution_state, issue) -> InstructionDocument`
   - When workflow encounters unexpected situation:
     - LLM analyzes issue
     - Generates modified workflow
     - Presents to user for approval
     - Resume with adapted workflow
   - Example: backend not available → LLM suggests alternative backend and updates workflow

6. **Implement Workflow Explanation:**
   - Method: `explain_workflow(workflow) -> str`
   - LLM generates natural language explanation of workflow:
     - What the workflow does
     - Why each step is necessary
     - Expected outcomes
     - Potential issues
   - Used for documentation and understanding complex workflows
   - Command: `agent explain workflow <name>`

7. **Implement Result Interpretation in Workflows:**
   - Workflows can use LLM for result interpretation
   - Directive: `@proxima:interpret_results $results`
   - LLM analyzes results and sets variables:
     - `$interpretation` - natural language interpretation
     - `$insights` - key findings
     - `$recommendations` - what to do next
   - Subsequent workflow steps can use these variables

8. **Testing LLM-Agent Collaboration:**
   - Test workflow generation from descriptions
   - Verify generated workflows are valid and executable
   - Test interactive building produces correct workflows
   - Test optimization suggestions improve workflows
   - Test dynamic adaptation handles unexpected situations
   - Test workflow explanations are accurate and helpful

### Step 12.5: Implement Agent UI and Integration

**Objective:** Provide user interface and seamless integration for agent instruction system.

**Detailed Instructions:**

1. **Create Agent Dashboard:**
   - TUI screen: `AgentDashboardScreen`
   - Display:
     - Available workflows (from library)
     - Recent workflow executions
     - Scheduled workflows
     - Workflow statistics
   - Actions:
     - Run workflow
     - Edit workflow
     - Schedule workflow
     - View execution history

2. **Implement Workflow Editor:**
   - TUI screen: `WorkflowEditorScreen`
   - Features:
     - Syntax highlighting for directives
     - Auto-completion for commands
     - Validation on-the-fly
     - Preview mode (show what workflow will do)
   - Save workflow to library
   - Test workflow (dry-run mode)

3. **Implement Execution Viewer:**
   - TUI screen: `WorkflowExecutionScreen`
   - Real-time display during workflow execution:
     - Current step
     - Progress bar
     - Variable values
     - Command outputs
     - Errors/warnings
   - Pause/resume controls
   - Step-through debugging mode

4. **Implement Command Interface:**
   - Command: `agent run <workflow_name>` - execute workflow
   - Command: `agent list` - list available workflows
   - Command: `agent create` - create new workflow (interactive)
   - Command: `agent edit <workflow_name>` - edit workflow
   - Command: `agent schedule <workflow> <schedule>` - schedule workflow
   - Command: `agent status <execution_id>` - check workflow status
   - Command: `agent generate <description>` - LLM generates workflow
   - Options:
     - `--param key=value` - provide parameter
     - `--dry-run` - simulate without executing
     - `--debug` - step-through execution

5. **Implement Workflow Marketplace:**
   - Community-contributed workflows
   - Method: `search_marketplace(query) -> List[WorkflowInfo]`
   - Method: `install_workflow(marketplace_id) -> bool`
   - Workflows rated by users
   - Security: review workflows before installation
   - Command: `agent marketplace search <query>`
   - Command: `agent marketplace install <id>`

6. **Implement Agent Configuration:**
   - In config: `agent` section
   - Config options:
     - `enable_agent` (bool, default true) - enable agent system
     - `workflow_library_path` (Path) - where workflows stored
     - `allow_llm_generation` (bool, default true) - allow LLM workflow generation
     - `require_approval` (bool, default true) - require user approval before execution
     - `max_loop_iterations` (int, default 1000) - prevent infinite loops
     - `execution_timeout` (int, default 3600) - seconds

7. **Implement Agent Permissions:**
   - Some workflows may perform sensitive operations
   - Permission levels:
     - `read_only` - can only read data
     - `execute` - can execute circuits
     - `modify` - can modify configuration
     - `admin` - full access
   - Workflows declare required permission in frontmatter
   - User approves permission grant
   - Config: `agent.default_permission` (default "execute")

8. **Implement Workflow Logging:**
   - Detailed logging of workflow execution:
     - Each step executed
     - Variables at each step
     - Outputs captured
     - Errors encountered
   - Logs stored: `logs/agent/workflow_{execution_id}.log`
   - Useful for debugging and auditing

9. **Implement Integration with Other Components:**
   - Agent can trigger executions via execution controller
   - Agent can create plans via planning engine
   - Agent can run comparisons via comparison system
   - Agent can interpret results via result interpreter
   - Seamless integration: workflows orchestrate entire Proxima

10. **Testing Agent UI and Integration:**
    - Test agent dashboard displays information correctly
    - Test workflow editor creates valid workflows
    - Test execution viewer shows real-time progress
    - Test all commands execute successfully
    - Test workflow marketplace search and install
    - Test permissions prevent unauthorized operations
    - Test logging captures all execution details
    - Test integration with execution/planning/comparison systems

### Phase 12 Implementation Complete

**Artifacts Produced:**
- Agent instruction format specification
- Instruction parser (Markdown + directives)
- Instruction interpreter with command/condition/loop execution
- Execution context with variables and functions
- Workflow library with templates
- Workflow scheduling system
- LLM-powered workflow generation and optimization
- Agent dashboard and workflow editor UI
- Command interface for agent operations
- Workflow marketplace for community workflows
- Agent permissions and security
- Integration with all Proxima components

**Verification:**
- Instruction parser correctly parses agent.md files
- Interpreter executes instructions accurately
- Commands execute successfully
- Conditions evaluate correctly
- Loops iterate correct number of times
- LLM generates valid workflows from descriptions
- Workflows execute end-to-end successfully
- Agent UI displays and controls workflows
- Scheduled workflows execute at correct times
- Integration with other components works seamlessly
- Tests pass with coverage >80%

---

## PHASE 13: SESSION MANAGEMENT & PERSISTENCE

### Step 13.1: Design Session Database Schema

**Objective:** Create a comprehensive database schema for persisting all session data, execution history, and user interactions.

**Detailed Instructions:**

1. **Select Database Technology:**
   - Use SQLite for local persistence (lightweight, no server required)
   - Library: `sqlite3` (Python standard library) or `sqlalchemy` for ORM
   - Database file location: `data/proxima.db`
   - Consider SQLAlchemy for:
     - Object-relational mapping
     - Database migrations (Alembic)
     - Query builder
     - Multiple database support (future: PostgreSQL for multi-user)

2. **Define Core Tables:**
   - In `proxima/persistence/schema.py`, define database schema using SQLAlchemy:
   - **sessions** table:
     - `id` (UUID, primary key)
     - `started_at` (datetime)
     - `ended_at` (datetime, nullable)
     - `duration_seconds` (float, nullable)
     - `status` (enum: ACTIVE, COMPLETED, CRASHED)
     - `user_id` (str, for multi-user support)
     - `proxima_version` (str)
     - `python_version` (str)
     - `platform` (str) - OS information
     - `metadata` (JSON) - additional session info
   - **executions** table:
     - `id` (UUID, primary key)
     - `session_id` (UUID, foreign key)
     - `execution_name` (str)
     - `circuit_file` (str) - path to circuit
     - `circuit_hash` (str) - hash for deduplication
     - `backend_id` (str)
     - `shots` (int)
     - `started_at` (datetime)
     - `completed_at` (datetime, nullable)
     - `duration_seconds` (float)
     - `status` (enum: PENDING, RUNNING, COMPLETED, FAILED, ABORTED)
     - `options` (JSON) - execution options
     - `error_message` (text, nullable)

3. **Define Result Tables:**
   - **results** table:
     - `id` (UUID, primary key)
     - `execution_id` (UUID, foreign key)
     - `result_type` (enum: MEASUREMENT, STATEVECTOR, DENSITY_MATRIX)
     - `measurements_json` (JSON) - measurement counts
     - `statevector_path` (str, nullable) - path to .npy file
     - `density_matrix_path` (str, nullable)
     - `metadata` (JSON)
     - `created_at` (datetime)
   - **interpretations** table:
     - `id` (UUID, primary key)
     - `result_id` (UUID, foreign key)
     - `interpretation_type` (enum: STATISTICAL, LLM, CUSTOM)
     - `summary` (text)
     - `insights` (JSON) - list of insights
     - `recommendations` (JSON) - list of recommendations
     - `created_at` (datetime)
     - `llm_provider` (str, nullable)
     - `llm_cost` (float, nullable)

4. **Define Plan Tables:**
   - **plans** table:
     - `id` (UUID, primary key)
     - `session_id` (UUID, foreign key)
     - `name` (str)
     - `description` (text)
     - `plan_json` (JSON) - full plan structure
     - `created_at` (datetime)
     - `started_at` (datetime, nullable)
     - `completed_at` (datetime, nullable)
     - `status` (enum: DRAFT, EXECUTING, PAUSED, COMPLETED, FAILED)
   - **stage_executions** table:
     - `id` (UUID, primary key)
     - `plan_id` (UUID, foreign key)
     - `stage_id` (str) - stage identifier within plan
     - `stage_type` (str)
     - `started_at` (datetime, nullable)
     - `completed_at` (datetime, nullable)
     - `status` (enum: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED)
     - `outputs` (JSON) - stage outputs
     - `error_message` (text, nullable)

5. **Define Comparison Tables:**
   - **comparisons** table:
     - `id` (UUID, primary key)
     - `session_id` (UUID, foreign key)
     - `circuit_hash` (str)
     - `backends` (JSON) - list of backend IDs
     - `created_at` (datetime)
     - `metrics` (JSON) - comparison metrics
     - `consensus_found` (bool)
     - `agreement_score` (float)
   - **comparison_executions** table:
     - `id` (UUID, primary key)
     - `comparison_id` (UUID, foreign key)
     - `execution_id` (UUID, foreign key)
     - Join table linking comparisons to executions

6. **Define Command History Table:**
   - **command_history** table:
     - `id` (UUID, primary key)
     - `session_id` (UUID, foreign key)
     - `command` (text) - full command string
     - `timestamp` (datetime)
     - `success` (bool)
     - `output` (text) - command output
     - `execution_time_ms` (float)
     - `context` (JSON) - session context at execution time

7. **Define Checkpoint Tables:**
   - **checkpoints** table:
     - `id` (UUID, primary key)
     - `execution_id` (UUID, foreign key, nullable)
     - `plan_id` (UUID, foreign key, nullable)
     - `checkpoint_type` (enum: MANUAL, AUTO, PRE_STAGE, POST_STAGE, EMERGENCY)
     - `created_at` (datetime)
     - `file_path` (str)
     - `size_bytes` (int)
     - `description` (text)
     - `can_resume` (bool)
     - `metadata` (JSON)

8. **Define LLM Usage Table:**
   - **llm_usage** table:
     - `id` (UUID, primary key)
     - `session_id` (UUID, foreign key)
     - `provider_id` (str)
     - `model` (str)
     - `purpose` (str) - what LLM was used for
     - `timestamp` (datetime)
     - `tokens_input` (int)
     - `tokens_output` (int)
     - `cost_usd` (float, nullable)
     - `latency_ms` (float)
     - `success` (bool)

9. **Define Indexes:**
   - Index on `sessions.started_at` for chronological queries
   - Index on `executions.session_id` for session lookups
   - Index on `executions.circuit_hash` for finding similar executions
   - Index on `command_history.session_id` and `timestamp`
   - Index on `results.execution_id`
   - Composite index on `executions(backend_id, status)` for filtering

10. **Implement Database Migrations:**
    - Use Alembic for schema migrations
    - Initialize: `alembic init proxima/persistence/migrations`
    - Create initial migration: captures schema version 1
    - Version control migrations for schema evolution
    - Method: `migrate_database(target_version)` - upgrade/downgrade schema

### Step 13.2: Implement Session Management

**Objective:** Create session lifecycle management with automatic tracking and persistence.

**Detailed Instructions:**

1. **Create Session Manager:**
   - In `proxima/persistence/session_manager.py`, define `SessionManager` singleton
   - Method: `start_session() -> Session`
   - Automatically called when Proxima starts
   - Creates session record in database
   - Captures session metadata: version, platform, start time
   - Returns Session object representing current session

2. **Implement Session Object:**
   - In `proxima/persistence/models.py`, define `Session` class (SQLAlchemy model)
   - Properties:
     - `id` (UUID) - session identifier
     - `started_at` (datetime)
     - `duration` (property, computed) - time since start
     - `executions` (relationship) - all executions in session
     - `commands` (relationship) - all commands in session
   - Methods:
     - `end_session()` - mark session as completed
     - `record_execution(execution)` - add execution to session
     - `record_command(command, output)` - add command to history
     - `get_statistics() -> SessionStats` - compute session statistics

3. **Implement Automatic Session Tracking:**
   - Hook into Proxima initialization:
     - When Proxima starts: create new session or resume previous
     - Track all user interactions automatically
     - Update session last_activity timestamp
   - Hook into shutdown:
     - When Proxima exits: finalize session
     - Set ended_at timestamp
     - Compute final statistics
     - Mark session as COMPLETED or CRASHED (if abnormal exit)

4. **Implement Session Context:**
   - In `proxima/persistence/session_context.py`, define `SessionContext` class
   - Stores session-wide state:
     - Active backend
     - User preferences
     - Recent commands
     - Working directory
     - Environment variables
   - Persisted to database in session metadata
   - Restored on session resume

5. **Implement Multi-Session Support:**
   - Track multiple concurrent sessions (different terminals, different users)
   - Session isolation: each session has independent state
   - Method: `get_active_sessions() -> List[Session]`
   - Method: `switch_session(session_id)` - for multi-terminal scenarios
   - Coordination: avoid conflicting executions

6. **Implement Session Resume:**
   - Method: `resume_session(session_id) -> Session`
   - Load session from database
   - Restore session context
   - Offer to resume any interrupted executions
   - Display: "Resumed session from 2 hours ago. 1 pending execution found."

7. **Implement Session Statistics:**
   - Method: `compute_session_stats(session) -> SessionStats`
   - SessionStats dataclass:
     - `total_executions` (int)
     - `successful_executions` (int)
     - `failed_executions` (int)
     - `total_duration` (timedelta)
     - `backends_used` (Dict[str, int]) - count per backend
     - `llm_usage_total_cost` (float)
     - `commands_executed` (int)
   - Displayed in session summary

8. **Testing Session Management:**
   - Test session creation on Proxima start
   - Test session persistence to database
   - Test session end on Proxima exit
   - Test session context save/restore
   - Test session statistics computation
   - Test multi-session isolation

### Step 13.3: Implement History Tracking and Querying

**Objective:** Provide comprehensive history tracking with powerful query capabilities.

**Detailed Instructions:**

1. **Create History Manager:**
   - In `proxima/persistence/history_manager.py`, define `HistoryManager` singleton
   - Central interface for querying history
   - Method: `record_execution(execution_data)` - save execution to history
   - Method: `record_command(command, result)` - save command to history
   - Method: `query_executions(**filters) -> List[Execution]` - flexible query interface

2. **Implement Execution History Queries:**
   - Method: `get_recent_executions(limit=10) -> List[Execution]`
   - Method: `get_executions_by_backend(backend_id) -> List[Execution]`
   - Method: `get_executions_by_circuit(circuit_hash) -> List[Execution]`
   - Method: `get_executions_by_date_range(start, end) -> List[Execution]`
   - Method: `get_failed_executions() -> List[Execution]`
   - Method: `search_executions(query: str) -> List[Execution]` - text search in execution names
   - All methods return Execution model objects from SQLAlchemy

3. **Implement Command History:**
   - Automatically record every command executed
   - Method: `get_command_history(session_id=None, limit=100) -> List[Command]`
   - Exclude sensitive commands (if any)
   - Store command success/failure
   - Enable command replay (re-execute previous command)
   - Command: `history commands` - display recent commands
   - Command: `!N` - re-execute command number N from history

4. **Implement Result History:**
   - Link results to executions in database
   - Method: `get_results_for_execution(execution_id) -> List[Result]`
   - Method: `get_results_by_type(result_type) -> List[Result]`
   - Results include measurements, statevectors, density matrices
   - Large data (arrays) stored separately, referenced by path

5. **Implement History Filtering:**
   - Complex filtering with SQLAlchemy queries:
     - Filter by backend: `backend_id == 'cirq-sv'`
     - Filter by status: `status == COMPLETED`
     - Filter by date: `started_at >= datetime(2026, 1, 1)`
     - Filter by duration: `duration_seconds < 10`
     - Combine filters: `(backend_id == 'qiskit-sv') AND (status == COMPLETED)`
   - Method: `filter_history(filters: dict) -> List[Execution]`
   - User-friendly filter syntax for commands

6. **Implement History Search:**
   - Full-text search across execution names, descriptions, metadata
   - Use SQLite FTS5 (Full-Text Search) extension
   - Method: `search_history(query: str) -> List[Execution]`
   - Search query syntax:
     - Simple: "bell state" - matches executions containing these words
     - Exact phrase: '"bell state"' - exact match
     - Boolean: "bell AND cirq" - both terms
   - Command: `search history "bell state"`

7. **Implement History Aggregation:**
   - Compute statistics across history
   - Method: `aggregate_history(metric: str, group_by: str) -> dict`
   - Examples:
     - Average execution time by backend
     - Success rate by backend
     - Most used backends
     - Busiest days/hours
   - Used for analytics and insights

8. **Implement History Export:**
   - Method: `export_history(filters: dict, format: str, filepath: str)`
   - Export formats: CSV, JSON, Excel
   - Exports executions matching filters
   - Include results and interpretations (optional)
   - Command: `export history --backend cirq-sv --format csv --output history.csv`

9. **Testing History Tracking:**
   - Test execution recording saves all fields correctly
   - Test command recording captures output
   - Test queries return correct filtered results
   - Test search finds relevant executions
   - Test aggregation computes correct statistics
   - Test export produces valid files

### Step 13.4: Implement Replay and Reproducibility

**Objective:** Enable exact reproduction of previous executions and workflows.

**Detailed Instructions:**

1. **Design Reproducibility Framework:**
   - Reproducibility requirements:
     - Same circuit
     - Same backend (including version)
     - Same execution options
     - Same random seed (if applicable)
   - Store all necessary information for replay in execution record

2. **Implement Execution Replay:**
   - Method: `replay_execution(execution_id) -> ReplayResult`
   - Steps:
     1. Load execution record from database
     2. Retrieve circuit (from file or database)
     3. Verify backend still available
     4. Check backend version compatibility
     5. Extract execution options
     6. Re-execute with same parameters
     7. Compare new result with original
     8. Return ReplayResult with comparison
   - ReplayResult dataclass:
     - `success` (bool)
     - `original_result` (BackendResult)
     - `replayed_result` (BackendResult)
     - `results_match` (bool)
     - `differences` (dict) - if results differ
     - `issues` (List[str]) - compatibility issues

3. **Implement Seed Management:**
   - Ensure reproducibility by controlling random seeds
   - Store seed in execution options
   - If no seed specified: generate and store seed
   - On replay: use same seed
   - Backend-specific seed handling:
     - LRET: set seed in execution options
     - Cirq: pass seed to simulator
     - Qiskit: set seed_simulator option

4. **Implement Version Tracking:**
   - Track versions of all components:
     - Proxima version
     - Backend library versions (cirq, qiskit, lret)
     - Python version
     - OS version
   - Store in execution metadata
   - On replay: check version compatibility
   - Warn if versions differ significantly

5. **Implement Circuit Persistence:**
   - Store circuits in database or file system
   - Options:
     - Embed small circuits in execution record (QASM string)
     - Store large circuits in files, reference by path
     - Hash circuits for deduplication
   - Method: `persist_circuit(circuit) -> str` - returns circuit_id or hash
   - Method: `retrieve_circuit(circuit_id) -> Circuit`

6. **Implement Workflow Replay:**
   - Method: `replay_plan(plan_id) -> ReplayResult`
   - Re-execute entire plan with same parameters
   - Compare stage outputs with original
   - Identify divergences
   - Use case: regression testing, validation

7. **Implement Comparison on Replay:**
   - Method: `compare_replay_results(original, replayed) -> ComparisonReport`
   - Metrics:
     - Measurement distribution similarity (fidelity, TVD)
     - Execution time comparison
     - Memory usage comparison
   - If results differ significantly: investigate why
     - Backend update changed behavior?
     - Non-deterministic execution?
     - Bug?

8. **Implement Batch Replay:**
   - Method: `replay_multiple(execution_ids) -> List[ReplayResult]`
   - Replay multiple executions
   - Useful for:
     - Regression testing after backend update
     - Validating reproducibility across many runs
   - Progress tracking for long batches
   - Command: `replay batch --date 2026-01-01 --backend cirq-sv`

9. **Implement Reproducibility Report:**
   - Method: `generate_reproducibility_report(replay_results) -> Report`
   - Report includes:
     - How many executions replayed successfully
     - How many results matched exactly
     - Fidelity distribution for approximate matches
     - Identified compatibility issues
     - Recommendations for ensuring reproducibility
   - Used for quality assurance

10. **Testing Replay Functionality:**
    - Test exact replay produces identical results (deterministic backends)
    - Test replay with different backend version warns user
    - Test replay with missing circuit fails gracefully
    - Test seed management ensures reproducibility
    - Test batch replay handles errors in individual replays
    - Test comparison identifies differences accurately

### Step 13.5: Implement Data Archiving and Cleanup

**Objective:** Manage long-term data storage, archiving, and cleanup to prevent database bloat.

**Detailed Instructions:**

1. **Design Archiving Strategy:**
   - Archive old data to reduce active database size
   - Archived data still queryable but slower access
   - Two-tier storage: active (SQLite) + archive (compressed files or separate DB)
   - Configurable archiving policies

2. **Implement Archiving Policies:**
   - In config: `persistence.archiving` section
   - Config options:
     - `archive_after_days` (default 90) - age threshold
     - `archive_inactive_sessions` (bool, default true)
     - `archive_failed_executions` (bool, default false) - keep for debugging
     - `compress_archives` (bool, default true)
   - Policy evaluation: which records to archive?

3. **Implement Archive Creation:**
   - Method: `archive_old_data(policy: ArchivePolicy) -> ArchiveResult`
   - Steps:
     1. Identify records meeting archive criteria (age, status)
     2. Extract records from active database
     3. Serialize to archive format (JSON or SQLite)
     4. Compress if configured (gzip or zstd)
     5. Store in archive directory: `data/archives/archive_YYYY-MM-DD.db.gz`
     6. Delete from active database
     7. Update archive index
   - ArchiveResult: records archived, space saved, archive file path

4. **Implement Archive Index:**
   - Maintain index of archived data
   - Table: `archive_index` in active database:
     - `archive_file` (str) - archive file path
     - `created_at` (datetime)
     - `records_count` (int)
     - `date_range_start` (datetime)
     - `date_range_end` (datetime)
     - `size_bytes` (int)
   - Used to locate archived data without opening archives

5. **Implement Archive Querying:**
   - Method: `query_archives(filters: dict) -> List[Execution]`
   - Check archive index to find relevant archives
   - Open and query archives matching criteria
   - Slower than active database queries (decompress, load)
   - Transparent to user: query both active and archives
   - Command: `history --include-archives` - search archives too

6. **Implement Archive Restoration:**
   - Method: `restore_from_archive(execution_ids) -> bool`
   - Move specific records from archive back to active database
   - Use cases:
     - User wants to replay archived execution
     - Analysis requires old data
   - Command: `restore execution <id>` - restore from archive

7. **Implement Data Cleanup:**
   - Method: `cleanup_database(policy: CleanupPolicy) -> CleanupResult`
   - CleanupPolicy:
     - Delete failed executions older than X days
     - Delete incomplete sessions
     - Delete orphaned results (execution deleted)
     - Delete old checkpoints
   - Cleanup without archiving (permanent deletion)
   - Config: `persistence.cleanup.enabled` (default false) - manual only
   - Command: `cleanup database --dry-run` - preview what would be deleted

8. **Implement Storage Monitoring:**
   - Method: `get_storage_stats() -> StorageStats`
   - StorageStats dataclass:
     - `database_size_mb` (float) - active database size
     - `archives_size_mb` (float) - total archive size
     - `total_executions` (int)
     - `archived_executions` (int)
     - `checkpoint_storage_mb` (float)
     - `result_files_storage_mb` (float)
   - Display: `storage stats` command
   - Warnings if storage approaching limits

9. **Implement Automatic Archiving:**
   - Background task: check archiving schedule
   - Config: `persistence.auto_archive_enabled` (default false)
   - Config: `persistence.auto_archive_schedule` (cron syntax, default "0 2 * * 0") - weekly
   - Run archiving according to schedule
   - Notify user of archiving activity
   - Log archives created

10. **Testing Archiving and Cleanup:**
    - Test archiving moves old records correctly
    - Test archived data is still queryable
    - Test archive restoration brings data back
    - Test cleanup deletes records permanently
    - Test storage monitoring computes correct sizes
    - Test automatic archiving runs on schedule
    - Test archive compression reduces size

### Step 13.6: Integration and Session UI

**Objective:** Integrate persistence throughout Proxima with comprehensive session management UI.

**Detailed Instructions:**

1. **Create Session Dashboard:**
   - TUI screen: `SessionDashboardScreen`
   - Display current session info:
     - Session ID, start time, duration
     - Total executions (successful, failed)
     - LLM usage and cost
     - Commands executed
   - Quick stats panel
   - Action buttons: end session, view history, export data

2. **Create History Browser:**
   - TUI screen: `HistoryBrowserScreen`
   - Table view of execution history:
     - Columns: timestamp, name, backend, duration, status
     - Sortable by any column
     - Filterable: status, backend, date range
   - Actions:
     - View details: select execution, press Enter
     - Replay: press R
     - Compare: select multiple, press C
     - Delete: press D (with confirmation)
   - Pagination for large histories

3. **Create Execution Detail View:**
   - TUI screen: show detailed execution information
   - Tabs:
     - Overview: metadata, status, timing
     - Configuration: backend, options, circuit info
     - Results: measurements, interpretations
     - Logs: execution logs
   - Actions from detail view: replay, export, delete

4. **Implement Command Interface:**
   - Command: `session info` - show current session stats
   - Command: `session list` - list all sessions
   - Command: `session resume <id>` - resume previous session
   - Command: `history` - show execution history
   - Command: `history commands` - show command history
   - Command: `replay <execution_id>` - replay execution
   - Command: `export session` - export session data
   - Command: `cleanup` - run cleanup with prompts
   - Command: `storage` - show storage statistics

5. **Implement Session Export:**
   - Method: `export_session(session_id, format, filepath)`
   - Export entire session: executions, results, commands, interpretations
   - Formats: JSON (complete), CSV (tabular), HTML (report)
   - Packaged export: ZIP file with all data + metadata
   - Command: `export session --format json --output session.json`

6. **Implement History Visualization:**
   - Method: `generate_history_visualizations() -> List[Figure]`
   - Visualizations:
     - Execution timeline (Gantt chart)
     - Backend usage pie chart
     - Success rate over time (line chart)
     - Execution duration distribution (histogram)
   - Display in TUI or export to files
   - Command: `visualize history`

7. **Implement Quick Access Features:**
   - Command: `last` - show last execution details
   - Command: `recent` - show recent 10 executions
   - Command: `favorite <execution_id>` - mark execution as favorite
   - Command: `favorites` - list favorite executions
   - Favorites stored with tag in database

8. **Implement Data Integrity Checks:**
   - Method: `verify_database_integrity() -> IntegrityReport`
   - Checks:
     - Foreign key consistency
     - Orphaned records
     - Corrupted result files
     - Missing checkpoints
   - Run periodically or on-demand
   - Command: `verify database`
   - Auto-fix option for common issues

9. **Implement Database Optimization:**
   - Method: `optimize_database() -> OptimizationResult`
   - SQLite optimizations:
     - VACUUM - reclaim space from deleted records
     - ANALYZE - update query planner statistics
     - Rebuild indexes
   - Run after large cleanups or periodically
   - Command: `optimize database`
   - May take time for large databases (show progress)

10. **Testing Integration:**
    - End-to-end: execute → record → query → replay
    - Test session dashboard displays correct stats
    - Test history browser filters and sorts correctly
    - Test all commands execute successfully
    - Test session export produces complete data
    - Test history visualizations render correctly
    - Test database integrity checks detect issues
    - Test database optimization improves performance

### Phase 13 Implementation Complete

**Artifacts Produced:**
- Comprehensive SQLite database schema with 12+ tables
- Session manager with automatic lifecycle tracking
- History manager with flexible querying and filtering
- Command history with replay capability
- Execution replay for reproducibility
- Circuit and result persistence
- Data archiving and cleanup system
- Session dashboard and history browser UI
- Command interface for session/history management
- Data export in multiple formats
- Storage monitoring and optimization

**Verification:**
- Database schema created successfully
- Sessions persist across Proxima restarts
- All executions, commands, results recorded correctly
- History queries return filtered results accurately
- Replay produces reproducible results
- Archiving moves old data without data loss
- Cleanup removes records correctly
- Session UI displays comprehensive information
- All commands execute successfully
- Database integrity maintained
- Tests pass with coverage >80%

---

## PHASE 14: ADVANCED FEATURES

### Step 14.1: Implement Model Context Protocol (MCP) Integration

**Objective:** Integrate Proxima with the Model Context Protocol for enhanced LLM interactions and tool usage.

**Detailed Instructions:**

1. **Understand MCP Architecture:**
   - MCP (Model Context Protocol): standardized protocol for LLM-tool integration
   - Components:
     - MCP Server: exposes tools/resources to LLMs
     - MCP Client: LLM or application consuming MCP services
     - MCP SDK: libraries for building servers/clients
   - Proxima roles:
     - MCP Server: expose Proxima capabilities as tools
     - MCP Client: consume external MCP servers

2. **Install MCP SDK:**
   - Add dependency: `mcp` (official MCP Python SDK)
   - Import: `from mcp import Server, ClientSession, Tool`
   - MCP protocol uses JSON-RPC 2.0 over stdio or HTTP

3. **Implement Proxima MCP Server:**
   - In `proxima/mcp/server.py`, define `ProximaMCPServer` class
   - Inherit from `mcp.Server`
   - Expose Proxima capabilities as MCP tools:
     - `execute_circuit` - run quantum circuit
     - `compare_backends` - multi-backend comparison
     - `interpret_results` - analyze results
     - `create_plan` - build execution plan
     - `replay_execution` - reproduce previous execution
   - Each tool has schema defining parameters and return type

4. **Define MCP Tool Schemas:**
   - For each exposed tool, define JSON schema:
   - Example: `execute_circuit` tool:
     ```
     {
       "name": "execute_circuit",
       "description": "Execute a quantum circuit on specified backend",
       "parameters": {
         "circuit": {"type": "string", "description": "Circuit in QASM format"},
         "backend": {"type": "string", "description": "Backend ID"},
         "shots": {"type": "integer", "default": 1000}
       },
       "returns": {
         "type": "object",
         "properties": {
           "measurements": {"type": "object"},
           "execution_time": {"type": "number"},
           "status": {"type": "string"}
         }
       }
     }
     ```
   - Schemas enable LLMs to understand tool capabilities

5. **Implement Tool Execution Handlers:**
   - Method: `handle_execute_circuit(params: dict) -> dict`
   - Parse parameters from MCP request
   - Validate parameters
   - Call Proxima functionality (execution controller)
   - Format response according to schema
   - Handle errors and return error responses
   - All tool handlers follow same pattern

6. **Implement MCP Server Initialization:**
   - Method: `start_mcp_server(transport: str = "stdio")`
   - Transport options:
     - stdio: communicate via standard input/output (default)
     - HTTP: expose HTTP endpoint
   - Register all tool handlers
   - Start server loop
   - Handle incoming MCP requests
   - Command: `proxima mcp server --transport stdio`

7. **Implement MCP Client:**
   - In `proxima/mcp/client.py`, define `ProximaMCPClient` class
   - Connect to external MCP servers
   - Method: `discover_tools(server_url) -> List[ToolInfo]`
   - Method: `call_tool(tool_name, parameters) -> Any`
   - Use cases:
     - Connect to specialized quantum tools (circuit optimizers, compilers)
     - Connect to data analysis tools
     - Chain Proxima with other services

8. **Implement MCP Tool Registry:**
   - Track available MCP tools (internal and external)
   - Method: `register_external_server(name, url)`
   - Method: `list_available_tools() -> List[ToolInfo]`
   - Display: `mcp tools` command
   - LLMs can query registry to find available tools

9. **Implement MCP-LLM Integration:**
   - LLMs can discover and use Proxima's MCP tools
   - Workflow:
     1. LLM asks: "Execute bell state circuit on Cirq"
     2. LLM discovers `execute_circuit` tool via MCP
     3. LLM constructs tool call with parameters
     4. Proxima MCP server receives request
     5. Proxima executes circuit
     6. Results returned to LLM
     7. LLM presents results to user
   - Enables LLM-driven workflows without hardcoded logic

10. **Testing MCP Integration:**
    - Test MCP server starts and accepts connections
    - Test each tool handler executes correctly
    - Test tool schemas are valid JSON Schema
    - Test MCP client can connect to servers
    - Test tool discovery finds all available tools
    - Test end-to-end: LLM → MCP → Proxima execution
    - Test error handling for invalid requests

### Step 14.2: Implement Language Server Protocol (LSP) Support

**Objective:** Provide LSP server for intelligent editing of Proxima instruction files and QASM circuits.

**Detailed Instructions:**

1. **Understand LSP Architecture:**
   - LSP (Language Server Protocol): standard for editor/IDE integration
   - LSP Server: provides language intelligence (completion, diagnostics, hover)
   - LSP Client: editor/IDE (VS Code, Neovim, etc.)
   - Proxima LSP Server: support for `.md` (instruction files) and `.qasm` (circuits)

2. **Install LSP SDK:**
   - Add dependency: `pygls` (Python Language Server library)
   - Import: `from pygls.server import LanguageServer`
   - pygls handles LSP protocol details

3. **Create Proxima Language Server:**
   - In `proxima/lsp/server.py`, define `ProximaLanguageServer` class
   - Inherit from `pygls.server.LanguageServer`
   - Support file types:
     - `.md` files containing Proxima instructions
     - `.qasm` files containing quantum circuits
   - Register LSP handlers for each feature

4. **Implement Diagnostics (Error Checking):**
   - Feature: real-time error checking as user types
   - For instruction files (`.md`):
     - Method: `validate_instruction_file(document) -> List[Diagnostic]`
     - Check for:
       - Invalid directive syntax
       - Undefined variables
       - Invalid commands
       - Malformed expressions
   - For QASM files:
     - Method: `validate_qasm(document) -> List[Diagnostic]`
     - Check for:
       - Syntax errors
       - Gate not defined
       - Qubit index out of range
       - Type errors
   - Return diagnostics with line numbers and messages
   - Editor displays errors inline

5. **Implement Autocompletion:**
   - Feature: suggest completions as user types
   - For instruction files:
     - Method: `get_completions(document, position) -> List[CompletionItem]`
     - Complete directives: `@proxima:` → suggest `command`, `foreach`, `if`, etc.
     - Complete commands: `execute` → suggest `execute circuit`, `execute plan`, etc.
     - Complete variables: `$` → suggest defined variables
     - Complete backend names: suggest available backends
   - For QASM files:
     - Complete gate names: `h` → suggest `h`, `hadamard`
     - Complete qubit names: `q[` → suggest valid indices
   - CompletionItem includes label, documentation, insert text

6. **Implement Hover Information:**
   - Feature: show documentation on hover
   - For instruction files:
     - Method: `get_hover_info(document, position) -> Hover`
     - Hover over directive: show directive documentation
     - Hover over command: show command syntax and options
     - Hover over variable: show variable value/type
   - For QASM files:
     - Hover over gate: show gate definition and matrix
     - Hover over qubit: show qubit properties
   - Return Hover with markdown-formatted documentation

7. **Implement Go to Definition:**
   - Feature: jump to variable/function definition
   - For instruction files:
     - Method: `get_definition(document, position) -> Location`
     - Click on variable: jump to where it's defined (`@proxima:set`)
     - Click on workflow call: jump to workflow file
   - Return Location with file path and line number

8. **Implement Formatting:**
   - Feature: format document (prettify)
   - For instruction files:
     - Method: `format_document(document) -> List[TextEdit]`
     - Consistent indentation
     - Align directives
     - Remove trailing whitespace
   - For QASM files:
     - Format gate definitions
     - Align qubit operations
   - Return TextEdits describing changes

9. **Implement Code Actions:**
   - Feature: quick fixes and refactorings
   - For instruction files:
     - Method: `get_code_actions(document, range, context) -> List[CodeAction]`
     - Quick fix: undefined variable → create variable definition
     - Refactor: extract repeated commands to loop
   - CodeActions include title and edits to apply

10. **Start LSP Server:**
    - Method: `start_lsp_server(transport: str = "stdio")`
    - Default transport: stdio (communicate via stdin/stdout)
    - Register all LSP handlers
    - Start server loop
    - Command: `proxima lsp server`
    - Configure editor to connect to Proxima LSP server

11. **Create Editor Extensions:**
    - VS Code extension: `proxima-vscode`
    - Extension activates on `.md` files with Proxima frontmatter
    - Extension starts Proxima LSP server automatically
    - Neovim config: configure LSP client for Proxima
    - Documentation: guide for setting up editor integration

12. **Testing LSP Support:**
    - Test diagnostics detect errors correctly
    - Test completions suggest relevant items
    - Test hover shows correct documentation
    - Test go-to-definition navigates correctly
    - Test formatting improves document structure
    - Test code actions provide useful fixes
    - Integration test with actual editor (VS Code)

### Step 14.3: Implement Custom Command System

**Objective:** Allow users to define and execute custom commands extending Proxima's functionality.

**Detailed Instructions:**

1. **Design Custom Command Framework:**
   - Custom commands: user-defined commands that execute arbitrary logic
   - Stored as Python scripts in `~/.proxima/commands/`
   - Commands registered and callable like built-in commands
   - Use cases:
     - Domain-specific workflows
     - Integration with external tools
     - Repetitive task automation

2. **Define Command Interface:**
   - In `proxima/commands/custom_command.py`, define `CustomCommand` base class
   - Required methods:
     - `get_name() -> str` - command name
     - `get_description() -> str` - help text
     - `get_parameters() -> List[Parameter]` - command parameters
     - `execute(args: dict, context: CommandContext) -> CommandResult`
   - Parameter dataclass:
     - `name` (str)
     - `type` (str) - "string", "int", "bool", "path", etc.
     - `required` (bool)
     - `default` (Any)
     - `description` (str)

3. **Implement Command Registry:**
   - In `proxima/commands/registry.py`, extend `CommandRegistry` to include custom commands
   - Method: `register_custom_command(command: CustomCommand)`
   - Method: `discover_custom_commands(directory: Path)`
   - Scan `~/.proxima/commands/` for Python files
   - Load and register each command
   - Validate commands before registration

4. **Implement Command Loading:**
   - Method: `load_custom_command(filepath: Path) -> CustomCommand`
   - Use dynamic import: `importlib.import_module()`
   - Instantiate command class
   - Handle import errors gracefully
   - Sandbox custom commands (restrict dangerous operations)

5. **Implement Command Execution:**
   - Custom commands execute with access to:
     - Proxima context (session, config)
     - Built-in commands (can call other commands)
     - Proxima APIs (execution controller, LLM hub, etc.)
   - CommandContext provides interfaces to Proxima internals
   - CommandResult:
     - `success` (bool)
     - `output` (str) - message to display
     - `data` (dict) - structured result data
     - `error` (Optional[str])

6. **Implement Command Template Generator:**
   - Method: `generate_command_template(name: str, description: str) -> str`
   - Generate boilerplate Python code for custom command
   - Template includes:
     - Class definition inheriting CustomCommand
     - Method stubs
     - Documentation comments
     - Example parameter definitions
   - Command: `create command <name>` - generates template file

7. **Implement Command Library:**
   - Community-contributed custom commands
   - Repository: `proxima-commands` on GitHub
   - Method: `search_command_library(query: str) -> List[CommandInfo]`
   - Method: `install_command(command_id: str) -> bool`
   - Download command from library
   - Verify signature (security)
   - Install to user's commands directory
   - Command: `commands search <query>`
   - Command: `commands install <id>`

8. **Implement Command Permissions:**
   - Custom commands declare required permissions:
     - `read` - read data
     - `execute` - execute circuits
     - `network` - make network requests
     - `filesystem` - write files
   - User approves permissions on first use
   - Permissions stored per command
   - Revoke permissions: `commands revoke <command_name>`

9. **Implement Command Documentation:**
   - Each custom command includes docstring
   - Method: `get_help() -> str` - detailed help text
   - Extracted docstring shown in help system
   - Command: `help <custom_command_name>`
   - Markdown formatting supported

10. **Testing Custom Commands:**
    - Test command discovery finds commands in directory
    - Test command loading imports correctly
    - Test command execution with various parameters
    - Test command template generation produces valid code
    - Test command library search and install
    - Test permissions system restricts operations
    - Test custom command can call built-in commands

### Step 14.4: Implement Plugin System and Extensibility

**Objective:** Create comprehensive plugin architecture for extending Proxima with new backends, features, and integrations.

**Detailed Instructions:**

1. **Design Plugin Architecture:**
   - Plugins: self-contained extensions that add functionality
   - Plugin types:
     - Backend plugins: add new quantum backends
     - Tool plugins: add new tools/utilities
     - UI plugins: add new TUI screens
     - Integration plugins: connect to external services
   - Plugin discovery: scan `~/.proxima/plugins/` and `site-packages/proxima_plugins`

2. **Define Plugin Interface:**
   - In `proxima/plugins/plugin.py`, define `Plugin` base class
   - Required methods:
     - `get_name() -> str` - plugin name
     - `get_version() -> str` - plugin version
     - `get_description() -> str`
     - `get_dependencies() -> List[str]` - required packages
     - `initialize(context: ProximaContext)` - setup plugin
     - `shutdown()` - cleanup
   - Optional methods:
     - `get_commands() -> List[Command]` - add custom commands
     - `get_backends() -> List[BackendAdapter]` - add backends
     - `get_ui_screens() -> List[Screen]` - add UI screens

3. **Implement Plugin Manager:**
   - In `proxima/plugins/manager.py`, define `PluginManager` singleton
   - Method: `discover_plugins() -> List[PluginInfo]`
   - Method: `load_plugin(plugin_name: str) -> Plugin`
   - Method: `enable_plugin(plugin_name: str) -> bool`
   - Method: `disable_plugin(plugin_name: str) -> bool`
   - Method: `list_plugins() -> List[PluginInfo]`
   - PluginInfo dataclass: name, version, status, description

4. **Implement Plugin Discovery:**
   - Scan plugin directories for Python packages
   - Look for `plugin.py` or `__init__.py` with Plugin subclass
   - Read plugin metadata (from pyproject.toml or plugin.json)
   - Build plugin registry
   - Run on Proxima startup

5. **Implement Plugin Loading:**
   - Method: `load_plugin_from_path(path: Path) -> Plugin`
   - Dynamic import plugin module
   - Instantiate plugin class
   - Validate plugin interface
   - Check dependencies (pip packages)
   - Install missing dependencies (with user consent)
   - Call plugin.initialize()
   - Register plugin components (commands, backends, etc.)

6. **Implement Backend Plugin Support:**
   - Plugins can register new backend adapters
   - Plugin provides backend class implementing `QuantumBackend` interface
   - Example: IBM Quantum plugin adds IBM Cloud backends
   - Backend automatically added to registry
   - Available for circuit execution immediately

7. **Implement Hook System:**
   - Plugins can register hooks for events:
     - `on_execution_start`
     - `on_execution_complete`
     - `on_result_analyzed`
     - `on_session_start`
   - Plugin method called when event occurs
   - Allows plugins to extend behavior without modifying core
   - In `proxima/plugins/hooks.py`, define hook registry
   - Method: `register_hook(event: str, handler: Callable)`
   - Method: `trigger_hook(event: str, data: dict)`

8. **Implement Plugin Configuration:**
   - Each plugin can have configuration section in Proxima config
   - Config: `plugins.<plugin_name>.*`
   - Plugin reads config during initialization
   - UI for managing plugin settings: `configure plugin <name>`

9. **Implement Plugin Marketplace:**
   - Similar to command library, but for plugins
   - Browse available plugins: `plugins search <query>`
   - Install plugin: `plugins install <id>`
   - Plugin packages distributed via PyPI or custom registry
   - Verified plugins: signed and reviewed

10. **Implement Plugin Sandboxing:**
    - Security: restrict plugin access to system
    - Plugin permissions:
      - File system access
      - Network access
      - Proxima API access level
    - User approves permissions on plugin installation
    - Runtime enforcement of permissions

11. **Implement Plugin Development Kit (PDK):**
    - In `proxima/plugins/pdk.py`, provide utilities for plugin developers
    - Functions:
      - `register_backend(backend_class)` - convenient registration
      - `register_command(command_class)`
      - `add_ui_screen(screen_class)`
      - `log(message)` - plugin logging
    - Documentation: plugin development guide
    - Example plugins in `examples/plugins/`

12. **Testing Plugin System:**
    - Test plugin discovery finds plugins
    - Test plugin loading initializes correctly
    - Test backend plugin adds functional backend
    - Test command plugin adds working command
    - Test hooks trigger on events
    - Test plugin configuration loads correctly
    - Test plugin sandbox enforces permissions
    - Create example plugin and test end-to-end

### Step 14.5: Implement Skills System

**Objective:** Enable Proxima to learn and improve from user interactions using a skills-based architecture.

**Detailed Instructions:**

1. **Design Skills Architecture:**
   - Skill: learned capability that improves over time
   - Skills based on:
     - Historical executions (which backends work best)
     - User preferences (which features used most)
     - Outcomes (which decisions led to success)
   - Skills stored as models or rules
   - Skills applied to assist users

2. **Define Core Skills:**
   - **Backend Selection Skill**: learn optimal backend for circuit types
   - **Parameter Tuning Skill**: learn good parameter values
   - **Error Recovery Skill**: learn how to fix common errors
   - **Workflow Optimization Skill**: learn efficient workflow patterns
   - **Resource Management Skill**: learn resource allocation strategies

3. **Implement Skill Interface:**
   - In `proxima/skills/skill.py`, define `Skill` base class
   - Methods:
     - `get_name() -> str`
     - `get_description() -> str`
     - `learn(data: dict)` - update skill from new data
     - `predict(context: dict) -> Prediction` - apply skill
     - `save(path: Path)` - persist skill model
     - `load(path: Path)` - restore skill model

4. **Implement Backend Selection Skill:**
   - In `proxima/skills/backend_selection_skill.py`, define `BackendSelectionSkill`
   - Learn from history:
     - Track which backends used for which circuits
     - Track execution success rates
     - Track user overrides (when user changes automatic selection)
   - Build predictive model:
     - Features: circuit characteristics (qubits, depth, gates)
     - Target: optimal backend
     - Model: decision tree or logistic regression
   - Method: `predict_backend(circuit) -> str`
   - Improves over time with more data

5. **Implement Parameter Tuning Skill:**
   - In `proxima/skills/parameter_tuning_skill.py`, define `ParameterTuningSkill`
   - Learn optimal parameters:
     - Shots count for desired accuracy
     - Noise levels for realistic simulation
     - Optimization levels for circuit compilation
   - Method: `suggest_shots(circuit, desired_accuracy) -> int`
   - Based on historical data: accuracy vs shots tradeoff

6. **Implement Error Recovery Skill:**
   - In `proxima/skills/error_recovery_skill.py`, define `ErrorRecoverySkill`
   - Learn from error patterns:
     - Track errors and how they were resolved
     - Build error → solution mapping
   - Method: `suggest_fix(error) -> List[Action]`
   - Actions: change backend, adjust parameters, modify circuit
   - Use LLM for complex error reasoning (with skill-learned context)

7. **Implement Skill Training:**
   - Method: `train_skill(skill: Skill, training_data: List[dict])`
   - Extract training data from session history
   - Feature engineering: convert raw data to model features
   - Train model (scikit-learn, PyTorch, or simple rules)
   - Validate model performance
   - Save trained model to disk
   - Command: `train skill <skill_name>` - manual training

8. **Implement Automatic Skill Learning:**
   - Background task: continuously learn from user interactions
   - After each execution: update relevant skills
   - Incremental learning: update model without full retrain
   - Config: `skills.auto_learn` (default true)
   - Config: `skills.learning_rate` (how aggressively to update)

9. **Implement Skill Application:**
   - Skills provide suggestions to user
   - Integration points:
     - Backend selection: skill suggests backend
     - Execution options: skill suggests shots
     - Error occurs: skill suggests fixes
   - Display skill suggestions with confidence:
     - "Skill suggests: use cirq-sv (confidence: 85%)"
   - User can accept or reject suggestion
   - Rejections used as negative feedback for learning

10. **Implement Skill Explainability:**
    - Method: `explain_prediction(skill, prediction) -> str`
    - Generate explanation for skill's suggestion
    - Use SHAP or LIME for model interpretability
    - Or use LLM to generate natural language explanation
    - Show: "Suggested cirq-sv because: similar circuits ran faster, sufficient memory available"

11. **Implement Skill Management:**
    - Command: `skills list` - show all skills and status
    - Command: `skills enable <name>` - activate skill
    - Command: `skills disable <name>` - deactivate skill
    - Command: `skills reset <name>` - reset skill (clear learned data)
    - Command: `skills stats <name>` - show skill performance statistics

12. **Testing Skills System:**
    - Test skill learning from training data
    - Test skill prediction produces reasonable suggestions
    - Test automatic learning updates skills
    - Test skill explanation is understandable
    - Test skills improve over time (with simulated history)
    - Test skill management commands work correctly

### Step 14.6: Integration and Advanced Features UI

**Objective:** Provide UI and final integration for all advanced features.

**Detailed Instructions:**

1. **Create Advanced Features Dashboard:**
   - TUI screen: `AdvancedFeaturesScreen`
   - Sections:
     - MCP Integration status
     - LSP server status
     - Custom commands (count, list)
     - Plugins (enabled, available)
     - Skills (active, performance)
   - Quick actions: enable/disable features

2. **Create Plugin Manager UI:**
   - TUI screen: `PluginManagerScreen`
   - List installed plugins with status
   - Actions: enable, disable, configure, remove
   - Browse plugin marketplace
   - Install new plugins with progress bar

3. **Create Skills Dashboard:**
   - TUI screen: `SkillsDashboardScreen`
   - Show each skill:
     - Name and description
     - Status (active, learning)
     - Performance metrics (accuracy, usage count)
     - Last trained timestamp
   - Actions: train, reset, configure

4. **Implement Feature Toggles:**
   - Config section: `advanced_features`
   - Toggles:
     - `mcp.enabled` (default false) - enable MCP server
     - `lsp.enabled` (default false) - enable LSP server
     - `custom_commands.enabled` (default true)
     - `plugins.enabled` (default true)
     - `skills.enabled` (default true)
   - Allows users to disable features they don't need

5. **Implement Command Interface:**
   - Command: `mcp server start` - start MCP server
   - Command: `lsp server start` - start LSP server
   - Command: `commands list` - list custom commands
   - Command: `plugins list` - list plugins
   - Command: `skills list` - list skills
   - Command: `features` - show advanced features status

6. **Implement Startup Configuration:**
   - On Proxima start:
     - Check feature toggles
     - Load enabled plugins
     - Initialize enabled skills
     - Optionally start MCP/LSP servers (if configured)
   - Fast startup: only load what's needed

7. **Implement Documentation Integration:**
   - Help system includes documentation for:
     - Using MCP server with LLMs
     - Setting up LSP in editors
     - Creating custom commands
     - Developing plugins
     - Understanding skills
   - Command: `help mcp`, `help plugins`, etc.

8. **Implement Diagnostics:**
   - Method: `run_advanced_features_diagnostics() -> DiagnosticReport`
   - Check:
     - MCP server responding
     - LSP server responding
     - Plugins loaded successfully
     - Skills functioning
     - Custom commands valid
   - DiagnosticReport shows issues and suggestions
   - Command: `diagnose advanced-features`

9. **Testing Integration:**
   - Test advanced features dashboard displays status
   - Test plugin manager can install/remove plugins
   - Test skills dashboard shows correct metrics
   - Test feature toggles enable/disable correctly
   - Test all commands execute successfully
   - Test diagnostics detect issues
   - End-to-end: enable all features, verify working

### Phase 14 Implementation Complete

**Artifacts Produced:**
- MCP server exposing Proxima capabilities as tools
- MCP client for consuming external MCP servers
- LSP server for intelligent editing (completions, diagnostics, hover)
- Custom command system with template generator
- Comprehensive plugin architecture with hooks
- Plugin marketplace for community extensions
- Skills system with learning from user interactions
- Backend selection, parameter tuning, error recovery skills
- Advanced features dashboard and management UI
- Complete integration with all Proxima components

**Verification:**
- MCP server accepts connections and executes tools
- LLMs can discover and use Proxima via MCP
- LSP server provides completions and diagnostics
- Editors integrate with LSP (VS Code tested)
- Custom commands load and execute correctly
- Plugins extend Proxima functionality
- Skills learn from history and make predictions
- Skill suggestions improve over time
- All advanced features toggleable via config
- Diagnostics detect and report issues
- Tests pass with coverage >80%

---

## PHASE 15: UI POLISH & DOCUMENTATION

### Step 15.1: Implement Theming and Visual Polish

**Objective:** Create professional, customizable themes and polish all UI elements for excellent user experience.

**Detailed Instructions:**

1. **Design Theme System Architecture:**
   - Themes define colors, styles, and visual elements
   - Support multiple built-in themes
   - User-customizable themes
   - Theme storage: `~/.proxima/themes/` (JSON or YAML)
   - Live theme switching without restart

2. **Create Theme Schema:**
   - In `proxima/ui/theme.py`, define `Theme` dataclass
   - Theme properties:
     - `name` (str) - theme name
     - `colors` (dict) - color palette
       - `primary` - main accent color
       - `secondary` - secondary accent
       - `background` - main background
       - `surface` - elevated surfaces (panels)
       - `text` - primary text
       - `text_dim` - secondary text
       - `success` - success indicators (green)
       - `warning` - warnings (yellow)
       - `error` - errors (red)
       - `info` - information (blue)
     - `styles` (dict) - text styles
       - `heading` - heading style (bold, underline)
       - `emphasis` - emphasized text (italic, bold)
       - `code` - inline code style
       - `link` - hyperlink style
     - `borders` (dict) - border styles
       - `style` - "solid", "double", "rounded", "heavy"
       - `color` - border color
     - `icons` (dict) - icon characters (Unicode symbols)

3. **Implement Built-in Themes:**
   - **Dark Theme** (default):
     - Dark background (#1e1e1e)
     - Light text (#d4d4d4)
     - Blue accent (#007acc)
     - Good contrast, easy on eyes
   - **Light Theme**:
     - Light background (#ffffff)
     - Dark text (#333333)
     - Blue accent (#0066cc)
     - Clean, professional
   - **Nord Theme**:
     - Nord color palette (popular developer theme)
     - Muted, pastel colors
   - **Solarized Dark/Light**:
     - Solarized color scheme
   - **High Contrast**:
     - Maximum contrast for accessibility
     - Black background, white text
   - Define themes in `proxima/ui/themes/` as JSON files

4. **Implement Theme Manager:**
   - In `proxima/ui/theme_manager.py`, define `ThemeManager` singleton
   - Method: `load_theme(name: str) -> Theme`
   - Method: `get_available_themes() -> List[str]`
   - Method: `apply_theme(theme: Theme)`
   - Method: `save_custom_theme(theme: Theme)`
   - Method: `export_theme(theme: Theme, filepath: Path)`
   - Apply theme to all Textual widgets

5. **Integrate Theme with Textual:**
   - Textual uses CSS-like styling
   - Method: `generate_textual_css(theme: Theme) -> str`
   - Convert theme to Textual CSS
   - Apply CSS to app: `app.css = generated_css`
   - Update dynamically when theme changes
   - All widgets use theme colors automatically

6. **Implement Theme Switcher UI:**
   - TUI component: `ThemeSelector` dropdown
   - Display in settings screen
   - Preview theme before applying
   - Command: `theme <name>` - switch theme
   - Command: `theme list` - list available themes
   - Show live preview of theme changes

7. **Polish All UI Screens:**
   - Review every TUI screen for consistency:
     - Consistent header styling
     - Consistent spacing and padding
     - Proper use of borders and dividers
     - Aligned elements
     - Appropriate use of colors (semantic)
   - Polish main dashboard:
     - Clean layout
     - Quick stats prominently displayed
     - Action buttons clearly visible
   - Polish execution screen:
     - Real-time progress indicators
     - Color-coded status (running=blue, success=green, failed=red)
     - Smooth animations

8. **Implement Status Indicators:**
   - Visual feedback for all operations:
     - Spinners for loading
     - Progress bars for long tasks
     - Status icons (✓ success, ✗ failure, ⚠ warning)
   - Use Textual's LoadingIndicator widget
   - Custom progress display for executions
   - In `proxima/ui/components/status.py`, create reusable status components

9. **Implement Animations and Transitions:**
   - Smooth transitions between screens
   - Fade-in effects for new content
   - Slide animations for panels
   - Use Textual's animation support
   - Subtle, not distracting
   - Config: `ui.animations.enabled` (can disable for performance)

10. **Implement Responsive Layout:**
    - UI adapts to terminal size
    - Small terminals: vertical stacking, hide less critical info
    - Large terminals: multi-column layouts, show more details
    - Test at various terminal sizes (80x24, 120x40, 200x60)
    - Method: `adapt_layout(width: int, height: int)`

11. **Implement Icons and Symbols:**
    - Use Unicode symbols for visual clarity:
      - ▶ Start
      - ⏸ Pause
      - ■ Stop
      - ✓ Success
      - ✗ Error
      - ⚠ Warning
      - ℹ Info
      - ⚙ Settings
      - 📊 Results
      - 🔍 Search
    - Fallback for terminals without Unicode support
    - Config: `ui.unicode.enabled` (can disable)

12. **Testing UI Polish:**
    - Test all themes apply correctly
    - Test theme switching works live
    - Test UI looks good at various terminal sizes
    - Test status indicators display correctly
    - Test animations are smooth
    - Test icons render on different terminals
    - User testing: gather feedback on aesthetics

### Step 15.2: Implement Comprehensive Help System

**Objective:** Create rich, context-aware help system with tutorials and interactive guides.

**Detailed Instructions:**

1. **Design Help System Architecture:**
   - Multi-level help:
     - Quick help (tooltips, inline)
     - Command help (syntax, examples)
     - Topic help (concepts, guides)
     - Interactive tutorials
   - Context-aware: help relevant to current screen
   - Searchable help content

2. **Create Help Content Structure:**
   - Help content stored in `proxima/help/content/` as Markdown files
   - Structure:
     - `commands/` - one file per command
     - `topics/` - conceptual topics (backends, circuits, noise)
     - `tutorials/` - step-by-step guides
     - `faq.md` - frequently asked questions
     - `glossary.md` - term definitions
   - Content format: Markdown with YAML frontmatter
     - Frontmatter: title, description, tags, related topics

3. **Implement Command Help:**
   - Every command has help documentation
   - Help includes:
     - Command syntax
     - Description
     - Parameters (with types, defaults)
     - Examples (multiple use cases)
     - Related commands
   - Command: `help <command>` - show command help
   - Example: `help execute circuit`
   - Display in formatted TUI panel

4. **Implement Topic Help:**
   - Conceptual documentation topics
   - Topics:
     - "Quantum Backends" - overview of backends
     - "Circuit Formats" - QASM, JSON, etc.
     - "Execution Options" - what options mean
     - "Noise Models" - understanding noise
     - "Result Interpretation" - understanding outputs
     - "Planning and Workflows" - multi-stage execution
   - Command: `help topic <topic_name>`
   - Rendered Markdown with syntax highlighting

5. **Implement Interactive Tutorials:**
   - Guided, step-by-step tutorials
   - Tutorial engine:
     - Load tutorial file
     - Display step instructions
     - Wait for user to complete action
     - Verify action completed
     - Proceed to next step
   - In `proxima/help/tutorial_engine.py`, define `TutorialEngine`
   - Method: `start_tutorial(name: str)`
   - Tutorials:
     - "First Execution" - run first circuit
     - "Using Multiple Backends" - comparison
     - "Creating a Plan" - staged execution
     - "Working with Noise" - noise models
     - "Agent Instructions" - automation
   - Command: `tutorial <name>` - start tutorial

6. **Implement Tutorial Format:**
   - Tutorial file format: Markdown with special directives
   - Directives:
     - `[STEP: title]` - mark tutorial step
     - `[ACTION: command]` - action user should perform
     - `[VERIFY: condition]` - verification check
     - `[TIP: text]` - helpful tip
   - Tutorial engine parses and executes
   - Example tutorial:
     ```markdown
     # Tutorial: First Execution
     
     [STEP: Load a circuit]
     First, load a quantum circuit. You can use a sample circuit.
     [ACTION: load sample bell]
     [VERIFY: circuit_loaded]
     
     [STEP: Execute the circuit]
     Now execute the circuit on a backend.
     [ACTION: execute circuit --backend cirq-sv --shots 1000]
     [VERIFY: execution_completed]
     
     [TIP: You can view results with the 'results' command]
     ```

7. **Implement Help Search:**
   - Method: `search_help(query: str) -> List[HelpResult]`
   - Search across all help content
   - Full-text search with ranking
   - Return relevant commands, topics, tutorials
   - Command: `help search <query>`
   - Display search results with snippets

8. **Implement Context-Sensitive Help:**
   - Press `?` or `F1` on any screen: show relevant help
   - Help content depends on current screen:
     - On execution screen: help about execution
     - On results screen: help about interpreting results
     - On backend selection: help about backends
   - Method: `get_context_help(screen_name: str) -> HelpContent`

9. **Implement Help Browser UI:**
   - TUI screen: `HelpBrowserScreen`
   - Left panel: table of contents (commands, topics, tutorials)
   - Right panel: help content display
   - Markdown rendering with Rich
   - Navigation: arrow keys, search
   - Bookmarks: save frequently accessed help pages

10. **Implement Quick Reference Card:**
    - Command: `help quickref` - show quick reference
    - Display common commands and shortcuts
    - Keyboard shortcuts table
    - One-page cheat sheet
    - Can print to file: `help quickref --output cheatsheet.txt`

11. **Implement Help Tooltips:**
    - Hover over UI elements: show tooltip
    - Tooltips explain what element does
    - Brief (1-2 sentences)
    - Use Textual's Tooltip support
    - Example: hover over backend name → "Cirq state vector simulator"

12. **Testing Help System:**
    - Test all help content renders correctly
    - Test command help shows accurate syntax
    - Test tutorials complete successfully
    - Test help search finds relevant content
    - Test context-sensitive help on each screen
    - Test help browser navigation
    - Verify all help content accurate and complete

### Step 15.3: Write Comprehensive Documentation

**Objective:** Create complete documentation for users and developers covering all aspects of Proxima.

**Detailed Instructions:**

1. **Design Documentation Structure:**
   - Documentation types:
     - User Guide - for end users
     - Developer Guide - for contributors
     - API Reference - for plugin/extension developers
     - Deployment Guide - for installation and setup
   - Format: Markdown + MkDocs (static site generator)
   - Host documentation on GitHub Pages or Read the Docs

2. **Set Up Documentation Framework:**
   - Install MkDocs: `pip install mkdocs mkdocs-material`
   - Initialize: `mkdocs new docs/`
   - Configure: `docs/mkdocs.yml`
   - Theme: Material for MkDocs (modern, searchable)
   - Structure:
     ```
     docs/
       index.md - landing page
       user-guide/
         getting-started.md
         installation.md
         basic-usage.md
         backends.md
         advanced-features.md
       developer-guide/
         architecture.md
         contributing.md
         plugin-development.md
       api-reference/
         cli-api.md
         python-api.md
       deployment/
         docker.md
         cloud.md
     ```

3. **Write User Guide:**
   - **Getting Started** (`user-guide/getting-started.md`):
     - What is Proxima?
     - Key features
     - Quick start tutorial (5 minutes)
     - First circuit execution
   - **Installation** (`user-guide/installation.md`):
     - System requirements
     - Installation methods (pip, conda, Docker)
     - Backend installation (LRET, Cirq, Qiskit)
     - Configuration setup
   - **Basic Usage** (`user-guide/basic-usage.md`):
     - Launching Proxima
     - UI overview
     - Loading circuits
     - Executing circuits
     - Viewing results
     - Common workflows
   - **Backends** (`user-guide/backends.md`):
     - Available backends
     - Backend comparison
     - When to use which backend
     - Backend-specific options
     - Adding custom backends
   - **Advanced Features** (`user-guide/advanced-features.md`):
     - Multi-backend comparison
     - Planning and staged execution
     - LLM integration
     - Checkpoint and resume
     - Agent instructions
     - MCP and LSP integration
     - Custom commands and plugins

4. **Write Developer Guide:**
   - **Architecture** (`developer-guide/architecture.md`):
     - System overview (5-tier architecture diagram)
     - Component descriptions
     - Data flow diagrams
     - Technology stack
   - **Contributing** (`developer-guide/contributing.md`):
     - How to contribute
     - Code style guide (PEP 8, type hints)
     - Testing requirements
     - Pull request process
     - Development setup
   - **Plugin Development** (`developer-guide/plugin-development.md`):
     - Plugin architecture
     - Creating a plugin (step-by-step)
     - Plugin API reference
     - Example plugins
     - Publishing plugins
   - **Backend Adapter Development** (`developer-guide/backend-adapter.md`):
     - QuantumBackend interface
     - Implementing a new backend
     - Testing backend adapters
     - Backend registration

5. **Write API Reference:**
   - **CLI API** (`api-reference/cli-api.md`):
     - All commands with syntax
     - Command options
     - Examples
     - Exit codes
   - **Python API** (`api-reference/python-api.md`):
     - Using Proxima programmatically
     - Core classes and methods
     - Code examples
     - API stability guarantees
   - Generate API docs from docstrings:
     - Use Sphinx or pdoc3
     - Auto-generate from source code docstrings
     - Ensure all public APIs documented

6. **Write Deployment Guide:**
   - **Docker Deployment** (`deployment/docker.md`):
     - Docker image usage
     - docker-compose examples
     - Volume mounting for persistence
     - Environment variables
   - **Cloud Deployment** (`deployment/cloud.md`):
     - Running on AWS, Azure, GCP
     - Using cloud quantum backends
     - Scaling considerations
     - Cost optimization

7. **Create Example Gallery:**
   - In `docs/examples/`, create example workflows
   - Examples:
     - "Bell State Execution" - simple example
     - "Noise Study" - compare noise models
     - "Parameter Sweep" - vary parameters
     - "Multi-Backend Benchmark" - performance comparison
     - "Automated Workflow" - using agent instructions
   - Each example includes:
     - Description
     - Required files (circuits, configs)
     - Step-by-step instructions
     - Expected output
   - Provide downloadable example files

8. **Create Video Tutorials:**
   - Record screen captures of common tasks
   - Topics:
     - "Getting Started with Proxima" (5 min)
     - "Running Your First Circuit" (3 min)
     - "Multi-Backend Comparison" (7 min)
     - "Using Agent Instructions" (10 min)
   - Host on YouTube or in documentation
   - Link from documentation

9. **Write FAQ:**
   - Common questions and answers
   - Categories:
     - Installation issues
     - Execution errors
     - Performance questions
     - Feature questions
   - Update based on user feedback

10. **Create Glossary:**
    - Define quantum and Proxima-specific terms
    - Terms:
      - Backend, Circuit, Qubit, Gate, Shot, Noise Model, Fidelity, etc.
      - Proxima terms: Execution, Plan, Stage, Checkpoint, etc.
    - Linked from throughout documentation

11. **Build and Deploy Documentation:**
    - Build: `mkdocs build` - generates static HTML
    - Preview: `mkdocs serve` - local preview server
    - Deploy to GitHub Pages: `mkdocs gh-deploy`
    - Custom domain: configure in repo settings
    - Versioned docs: document each release version

12. **Testing Documentation:**
    - Review all documentation for accuracy
    - Check all links work
    - Verify code examples run successfully
    - Test tutorials step-by-step
    - Get user feedback on clarity
    - Check documentation builds without errors

### Step 15.4: Create Example Workflows and Templates

**Objective:** Provide ready-to-use example workflows and templates for common use cases.

**Detailed Instructions:**

1. **Design Example Library Structure:**
   - Examples stored in `examples/` directory
   - Categories:
     - `basic/` - simple examples for beginners
     - `intermediate/` - common workflows
     - `advanced/` - complex scenarios
     - `templates/` - reusable templates
   - Each example in own directory with all files

2. **Create Basic Examples:**
   - **Bell State** (`examples/basic/bell-state/`):
     - `bell.qasm` - Bell state circuit
     - `config.yaml` - execution config
     - `README.md` - instructions
     - Command: `proxima execute examples/basic/bell-state/bell.qasm`
   - **Superposition** (`examples/basic/superposition/`):
     - Simple superposition circuit
   - **Entanglement** (`examples/basic/entanglement/`):
     - 3-qubit entanglement
   - **Quantum Teleportation** (`examples/basic/teleportation/`):
     - Teleportation protocol circuit

3. **Create Intermediate Examples:**
   - **Grover's Algorithm** (`examples/intermediate/grover/`):
     - Grover search implementation
     - Parameter sweep over iterations
     - Analysis of success probability
   - **Quantum Fourier Transform** (`examples/intermediate/qft/`):
     - QFT circuit
     - Verification with known inputs
   - **Noise Study** (`examples/intermediate/noise-study/`):
     - Same circuit with different noise models
     - Comparison of results
     - Analysis of noise impact
   - **Backend Comparison** (`examples/intermediate/backend-comparison/`):
     - Run circuit on all backends
     - Compare execution times and results
     - Consensus analysis

4. **Create Advanced Examples:**
   - **Variational Quantum Eigensolver** (`examples/advanced/vqe/`):
     - VQE for small molecule
     - Parameterized circuits
     - Classical optimization loop
     - Agent instruction for full workflow
   - **Quantum Error Correction** (`examples/advanced/qec/`):
     - Surface code implementation
     - Error injection and correction
     - Logical qubit operations
   - **Circuit Optimization** (`examples/advanced/circuit-optimization/`):
     - Original and optimized circuits
     - Comparison of depth and gates
     - Performance impact analysis
   - **Large-Scale Simulation** (`examples/advanced/large-scale/`):
     - 20+ qubit circuit
     - Resource monitoring
     - Checkpointing and resume
     - Distributed execution (if supported)

5. **Create Workflow Templates:**
   - **Parameter Sweep Template** (`templates/parameter-sweep.md`):
     - Agent instruction template
     - Variables: circuit file, parameter range
     - Loops over parameter values
     - Collects and analyzes results
   - **Benchmark Template** (`templates/benchmark.md`):
     - Run same circuit multiple times
     - Collect timing statistics
     - Generate performance report
   - **Noise Calibration Template** (`templates/noise-calibration.md`):
     - Run calibration circuits
     - Fit noise model to results
     - Apply calibrated model to target circuit
   - **Multi-Backend Validation Template** (`templates/multi-backend-validation.md`):
     - Execute on all backends
     - Compare results
     - Flag discrepancies

6. **Create Circuit Library:**
   - Common quantum circuits in `circuits/` directory
   - Categories:
     - `gates/` - single circuits demonstrating each gate
     - `algorithms/` - famous quantum algorithms
     - `benchmarks/` - standard benchmark circuits
   - Formats: QASM, JSON, Cirq, Qiskit
   - Each circuit with documentation

7. **Create Configuration Examples:**
   - Example configs in `examples/configs/`
   - **Minimal Config** (`minimal.yaml`):
     - Bare minimum configuration
   - **Production Config** (`production.yaml`):
     - Recommended settings for production
     - Logging, persistence enabled
   - **High-Performance Config** (`high-performance.yaml`):
     - Optimized for speed
   - **Development Config** (`development.yaml`):
     - Debug logging, verbose output

8. **Create Plugin Examples:**
   - Example plugins in `examples/plugins/`
   - **Hello World Plugin** (`hello-plugin/`):
     - Minimal plugin demonstrating basics
   - **Custom Backend Plugin** (`custom-backend-plugin/`):
     - Add fictional backend
   - **Analysis Plugin** (`analysis-plugin/`):
     - Add custom result analysis tool

9. **Create LLM Prompt Templates:**
   - Prompt templates for LLM integration
   - In `examples/prompts/`
   - **Result Interpretation Prompt**:
     - Template for asking LLM to interpret results
   - **Workflow Generation Prompt**:
     - Template for generating agent instructions
   - **Error Analysis Prompt**:
     - Template for LLM-assisted debugging

10. **Package Examples:**
    - Include examples in installation
    - Command: `proxima examples list` - list examples
    - Command: `proxima examples run <name>` - run example
    - Command: `proxima examples copy <name> <destination>` - copy example to directory
    - Method: `list_examples() -> List[ExampleInfo]`
    - Method: `run_example(name: str)`

11. **Testing Examples:**
    - Automated testing: run all examples
    - Verify examples complete successfully
    - Check expected outputs match
    - Test on clean installations
    - Update examples when APIs change

### Step 15.5: Implement Accessibility Features

**Objective:** Ensure Proxima is accessible to users with disabilities and various terminal environments.

**Detailed Instructions:**

1. **Implement Screen Reader Support:**
   - Screen readers read terminal content aloud
   - Ensure all UI elements have descriptive text
   - Use semantic markup where possible
   - Test with screen readers (NVDA, JAWS, Orca)
   - Use aria-labels equivalent for terminal UI (descriptive text)

2. **Implement Keyboard Navigation:**
   - Full keyboard navigation (no mouse required)
   - Clear focus indicators
   - Tab order logical and intuitive
   - Keyboard shortcuts for all major actions
   - Document all keyboard shortcuts
   - Config: `ui.keyboard_shortcuts` - customizable

3. **Implement High Contrast Mode:**
   - High contrast theme for visual impairments
   - Maximum contrast between text and background
   - No color-coding only (use symbols too)
   - Test with contrast checking tools

4. **Implement Reduced Motion Mode:**
   - Some users sensitive to animations
   - Config: `ui.animations.enabled = false`
   - Disable all animations and transitions
   - Instant UI updates instead of smooth transitions

5. **Implement Configurable Font Sizes:**
   - Allow users to adjust text size
   - Config: `ui.font_size` - small, medium, large, extra-large
   - Scales all text proportionally
   - Command: `set font-size large`

6. **Implement Color-Blind Modes:**
   - Themes optimized for color blindness types:
     - Deuteranopia (red-green)
     - Protanopia (red-green)
     - Tritanopia (blue-yellow)
   - Use patterns/symbols in addition to colors
   - Status indicators: ✓ ✗ ⚠ not just colors

7. **Implement Screen Reader Announcements:**
   - Announce important events:
     - "Execution started"
     - "Execution completed successfully"
     - "Error occurred"
   - Method: `announce(message: str, priority: str)`
   - Priority: low, medium, high, urgent
   - Integrates with screen reader APIs

8. **Implement Alternative Text for Visual Elements:**
   - Progress bars: also show percentage numerically
   - Charts/graphs: provide textual summary
   - Icons: include text labels
   - Command: `describe <element>` - get textual description

9. **Implement Accessibility Settings UI:**
   - Settings screen section: "Accessibility"
   - Options:
     - Enable screen reader support
     - High contrast mode
     - Disable animations
     - Font size
     - Color-blind mode
   - Quick toggle: `accessibility mode on/off`

10. **Testing Accessibility:**
    - Test with screen readers
    - Test keyboard-only navigation
    - Test with high contrast theme
    - Test with animations disabled
    - User testing with users with disabilities
    - Automated accessibility checks

### Step 15.6: Final UI/UX Polish and Integration

**Objective:** Final polish pass and integration of all UI/UX improvements.

**Detailed Instructions:**

1. **Conduct UI/UX Audit:**
   - Review every screen with critical eye
   - Check for:
     - Consistency across screens
     - Proper alignment and spacing
     - Clear information hierarchy
     - Logical flow between screens
     - No UI bugs or glitches
   - Create issue list, prioritize, fix

2. **Implement Onboarding Flow:**
   - First-time user experience
   - Welcome screen on first launch
   - Quick setup wizard:
     - Select backends to enable
     - Configure LLM providers (optional)
     - Choose theme
   - Offer to run first tutorial
   - Method: `show_onboarding_wizard()`

3. **Implement User Preferences:**
   - Comprehensive preferences system
   - Settings categorized:
     - Appearance (theme, font size, animations)
     - Behavior (auto-save, confirmations)
     - Backends (default backend, options)
     - LLM (default provider, model)
     - Advanced (logging, performance)
   - Settings screen with search
   - Settings stored in `~/.proxima/config/user_preferences.yaml`

4. **Implement Notification System:**
   - Non-intrusive notifications for events
   - Notification types:
     - Info (execution completed)
     - Warning (backend slow)
     - Error (execution failed)
     - Success (operation succeeded)
   - Display: toast notifications (corner of screen)
   - Notification history: view past notifications
   - Config: `ui.notifications.enabled`

5. **Implement Command Palette:**
   - Quick command access (like VS Code)
   - Keyboard shortcut: `Ctrl+Shift+P` or `Cmd+Shift+P`
   - Fuzzy search commands
   - Show recently used commands
   - Execute command directly from palette
   - Use Textual's command palette support

6. **Implement Status Bar:**
   - Bottom status bar showing:
     - Current session duration
     - Active backend
     - Recent execution status
     - Memory usage
     - LLM usage (cost)
   - Clickable elements: click backend → switch backend
   - Auto-hide on small terminals

7. **Implement Quick Actions:**
   - Context-sensitive quick actions
   - Right-click or action menu shows relevant actions
   - Example: on execution in history → Replay, Export, Delete
   - Keyboard shortcuts for quick actions

8. **Implement Smart Suggestions:**
   - Proactive suggestions based on context
   - Examples:
     - "This circuit might run faster on cirq-sv"
     - "Consider using noise model for more realistic results"
     - "Try multi-backend comparison to validate"
   - Display as info boxes (dismissable)
   - Learn from user actions (if user dismisses, don't suggest again)

9. **Implement Error Handling UI:**
   - User-friendly error messages
   - Error dialog with:
     - What went wrong (user-friendly)
     - Why it happened (technical detail, collapsible)
     - How to fix (suggestions)
     - Actions: Retry, Report Bug, Get Help
   - Copy error to clipboard for reporting

10. **Performance Optimization:**
    - Profile UI performance
    - Optimize slow screens
    - Lazy loading for large data sets
    - Virtual scrolling for long lists
    - Debounce search inputs
    - Minimize redraws
    - Target: <16ms frame time for 60 FPS

11. **Integration Testing:**
    - End-to-end UI testing
    - Test all screens accessible
    - Test all commands work from UI
    - Test theme changes apply everywhere
    - Test help system accessible from anywhere
    - Test examples run from UI

12. **User Testing:**
    - Beta testing with real users
    - Collect feedback on usability
    - Identify pain points
    - Iterate on UI based on feedback
    - Conduct usability studies

### Phase 15 Implementation Complete

**Artifacts Produced:**
- Professional theming system with multiple built-in themes
- Comprehensive help system with interactive tutorials
- Complete documentation (user guide, developer guide, API reference)
- Example library with basic, intermediate, and advanced workflows
- Workflow templates for common use cases
- Circuit library with standard quantum circuits
- Accessibility features (screen reader, high contrast, keyboard navigation)
- Polished UI with smooth animations and responsive layouts
- Onboarding flow for new users
- Command palette for quick access
- User preferences system
- Notification system for events
- Smart suggestions and error handling

**Verification:**
- All themes apply correctly and look professional
- Help system is comprehensive and searchable
- Documentation is complete, accurate, and well-organized
- Examples run successfully and demonstrate features
- Accessibility features work with assistive technologies
- UI is consistent, polished, and responsive
- User testing feedback is positive
- Performance is smooth (60 FPS target met)
- All UI tests pass with coverage >80%

---

## PHASE 16: TESTING, HARDENING & RELEASE

### Step 16.1: Implement Comprehensive Test Suite

**Objective:** Create complete test coverage for all Proxima components ensuring reliability and correctness.

**Detailed Instructions:**

1. **Set Up Testing Framework:**
   - Testing library: pytest (Python standard)
   - Install: `pip install pytest pytest-cov pytest-asyncio pytest-mock`
   - Test directory structure:
     ```
     tests/
       unit/           - unit tests (individual functions/classes)
       integration/    - integration tests (component interactions)
       e2e/           - end-to-end tests (full workflows)
       fixtures/      - test data (circuits, configs)
       conftest.py    - pytest configuration and fixtures
     ```
   - Run tests: `pytest tests/`
   - Coverage report: `pytest --cov=proxima --cov-report=html`

2. **Implement Unit Tests:**
   - Test every module in isolation
   - **Config Module** (`tests/unit/test_config.py`):
     - Test config loading from YAML
     - Test config validation (Pydantic)
     - Test default values
     - Test invalid configs raise errors
   - **Backend Adapters** (`tests/unit/test_backends.py`):
     - Test each backend adapter
     - Mock actual backend libraries
     - Test circuit translation
     - Test result parsing
     - Test error handling
   - **Execution Controller** (`tests/unit/test_execution.py`):
     - Test execution flow
     - Test stage management
     - Test error handling and retries
     - Test resource checks
   - **LLM Hub** (`tests/unit/test_llm.py`):
     - Test LLM provider integration
     - Mock LLM API calls
     - Test prompt formatting
     - Test response parsing
   - **Parsers** (`tests/unit/test_parsers.py`):
     - Test QASM parser
     - Test JSON parser
     - Test CSV parser
   - Target: 80%+ unit test coverage

3. **Implement Integration Tests:**
   - Test interactions between components
   - **Backend Integration** (`tests/integration/test_backend_integration.py`):
     - Test actual execution on real backends (LRET, Cirq, Qiskit)
     - Use simple test circuits (1-2 qubits)
     - Verify results are reasonable
     - Test all backend methods
   - **Execution Pipeline** (`tests/integration/test_execution_pipeline.py`):
     - Test full execution: circuit → backend → result
     - Test with different circuits and backends
     - Test execution options
   - **Persistence** (`tests/integration/test_persistence.py`):
     - Test database operations
     - Test session creation and storage
     - Test history queries
     - Test checkpointing
   - **LLM Integration** (`tests/integration/test_llm_integration.py`):
     - Test LLM-powered features (if API key available)
     - Mock LLM if no API key
     - Test result interpretation
     - Test workflow generation
   - **UI Components** (`tests/integration/test_ui.py`):
     - Test TUI screens render correctly
     - Use Textual's testing framework
     - Test user interactions
     - Test screen transitions

4. **Implement End-to-End Tests:**
   - Test complete user workflows
   - **Basic Workflow** (`tests/e2e/test_basic_workflow.py`):
     - Load config
     - Launch Proxima
     - Load circuit
     - Execute circuit
     - View results
     - Exit
   - **Multi-Backend Comparison** (`tests/e2e/test_comparison.py`):
     - Execute same circuit on multiple backends
     - Compare results
     - Verify consensus
   - **Planning Workflow** (`tests/e2e/test_planning.py`):
     - Create multi-stage plan
     - Execute plan
     - Verify all stages complete
   - **Checkpoint and Resume** (`tests/e2e/test_checkpoint_resume.py`):
     - Start long execution
     - Create checkpoint
     - Interrupt execution
     - Resume from checkpoint
     - Verify completion
   - **Agent Instructions** (`tests/e2e/test_agent_instructions.py`):
     - Load instruction file
     - Execute workflow
     - Verify outputs

5. **Implement Performance Tests:**
   - Benchmark critical operations
   - In `tests/performance/test_performance.py`
   - **Execution Performance**:
     - Measure execution time for various circuits
     - Baseline performance metrics
     - Detect performance regressions
   - **UI Responsiveness**:
     - Measure screen render times
     - Ensure UI remains responsive during executions
   - **Memory Usage**:
     - Monitor memory during large simulations
     - Detect memory leaks
   - **Database Performance**:
     - Query performance for large histories
     - Ensure queries complete quickly

6. **Implement Property-Based Tests:**
   - Use Hypothesis library: `pip install hypothesis`
   - Generate random inputs, test properties hold
   - Example: `tests/unit/test_circuit_properties.py`
     - Generate random circuits
     - Test: circuit parsing then unparsing equals original
     - Test: executing circuit doesn't crash
     - Test: result measurements sum to shots count

7. **Implement Snapshot Tests:**
   - For UI testing: compare rendered output to snapshots
   - Library: `pytest-snapshot`
   - Capture TUI screen output
   - Compare to saved snapshot
   - Detect unintended UI changes

8. **Implement Regression Tests:**
   - Tests for previously fixed bugs
   - Each bug fix gets a regression test
   - Ensures bug doesn't reappear
   - In `tests/regression/`

9. **Implement Test Fixtures:**
   - Reusable test data in `tests/fixtures/`
   - **Test Circuits**:
     - Small circuits (1-3 qubits) for fast tests
     - Medium circuits (5-10 qubits)
     - Large circuits (15-20 qubits) for stress tests
   - **Test Configs**:
     - Valid configs for different scenarios
     - Invalid configs to test error handling
   - **Mock Data**:
     - Mock LLM responses
     - Mock backend results
   - Load fixtures in tests via pytest fixtures

10. **Implement Continuous Testing:**
    - GitHub Actions CI/CD pipeline
    - `.github/workflows/test.yml`
    - On every push and PR:
      - Run unit tests
      - Run integration tests
      - Generate coverage report
      - Fail if coverage drops below threshold
    - Run on multiple platforms: Linux, macOS, Windows
    - Run on multiple Python versions: 3.9, 3.10, 3.11, 3.12

11. **Testing Test Quality:**
    - Mutation testing: `pip install mutmut`
    - Mutate code (introduce bugs)
    - Check if tests catch mutations
    - High mutation score = good tests
    - Code review for test quality

### Step 16.2: Security Hardening

**Objective:** Ensure Proxima is secure against common vulnerabilities and threats.

**Detailed Instructions:**

1. **Conduct Security Audit:**
   - Review code for security issues:
     - Injection vulnerabilities
     - Insecure dependencies
     - Insecure configurations
     - Information disclosure
   - Use automated tools: Bandit, Safety
   - Manual code review for sensitive areas

2. **Implement Input Validation:**
   - Validate all user inputs
   - **Command Inputs**:
     - Sanitize command arguments
     - Prevent command injection
     - Whitelist allowed characters
   - **File Paths**:
     - Prevent directory traversal (../ attacks)
     - Validate paths before accessing
     - Use `pathlib.Path.resolve()` for safe path handling
   - **QASM/Circuit Inputs**:
     - Parse safely, reject malformed circuits
     - Limit circuit size (qubits, depth)
     - Prevent resource exhaustion
   - **Config Files**:
     - Validate against schema
     - Reject suspicious values

3. **Implement Secure API Key Handling:**
   - Never store API keys in code or config files
   - Store in environment variables or secure key store
   - Config: reference env vars: `llm.openai.api_key = ${OPENAI_API_KEY}`
   - Mask API keys in logs and UI (show only last 4 chars)
   - Warn if API key in config file

4. **Implement Secure File Handling:**
   - Safe file operations:
     - Check file permissions before reading/writing
     - Avoid overwriting critical files
     - Prompt user for confirmation before destructive ops
   - Sandbox custom commands and plugins:
     - Restrict file system access
     - Use chroot or similar isolation
   - Validate file formats before parsing

5. **Implement Dependency Security:**
   - Audit dependencies for known vulnerabilities
   - Tool: `pip install safety`
   - Run: `safety check` - check for CVEs
   - Update vulnerable dependencies promptly
   - Pin dependency versions in `requirements.txt`
   - Automated: Dependabot alerts on GitHub

6. **Implement Network Security:**
   - Secure communication with external services:
     - Use HTTPS for API calls
     - Validate SSL certificates
     - Timeout on network requests (prevent hanging)
   - Firewall considerations:
     - Document network ports used (MCP, LSP servers)
     - Allow users to configure bindings (localhost only by default)

7. **Implement Secrets Management:**
   - Method: `get_secret(key: str) -> str`
   - Secrets loaded from:
     - Environment variables
     - OS keyring (keyring library)
     - Encrypted config files
   - Never log secrets
   - Clear secrets from memory when done

8. **Implement Audit Logging:**
   - Log security-relevant events:
     - Authentication attempts (if multi-user)
     - Permission changes
     - Data exports
     - API key usage
   - Tamper-proof logs (append-only)
   - Separate security log file
   - Review logs periodically

9. **Implement Sandboxing for Plugins:**
   - Plugins run in restricted environment
   - Use Python sandboxing:
     - RestrictedPython library
     - Limit access to dangerous modules (os, subprocess, etc.)
   - Plugins declare permissions, user approves
   - Runtime enforcement of permissions

10. **Security Testing:**
    - Penetration testing (if applicable)
    - Fuzz testing: send random/malformed inputs
    - Verify input validation catches attacks
    - Test with vulnerable dependencies (in test env)
    - Security scan with Bandit: `bandit -r proxima/`

### Step 16.3: Performance Optimization and Profiling

**Objective:** Optimize Proxima for maximum performance and efficiency.

**Detailed Instructions:**

1. **Profile Application Performance:**
   - Profiling tools:
     - cProfile (Python standard library)
     - py-spy (sampling profiler)
     - memory_profiler (memory usage)
   - Profile typical workflows:
     - Startup time
     - Circuit execution
     - Large result processing
     - UI rendering
   - Command: `python -m cProfile -o profile.stats proxima/main.py`
   - Analyze: `snakeviz profile.stats` (visualization)

2. **Optimize Critical Paths:**
   - Identify bottlenecks from profiling
   - **Startup Optimization**:
     - Lazy imports (import modules only when needed)
     - Cache config loading
     - Defer plugin loading
   - **Execution Optimization**:
     - Minimize overhead between Proxima and backend
     - Batch operations where possible
     - Avoid unnecessary data copies
   - **UI Optimization**:
     - Virtual scrolling for large lists
     - Debounce updates (don't redraw every millisecond)
     - Use efficient Textual widgets

3. **Optimize Memory Usage:**
   - Profile memory: `python -m memory_profiler proxima/main.py`
   - Identify memory-intensive operations
   - **Result Storage**:
     - Store large arrays on disk, not in memory
     - Use memory-mapped files for large data
     - Compress stored data
   - **Circuit Representation**:
     - Efficient data structures
     - Avoid redundant copies
   - **Garbage Collection**:
     - Tune Python GC settings if needed
     - Explicitly delete large objects when done

4. **Optimize Database Queries:**
   - Use SQLAlchemy query profiling
   - Identify slow queries
   - Add indexes on frequently queried columns
   - Optimize complex joins
   - Use query caching for repeated queries
   - Batch inserts instead of individual inserts

5. **Implement Caching:**
   - Cache frequently accessed data
   - **Result Caching**:
     - Cache recent execution results
     - LRU cache (Least Recently Used)
     - Library: `functools.lru_cache`
   - **Config Caching**:
     - Cache parsed config, avoid re-parsing
   - **Backend Caching**:
     - Cache backend initialization
     - Reuse backend instances
   - Clear caches when data changes

6. **Implement Parallel Processing:**
   - Use multiprocessing for CPU-intensive tasks
   - **Parallel Executions**:
     - Run multiple executions concurrently
     - Use ProcessPoolExecutor
     - Limit concurrent executions based on resources
   - **Parallel Comparisons**:
     - Execute on multiple backends in parallel
     - Wait for all to complete, then compare
   - Thread safety: ensure shared data is protected

7. **Optimize Backend Adapters:**
   - Minimize translation overhead
   - Cache translated circuits if executed multiple times
   - Use native backend features when possible
   - Avoid unnecessary conversions

8. **Implement Performance Monitoring:**
   - Built-in performance monitoring
   - Method: `get_performance_metrics() -> PerfMetrics`
   - Metrics:
     - Average execution time per backend
     - Execution overhead (Proxima vs backend)
     - Memory usage over time
     - UI frame rate
   - Display: `performance` command

9. **Set Performance Targets:**
   - Define acceptable performance levels:
     - Startup: <2 seconds
     - Simple execution overhead: <100ms
     - UI responsiveness: 60 FPS (16ms frame time)
     - Database queries: <50ms (most)
   - Automated tests fail if targets not met

10. **Testing Performance:**
    - Run performance tests regularly
    - Compare against baseline
    - Detect regressions early
    - Optimize iteratively

### Step 16.4: Error Handling and Robustness

**Objective:** Ensure Proxima handles errors gracefully and recovers from failures.

**Detailed Instructions:**

1. **Implement Comprehensive Error Handling:**
   - Catch all exceptions at appropriate levels
   - Never let exceptions crash Proxima
   - Top-level exception handler:
     - Catch unexpected exceptions
     - Log error with full traceback
     - Display user-friendly message
     - Offer to save state and exit gracefully

2. **Implement Error Types:**
   - Custom exception hierarchy
   - In `proxima/errors.py`:
     - `ProximaError` (base class)
     - `ConfigError` (invalid config)
     - `BackendError` (backend failure)
     - `ExecutionError` (execution failure)
     - `ParseError` (parsing failure)
     - `ResourceError` (insufficient resources)
   - Use specific exceptions for different error types

3. **Implement Retry Logic:**
   - Transient errors (network, temporary backend issues) → retry
   - Method: `retry_with_backoff(func, max_retries=3)`
   - Exponential backoff: wait 1s, 2s, 4s, etc.
   - Configurable: `execution.retry.enabled`, `execution.retry.max_attempts`

4. **Implement Graceful Degradation:**
   - If non-critical feature fails, continue with reduced functionality
   - Examples:
     - LLM unavailable → skip interpretation, show raw results
     - Plugin fails to load → disable plugin, continue
     - Visualization fails → show textual output
   - Notify user of degraded functionality

5. **Implement Circuit Validation:**
   - Validate circuits before execution
   - Method: `validate_circuit(circuit) -> ValidationResult`
   - Checks:
     - Syntax correctness
     - Gate definitions exist
     - Qubit indices in range
     - No undefined classical registers
   - Prevent execution of invalid circuits

6. **Implement Resource Validation:**
   - Check resources before execution
   - Fail fast if insufficient resources
   - Clear error message: "Circuit requires 16 GB memory, 8 GB available"
   - Suggest: reduce shots, simplify circuit, use different backend

7. **Implement Crash Recovery:**
   - Auto-save state periodically
   - On crash: save emergency checkpoint
   - On restart: offer to recover
   - Method: `recover_from_crash() -> bool`

8. **Implement Error Reporting:**
   - Easy bug reporting for users
   - Command: `report bug` - collect diagnostics and create report
   - Report includes:
     - Error message and traceback
     - Proxima version
     - System info (OS, Python version)
     - Config (sanitized, no secrets)
     - Steps to reproduce (if known)
   - Submit to GitHub issues or email

9. **Implement Warning System:**
   - Non-fatal issues → warnings, not errors
   - Warning types:
     - Deprecated features
     - Performance concerns
     - Unusual configurations
   - Display warnings to user (dismissable)
   - Log warnings

10. **Testing Error Handling:**
    - Negative testing: test with invalid inputs
    - Inject failures: mock backend failures, network errors
    - Verify errors caught and handled
    - Verify error messages are helpful
    - Test crash recovery

### Step 16.5: Build and Packaging

**Objective:** Package Proxima for easy installation and distribution.

**Detailed Instructions:**

1. **Set Up Build System:**
   - Use setuptools or Poetry for packaging
   - `pyproject.toml` configuration:
     - Project metadata (name, version, description, authors)
     - Dependencies
     - Entry points (CLI command)
     - Build system (setuptools, Poetry)
   - Version management: semantic versioning (MAJOR.MINOR.PATCH)

2. **Configure Package Metadata:**
   - In `pyproject.toml`:
     ```toml
     [project]
     name = "proxima"
     version = "1.0.0"
     description = "AI-powered quantum simulation orchestration agent"
     authors = [{name = "Your Name", email = "email@example.com"}]
     license = {text = "MIT"}
     readme = "README.md"
     requires-python = ">=3.9"
     dependencies = [
         "textual>=0.40.0",
         "pydantic>=2.0.0",
         "loguru>=0.7.0",
         # ... all dependencies
     ]
     [project.scripts]
     proxima = "proxima.cli:main"
     ```

3. **Include Non-Python Files:**
   - Package data files (themes, help content, examples)
   - In `pyproject.toml`:
     ```toml
     [tool.setuptools.package-data]
     proxima = [
         "ui/themes/*.json",
         "help/content/**/*.md",
         "examples/**/*"
     ]
     ```
   - Use `importlib.resources` to access packaged files

4. **Build Package:**
   - Build distribution files:
     - Source distribution: `python -m build --sdist`
     - Wheel: `python -m build --wheel`
   - Output: `dist/proxima-1.0.0.tar.gz` and `dist/proxima-1.0.0-py3-none-any.whl`

5. **Create Docker Image:**
   - `Dockerfile` for containerized deployment
   - Base image: Python 3.11 slim
   - Install Proxima and dependencies
   - Install backend libraries (LRET, Cirq, Qiskit)
   - Entrypoint: `proxima`
   - Build: `docker build -t proxima:1.0.0 .`
   - Test: `docker run -it proxima:1.0.0`

6. **Create Conda Package:**
   - For Conda users
   - `meta.yaml` for Conda recipe
   - Build: `conda build .`
   - Publish to conda-forge (if open source)

7. **Platform-Specific Builds:**
   - Test installation on:
     - Linux (Ubuntu, Fedora, Arch)
     - macOS (Intel and Apple Silicon)
     - Windows (10, 11)
   - Handle platform-specific dependencies
   - Provide installation instructions for each

8. **Create Standalone Executables:**
   - For users without Python installed
   - Use PyInstaller or Nuitka
   - Bundle Python and all dependencies
   - Single executable file
   - Build: `pyinstaller --onefile proxima/cli.py`
   - Larger file size, but no Python required

9. **Publish to PyPI:**
   - PyPI (Python Package Index) for `pip install proxima`
   - Register account on PyPI
   - Configure authentication: `~/.pypirc`
   - Upload: `twine upload dist/*`
   - Verify: `pip install proxima`

10. **Create Installation Guides:**
    - Document installation methods:
      - pip: `pip install proxima`
      - conda: `conda install -c conda-forge proxima`
      - Docker: `docker pull proxima/proxima:latest`
      - From source: `git clone ... && pip install -e .`
    - Include troubleshooting for common issues

11. **Testing Packages:**
    - Install from built package (not source)
    - Test in clean environment
    - Verify all features work
    - Verify all files included
    - Test on multiple platforms

### Step 16.6: Release Preparation and Deployment

**Objective:** Prepare for public release and establish release process.

**Detailed Instructions:**

1. **Create Release Checklist:**
   - Pre-release checklist:
     - ☐ All tests pass
     - ☐ Code coverage >80%
     - ☐ Documentation complete
     - ☐ Examples tested
     - ☐ Security audit passed
     - ☐ Performance targets met
     - ☐ Packaging tested
     - ☐ Changelog updated
     - ☐ Version numbers updated
     - ☐ License file included
   - Use checklist for each release

2. **Write CHANGELOG:**
   - Document changes in each version
   - `CHANGELOG.md` format:
     ```markdown
     # Changelog
     
     ## [1.0.0] - 2026-01-15
     ### Added
     - Initial release
     - LRET, Cirq, Qiskit backend support
     - LLM integration for result interpretation
     - Multi-backend comparison
     - Planning and staged execution
     - Checkpoint and resume
     - Agent instruction system
     - MCP and LSP integration
     
     ### Fixed
     - None (initial release)
     ```
   - Keep changelog for all releases

3. **Version Management:**
   - Semantic versioning: MAJOR.MINOR.PATCH
   - MAJOR: breaking changes
   - MINOR: new features (backwards compatible)
   - PATCH: bug fixes
   - Update version in:
     - `pyproject.toml`
     - `proxima/__init__.py` (`__version__`)
     - Documentation

4. **Create Release Notes:**
   - For each release, write release notes
   - Highlight:
     - Major new features
     - Important bug fixes
     - Breaking changes (if any)
     - Upgrade instructions
   - Post on GitHub releases page

5. **Tag Release:**
   - Git tag for each release
   - Tag name: `v1.0.0` (semantic version with 'v' prefix)
   - Command: `git tag -a v1.0.0 -m "Release version 1.0.0"`
   - Push tags: `git push --tags`

6. **Create GitHub Release:**
   - On GitHub: Releases → Draft a new release
   - Tag: select tag (or create on release)
   - Release title: "Proxima 1.0.0 - Initial Release"
   - Release description: copy from release notes
   - Attach distribution files:
     - Source tarball
     - Wheel file
     - Standalone executables (if applicable)
   - Publish release

7. **Publish to PyPI:**
   - Upload to PyPI (if not already done)
   - Verify package appears on PyPI
   - Test: `pip install proxima`
   - PyPI page includes description from README

8. **Publish Docker Image:**
   - Push to Docker Hub or GitHub Container Registry
   - Command: `docker push proxima/proxima:1.0.0`
   - Tag as latest: `docker tag proxima/proxima:1.0.0 proxima/proxima:latest`
   - Update Docker Hub description

9. **Deploy Documentation:**
   - Deploy docs to hosting:
     - GitHub Pages: `mkdocs gh-deploy`
     - Read the Docs: connect GitHub repo
   - Versioned docs: maintain docs for each major version
   - Set up custom domain (if desired)

10. **Announce Release:**
    - Announce on relevant channels:
      - GitHub Discussions
      - Twitter/X, LinkedIn
      - Reddit (r/QuantumComputing, etc.)
      - Quantum computing forums
      - Blog post (if you have a blog)
    - Announcement includes:
      - What Proxima is
      - Key features
      - Installation instructions
      - Link to documentation
      - Link to examples

11. **Set Up Release Automation:**
    - GitHub Actions workflow for releases
    - `.github/workflows/release.yml`
    - On tag push:
      - Run tests
      - Build packages
      - Create GitHub release
      - Publish to PyPI
      - Build and push Docker image
      - Deploy documentation
    - Automated, consistent releases

12. **Establish Support Channels:**
    - How users can get help:
      - GitHub Issues (bug reports, feature requests)
      - GitHub Discussions (questions, community)
      - Email (for security issues)
      - Documentation (first point of reference)
    - Document in README and website

13. **Monitor Post-Release:**
    - After release, monitor:
      - GitHub issues (new bug reports)
      - PyPI downloads (usage metrics)
      - User feedback (Discussions, social media)
    - Respond to issues promptly
    - Plan hotfixes if critical bugs found

14. **Plan Future Releases:**
    - Roadmap for future versions:
      - 1.1.0 - Additional features
      - 1.2.0 - Performance improvements
      - 2.0.0 - Major redesign (if planned)
    - Collect feature requests
    - Prioritize based on user feedback
    - Publish roadmap for transparency

### Phase 16 Implementation Complete

**Artifacts Produced:**
- Comprehensive test suite (unit, integration, e2e) with >80% coverage
- Continuous integration pipeline (GitHub Actions)
- Security hardening (input validation, API key handling, sandboxing)
- Performance optimizations (profiling, caching, parallel processing)
- Robust error handling and crash recovery
- Build and packaging configuration (PyPI, Conda, Docker)
- Release automation (CI/CD for releases)
- Complete documentation and examples
- GitHub release with distribution files
- Published package on PyPI
- Docker image on Docker Hub
- Deployed documentation website
- Release announcement and support channels

**Verification:**
- All tests pass (unit, integration, e2e, performance)
- Code coverage >80%
- Security scan (Bandit) shows no critical issues
- Performance targets met (startup <2s, UI 60 FPS)
- Package installs successfully on all platforms
- Docker image runs correctly
- Documentation website accessible and complete
- Examples run successfully
- GitHub release created
- PyPI package published
- Community can install and use Proxima
- Support channels established and monitored

---

# END OF PHASE-BY-PHASE IMPLEMENTATION GUIDE

**All 16 Phases Complete!**

This completes the comprehensive Phase-by-Phase Implementation Guide for **Proxima**, the AI-powered quantum simulation orchestration agent. The guide covers:

1. Foundation & Core Infrastructure
2. Backend Abstraction Layer
3. Execution Control System
4. Resource Monitoring & Fail-Safe
5. Additional Backend Adapters (Cirq, Qiskit Aer)
6. LLM Integration Hub
7. Intelligent Backend Selection
8. Planning & Staged Execution
9. Result Interpretation & Insights
10. Multi-Backend Comparison
11. Pause/Resume/Rollback
12. Agent Instruction System
13. Session Management & Persistence
14. Advanced Features (MCP, LSP, Plugins, Skills)
15. UI Polish & Documentation
16. Testing, Hardening & Release

**Implementation Timeline:** 24-34 weeks (6-8 months) for full implementation

**Next Steps:**
- Begin implementation following this guide phase by phase
- Use advanced AI coding assistants (GPT-5.1 Codex Max, Claude Opus 4.5) to implement each step
- Test thoroughly after each phase
- Gather user feedback during beta testing
- Iterate and improve based on real-world usage

**This guide provides everything needed to build Proxima from scratch into a production-ready quantum simulation orchestration platform.**

---

# QUICK REFERENCE: ALL 16 PHASES

1. **Foundation & Core Infrastructure** - Project setup, config, logging, TUI framework
2. **Backend Abstraction Layer** - Unified interface for quantum backends, LRET integration
3. **Execution Control System** - Job submission, monitoring, error handling
4. **Resource Monitoring & Fail-Safe** - Memory/CPU tracking, preemptive checks
5. **Additional Backend Adapters** - Cirq DM/SV, Qiskit Aer DM/SV adapters
6. **LLM Integration Hub** - Multi-provider LLM support, API management
7. **Intelligent Backend Selection** - Smart backend recommendation system
8. **Planning & Staged Execution** - Multi-stage workflows, dependencies
9. **Result Interpretation & Insights** - Statistical analysis, LLM-powered insights
10. **Multi-Backend Comparison** - Run same circuit on multiple backends, compare
11. **Pause/Resume/Rollback** - Checkpoint system, state management
12. **Agent Instruction System** - Markdown-based automation workflows
13. **Session Management & Persistence** - Database, history tracking, replay
14. **Advanced Features** - MCP, LSP, plugins, custom commands, skills
15. **UI Polish & Documentation** - Themes, help system, comprehensive docs
16. **Testing, Hardening & Release** - Testing, security, packaging, deployment