# CODE4LOGIC vs NL2LOGIC: Phase Transition Evaluation

Evaluating the CODE4LOGIC and NL2LOGIC pipelines through the lens of Zhang et al. (2026), analysing performance as a function of **Logical Complexity Metric (LoCM)** to determine whether NL2LOGIC's AST-guided recursive approach extends the reasoning "collapse threshold" beyond CODE4LOGIC's progressive code generation.

---

## Repository Structure

```
Final_Year_Project/
├── data/                           # Raw datasets & pre-processing logic
│   ├── FOLIO/
│   ├── LOGICNLI/
│   ├── NSA_LR/
│   ├── preprocess.py               # Dataset downloading
│   └── prepare_dataset.py          # Dataset LoCM calculation & bin-stratification
│
├── pipelines/                      # Evaluation pipelines & orchestration
│   ├── run_pipeline.py             # Orchestrator to run preprocessing and evaluations end-to-end
│   ├── evaluate_qwen.py            # CODE4LOGIC evaluation script
│   ├── evaluate_nl2logic.py        # NL2LOGIC evaluation script
│   ├── locm_metric.py              # LoCM metric scorer function
│   ├── plot_phase_transitions.py   # Accuracy vs. LoCM plotter
│   │
│   ├── code4logic/                 # CODE4LOGIC logic generation & verification
│   │   ├── basis_functions.py      # FOL construction primitives
│   │   ├── prompts.py              # In-context learning prompt builder
│   │   ├── fol_verifier.py         # Equivalence checker (Z3 → Prover9 → string match)
│   │   └── fol_grammar.py          # BNF grammar for FOL parsing
│   │
│   ├── dataset_adapters/           # Adapters for standardising dataset formats
│   │   ├── folio_adapter.py        # Standard 3-label logic with global signatures
│   │   ├── logicnli_adapter.py     # 4-label logic with Paradox detection
│   │   └── nsa_lr_adapter.py       # Fine-grained step-by-step chain analysis
│   │
│   └── nl2logic/                   # NL2LOGIC parsing
│       ├── pipeline.py             # Core rephrase & parse pipeline
│       ├── ast_rl.py               # Typed AST nodes + Z3 code generation
│       └── structured_output.py    # Pydantic schemas for LLM parsing
│
└── Result/                         # Generated results and plots
```

---

## Datasets Supported

The evaluation supports three primary datasets through the `dataset_adapters` abstraction:

1. **FOLIO**: Features expert-written, multi-sentence natural language stories. Tested via the standard 3-label logic (True, False, Unknown). The adapter utilizes a Global Signature Prompt to prevent symbol drift across complex discourse context.
2. **LogicNLI**: A 4-label NLI benchmark that adds a "Paradox" relation. A Paradox arises when the premises are internally contradictory (i.e. both H and ¬H are entailed).
3. **NSA-LR**: Designed for fine-grained Logical Phase Transitions analysis. It provides exhaustive intermediate reasoning steps as structured FOL chains.

---

## Evaluation Methodology

Run both pipelines across all complexity bins and measure the following SOTA metrics:

- **Execution Rate (ExcRate)**: The percentage of generated formulas that are syntactically valid and run in Z3/Prover9 without errors.
- **Semantic Accuracy**: The percentage of instances where the solver's output (True/False/Uncertain/Paradox) matches the ground-truth label.

### LoCM Formula

$$LoCM(\phi) = \sqrt{\sum_{o \in \mathcal{O}} \omega(o)\,freq(o, \phi) + \gamma\, h(\phi)}$$

**Calibrated weights** (Zhang et al., 2026 — Table 7):

| Operator | Weight $\omega$ |
|---|---|
| $\wedge,\ \vee$ | 1.0 |
| $\forall,\ \exists,\ \neg$ | 2.0 |
| $\rightarrow,\ \leftrightarrow$ | 3.0 |
| $\oplus$ (XOR) | 3.5 |
| Hop term $h(\phi)$ | $\gamma = 2.0$ |

---

## How to Run

### Prerequisites

```bash
# Python ≥ 3.9, all deps installed in venv
pip install -r requirements.txt
```

If you plan to use vLLM for high-throughput inference (recommended for the recursive pipeline):
```bash
# Start a local vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --gpu-memory-utilization 0.9 --max-model-len 1024 --enforce-eager
```

### End-to-End Orchestrator

The simplest way to run the entire project is via the orchestrator script. This automatically prepares the datasets, evaluates both pipelines, and generates the final plots. Any arguments passed will be forwarded to the evaluation scripts.

```bash
python pipelines/run_pipeline.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ --dataset folio
```

### Running Scripts Individually

#### 1. Prepare Data
```bash
python data/preprocess.py
python data/prepare_dataset.py
```

#### 2. CODE4LOGIC Evaluation
```bash
python pipelines/evaluate_qwen.py --dataset folio --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ
```
**Arguments:**
- `--dataset` (default: `folio`): The dataset to evaluate (`folio`, `logicnli`, `nsa_lr`).
- `--backend` (default: `vllm`): LLM backend (`transformers` or `vllm`).
- `--model` (default: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`): Model name or path.
- `--samples` (int, default: `None`): Max samples to evaluate per bin (useful for smoke tests).
- `--concurrency` (int, default: `128`): Batching concurrency limits.

#### 3. NL2LOGIC Evaluation
```bash
python pipelines/evaluate_nl2logic.py --dataset folio --backend vllm --model Qwen/Qwen2.5-7B-Instruct-AWQ
```
**Arguments:**
- `--dataset` (default: `folio`): The dataset to evaluate (`folio`, `logicnli`, `nsa_lr`).
- `--backend` (default: `vllm`): LLM backend (`vllm`, `openai`, `mock`).
- `--model` (default: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`): Model name or path.
- `--url` (default: `http://localhost:8000/v1`): vLLM server URL to connect to.
- `--samples` (int, default: `None`): Max samples to evaluate per bin.
- `--concurrency` (int, default: `128`): Async execution concurrency limit.
- `--verbose`: Enable detailed logging of the NL2LOGIC AST parsing process.

#### 4. Generate Plot
```bash
python pipelines/plot_phase_transitions.py --dataset folio
```