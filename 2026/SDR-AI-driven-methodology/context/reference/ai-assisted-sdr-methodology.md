# AI-Assisted SDR Design: Methodology Atlas

**Researched 26 September 2026**

> **AI-Assisted SDR Design** can be defined as a design methodology in which artificial intelligence supports one or more stages of the software-defined radio (SDR) engineering lifecycle — from requirements and architecture selection to DSP implementation, parameter optimization, testing, and deployment.
>
> The central distinction: **AI inside SDR ≠ AI designing SDR.** The AI may work at the design and orchestration level while real-time signal processing stays conventional DSP.

---

## Table of Contents

1. [Overview & Lifecycle Coverage](#1-overview--lifecycle-coverage)
2. [Methodology Landscape](#2-methodology-landscape)
3. [Methodology 1 — LLM-Assisted SDR Design](#3-methodology-1--llm-assisted-sdr-design)
4. [Methodology 2 — RAG-Grounded SDR Design](#4-methodology-2--rag-grounded-sdr-design)
5. [Methodology 3 — Agentic SDR Design](#5-methodology-3--agentic-sdr-design)
6. [Methodology 4 — AI-in-the-Loop Optimization](#6-methodology-4--ai-in-the-loop-optimization)
7. [Methodology 5 — ML-Assisted DSP Design](#7-methodology-5--ml-assisted-dsp-design)
8. [Methodology 6 — Simulation-in-the-Loop](#8-methodology-6--simulation-in-the-loop)
9. [Methodology 7 — Hardware-in-the-Loop (HIL)](#9-methodology-7--hardware-in-the-loop-hil)
10. [Methodology 8 — Closed-Loop Autonomous SDR](#10-methodology-8--closed-loop-autonomous-sdr)
11. [Integrated Framework (∑)](#11-integrated-ai-assisted-sdr-design-framework-)
12. [Relationship Map](#12-relationship-map)
13. [Hybrid Workflows](#13-hybrid-workflows)
14. [Evidence Notes & Discrepancies](#14-evidence-notes--discrepancies)
15. [Sources & References](#15-sources--references)

---

## Provenance Key

Throughout this document, provenance marks indicate where each claim originates:

- **[GR]** — Golden Reference: stated in the attached defining text.
- **[n]** — External evidence: a numbered source from the references section.
- **[INT]** — Interpretation: the atlas's own synthesis — useful, but not a sourced fact.
- **[EX]** — Example: a real project that demonstrates a concept, rated by evidence level.

---

## 1. Overview & Lifecycle Coverage

The ten-step research framework columns, plus runtime operation, are:

| Stage | Index |
|---|---|
| Requirements | 0 |
| Knowledge | 1 |
| Architecture | 2 |
| Implementation | 3 |
| Static validation | 4 |
| Simulation | 5 |
| Hardware | 6 |
| Optimization | 7 |
| Approval | 8 |
| Runtime | 9 |

### Lifecycle Coverage Matrix

| Methodology | Req | Know | Arch | Impl | Static | Sim | HW | Opt | Appr | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. LLM-Assisted | ● | | ● | ● | | | | | ◐ | |
| 2. RAG-Grounded | ◐ | ● | ● | | | | | | | |
| 3. Agentic | ● | ● | ● | ● | | ● | ◐ | ◐ | | |
| 4. Optimization | | | | | | ◐ | ◐ | ● | | |
| 5. ML-Assisted DSP | | | | ● | | ◐ | ◐ | | | ● |
| 6. Simulation-in-Loop | | | | | | ● | | ◐ | | |
| 7. HIL | | | | | | | ● | ◐ | | |
| 8. Closed-Loop | | | | | | | | | | ● |
| ∑ Integrated | ● | ● | ● | ● | ● | ● | ● | ● | ● | |

*(● = primary focus, ◐ = partly involved, blank = not covered)*

### Lifecycle Levels

- **Design-time** — AI helps design the SDR before anything runs (M1, M2, M3, M4).
- **Evaluation** — Where proposed designs get empirical feedback (M6, M7).
- **Runtime** — AI operates inside or on the running radio (M5, M8).
- **Integrated** — A lifecycle that combines several methodologies (∑).

---

## 2. Methodology Landscape

The methodologies sit at different levels of abstraction, so not every pair is a like-for-like alternative. Classification columns are interpretation [INT]; names and roles come from the reference [GR].

| Methodology | Type | Scope | Primary Goal | Level | Comparable | Complementary |
|---|---|---|---|---|---|---|
| 1. LLM-Assisted | Design-assistance (generative) | Req → arch → code; ends at validation | Translate NL requirements into DSP architecture and code drafts | Design | M2, M3 | M6, M7, M4 |
| 2. RAG-Grounded | Knowledge-grounded design (extends M1) | Corpus curation, retrieval, reasoning, design proposal | Propose SDR designs with evidence-backed parameters | Design | M1, M3 | M6, M7, M9 |
| 3. Agentic | Workflow-orchestration (tool-using agent) | Req through testing via tools | Automate the design iteration with tool-use loop | Design | M1, M2 | M4, M6, M7 |
| 4. Optimization | Parameter-optimization | Parameter tuning of known architecture | Search θ for best trade-off defined by J(θ) | Design | M8 | M6, M7, M3 |
| 5. ML-Assisted DSP | Component-level technique (AI inside SDR) | One block's function and data pipeline | Replace or augment a DSP block with a learned model | Runtime | Conventional DSP | M6, M7, M8 |
| 6. Sim-in-the-Loop | Verification (automated evaluation loop) | The proposal–simulation–metrics loop | Verify AI-generated designs empirically before accepting | Evaluation | M7 | M1, M3, M4 |
| 7. HIL | Experimental evaluation and optimization | Configuration → device → IQ → metrics → AI | Optimize and validate against the physical RF chain | Evaluation | M6 | M4, M3, M8 |
| 8. Closed-Loop | Runtime control (cognitive-radio pattern) | Runtime sensing, reasoning, reconfiguration | Continuously adapt {f_c, B, G, waveform, detector} to the RF environment | Runtime | M4 | M5, M7 |
| ∑ Integrated | Integrated lifecycle (hybrid) | Ten steps from requirements to human approval | Evidence-grounded, human-supervised, closed-loop engineering | Integrated | None directly | M5, M8 |

---

## 3. Methodology 1 — LLM-Assisted SDR Design

**Alias:** LLM-Assisted Design · **Role:** Copilot · **Level:** Design-time

### Overview

- **Type:** Design-assistance practice (generative) [INT]
- **Abstraction:** Task level — one requirements-to-artifact generation step
- **Domain:** SDR prototyping, GNU Radio flowgraphs, DSP code in Python and C++
- **Origin:** Defined in the Golden Reference as the simplest methodology [GR]. External precedent: a GRCon 2025 pipeline fine-tuned LLMs for GNU Radio flowgraph construction [2].
- **Purpose:** Translate natural-language requirements into a DSP architecture and code or flowgraph drafts that an engineer validates. [GR]
- **Scope:** Requirements → architecture → code/flowgraph; ends at engineer validation. [GR]

### Process

- **Lifecycle:** Linear: Requirements → LLM → DSP architecture → code/flowgraph → engineer validation. [GR]
- **Planning:** Light — the prompt carries the plan; no formal design review before generation. [INT]
- **Iteration:** Manual re-prompting after the engineer spots a problem; no automated feedback. [INT]
- **Experimentation:** None built in. Generated artifacts stay untested proposals until they run somewhere else. [INT]
- **Validation:** Engineer validation is the single gate (HITL). [GR]
- **Documentation:** Keep requirements, prompts, model version and raw outputs beside the accepted code so the acceptance decision can be reconstructed. [INT]
- **Collaboration:** One engineer with a copilot; team review happens through ordinary code review. [INT]
- **Automation:** Generation is automated; evaluation and correction are manual. [INT]
- **Reproducibility:** Weak by default because LLM outputs vary between runs; mitigated by logging prompts, model versions and seeds where available. [INT]
- **Deployment:** Generated code runs as conventional DSP. The LLM is not placed in the real-time, sample-by-sample processing loop. [GR]
- **Governance:** Accountability sits with the validating engineer. [INT]

### Philosophy & Principles

AI speeds up design-time authoring while humans keep judgement over correctness. [GR] [INT]

1. Keep the LLM out of the latency-critical sample loop [GR]
2. Treat every output as a draft until an engineer validates it [GR]
3. Record what was generated versus what was edited [INT]

### Inputs & Outputs

- **Inputs:** Natural-language requirements (e.g. an FM spectrum-sensing receiver on HackRF One covering 88–108 MHz). [GR]
- **Outputs:** A proposed chain (HackRF → IQ acquisition → FFT/Welch → CFAR → clustering → signal report) plus Python, C++ or GNU Radio components. [GR]
- **Feedback:** Human only — rejected outputs go back to the prompt. [INT]
- **Key Decision:** The engineer accepts, edits or rejects the generated architecture and code. [GR]

### Fit & Limitations

- **Best fit:** Prototypes and textbook DSP chains where an engineer can judge correctness quickly. [INT]
- **Not a fit:** Novel algorithms without a reference design, regulated transmit behaviour, anything needing verified performance numbers. [INT]
- **Assumptions:** The validating engineer knows enough DSP to catch plausible-but-wrong output. [INT]
- **Constraints:** No empirical verification inside the method, so parameter choices may be unsupported. [GR]
- **Strengths:** Fastest path from idea to runnable draft, with almost no infrastructure. [INT]
- **Limitations:** Unsupported parameter selection — the gap RAG targets [GR]. Without domain tooling, LLM-generated flowgraphs are often syntactically plausible but non-functional [9]. LLMs are ill-suited to latency-sensitive or embedded operation [2].

### Workflow Diagram

```
Requirements (NL specification)
    │
    ▼
LLM Generation (copilot prompt)
    │
    ▼
DSP Architecture (proposed block chain)
    │
    ▼
Code / Flowgraph (Python, C++, GNU Radio)
    │                    ╌╌╌╌> Real-time DSP runtime (conventional, no LLM)
    ▼
Engineer Validation (HITL gate)
    │         │
    ▼         ╰──── reject: refine prompt ──▶ [back to LLM Generation]
Accepted Draft (code + flowgraph + review log)
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| GRCon 2025 LLM fine-tuning pipeline | Explicitly documented | Fine-tunes language models for flowgraph construction from natural language. Authors report the pipeline works for construction tasks but needs more prompt diversity, trace coverage and runtime behaviour capture. |
| GR-MCP (Dollarhyde) | Strong evidence | Natural-language flowgraph creation through tool calls that enforce GNU Radio's type system and connection rules. States that LLMs without domain tooling produce plausible but frequently non-functional flowgraphs. |

### Project Layout

```
llm-assisted-sdr/
├── README.md
├── requirements/
│   └── fm_sensing_88-108MHz.md
├── prompts/
│   ├── system_prompt.md
│   └── design_request_v3.md
├── generations/
│   ├── 2026-09-12_run01/
│   │   ├── architecture.md
│   │   ├── fm_sensing.grc
│   │   └── transcript.jsonl
│   └── README.md
├── src/
│   └── fm_sensing/
│       ├── welch_psd.py
│       └── cfar_detector.py
├── flowgraphs/
│   └── fm_sensing.grc
├── review/
│   ├── validation_checklist.md
│   └── review_log.md
└── tests/
    └── test_cfar_detector.py
```

### Tooling

| Tool | Need | Examples | Purpose |
|---|---|---|---|
| Generation | Required | LLM chat or IDE assistant; fine-tuned small models | Produces architecture text, code and flowgraphs |
| SDR framework | Compatible | GNU Radio Companion, Python, C++ | Target formats named in the Golden Reference |
| Version control | Compatible | Git on GitHub or GitLab | Separates generated from edited code |
| Review | Compatible | Pull requests, review checklists | Where the engineer validation gate lives |
| Documentation | Optional | Markdown in the repository | Stores prompts, model versions and transcripts |

### Relationships

| Related To | Type | Note |
|---|---|---|
| M2 (RAG-Grounded) | Overlapping | RAG keeps this generation step and adds retrieved evidence before it; the Golden Reference calls that considerably stronger. [GR] |
| M3 (Agentic) | Overlapping | An agent uses LLM generation as one step inside a larger tool loop. [GR] |
| M6 (Simulation) | Upstream | Outputs need empirical verification; simulation is "significantly safer than accepting generated DSP code without empirical verification." [GR] |
| M5 (ML-Assisted DSP) | Orthogonal | AI designing SDR versus AI inside SDR. [GR] |
| ∑ (Integrated) | Overlapping | Corresponds to steps 3–4 of the integrated framework when a copilot does them. [INT] |

---

## 4. Methodology 2 — RAG-Grounded SDR Design

**Alias:** RAG-Grounded Design · **Role:** Domain-aware assistant · **Level:** Design-time

### Overview

- **Type:** Knowledge-grounded design practice (extends M1)
- **Abstraction:** Task level with a persistent knowledge layer
- **Domain:** Hardware configuration, standards-constrained design, reuse of lab knowledge
- **Origin:** Golden Reference formulation: LLM + SDR knowledge base + retrieval [GR]. Builds on the RAG pattern introduced by Lewis et al. [12]; telecom-specific precedent: Telco-RAG for 3GPP standards [13].
- **Purpose:** Propose SDR designs whose parameters are backed by retrieved evidence, which greatly reduces unsupported parameter selection. [GR]
- **Scope:** Corpus curation, retrieval, reasoning and a design proposal; nothing is executed. [GR] [INT]

### Process

- **Lifecycle:** Q → retrieve evidence → LLM reasoning → SDR design [GR], preceded by corpus ingestion and indexing [INT].
- **Planning:** Upfront work on what enters the corpus and how it is versioned. [INT]
- **Iteration:** Re-query or expand the corpus when evidence is missing or conflicting; the corpus grows as experiments are added. [INT]
- **Experimentation:** Not inherent. It draws on previous experiments and measured datasets stored in the knowledge base. [GR]
- **Validation:** Traceability — each parameter (f_s, N_FFT, G_RF, B, P_FA, detector) should point to a retrieved source a reviewer can check. [GR] [INT]
- **Documentation:** High — corpus manifest, index configuration, and an evidence map per design. [INT]
- **Collaboration:** The knowledge base is a shared team asset and curating it is a recurring job. [INT]
- **Automation:** Retrieval and drafting are automated; ingestion is semi-automated; curation is manual. [INT]
- **Reproducibility:** Better than M1 when corpus and index are versioned, because each answer's provenance is explicit. [INT]
- **Deployment:** Produces a design specification; the method deploys nothing itself. [INT]
- **Governance:** Source licences, standard versions and document freshness become design risks. [INT]

### Philosophy & Principles

Evidence first, then reasoning. [GR]

1. Retrieve before proposing parameters [GR]
2. Cite the evidence behind each parameter [INT]
3. Treat previous experiments and measured datasets as first-class knowledge [GR]

### Inputs & Outputs

- **Inputs:** Design question Q plus a corpus: SDR hardware manuals, GNU Radio documentation, IEEE papers, regulatory standards, RF/DSP textbooks, previous experiments, project requirements, source code, measured datasets. [GR]
- **Outputs:** A design with f_s, N_FFT, G_RF, B, P_FA and detector choice, linked to evidence. [GR]
- **Feedback:** Unsupported or conflicting answers trigger a new query or corpus expansion. [INT]
- **Key Decision:** Is every parameter supported by retrieved evidence? [INT]

### Fit & Limitations

- **Best fit:** Standards- or regulation-constrained projects and teams that reuse internal experiment history. [INT]
- **Not a fit:** Greenfield research where little written evidence exists, and tiny one-off prototypes where corpus cost dominates. [INT]
- **Strengths:** Grounded parameters, reusable team memory and explicit provenance. [INT]
- **Limitations:** Only as good as the corpus and retriever, and still no empirical verification of the design. [INT]

### Workflow Diagram

```
Ingest & Index (curate corpus)
    │                              SDR Knowledge Base
    │                              (manuals, standards, experiments)
    ▼                                    │
Design question Q ───────▶ Retrieve evidence (top-k with sources)
                                         │
                                         ▼
                              LLM Reasoning (over retrieved context)
                                         │
                                         ▼
                              Evidence Check (every parameter cited?)
                                  │           │
                                  ▼           ╰─── fail: re-query ──▶ [Retrieve]
                          Grounded SDR design
                          (f_s, N_FFT, G_RF, B, P_FA, detector)
                                  │
                                  ╰─── store results ──▶ [Knowledge Base]
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Telco-RAG (Huawei/Yale) | Related | Open-source RAG framework tailored to 3GPP standards — the retrieval layer this methodology needs, in the telecom-standards domain rather than SDR design. |
| SigMF-described measurement archives | Conceptual | Standard metadata makes measured datasets searchable and interpretable. |
| GR4 reflection metadata | Conceptual | Machine-readable block definitions (parameters, ports, constraints) could be indexed as authoritative design knowledge. |

### Project Layout

```
rag-grounded-sdr/
├── README.md
├── corpus/
│   ├── hardware/
│   ├── gnuradio_docs/
│   ├── papers/
│   ├── standards/
│   ├── textbooks_notes/
│   ├── experiments/
│   └── datasets/
├── corpus_manifest.yaml
├── index/
│   ├── chunking.yaml
│   └── embeddings.faiss
├── retrieval/
│   ├── retriever.py
│   └── eval_questions.jsonl
├── designs/
│   └── fm_sensing_v2/
│       ├── design.md
│       ├── parameters.yaml
│       └── evidence_map.json
└── reports/
    └── retrieval_eval.md
```

### Tooling

| Tool | Need | Examples | Purpose |
|---|---|---|---|
| Knowledge base + retrieval | Required | Document store with vector or keyword index (FAISS, Elasticsearch) | The Golden Reference defines the method as LLM + SDR knowledge base + retrieval |
| Retrieval framework | Compatible | LangChain, LlamaIndex; Telco-RAG | Chunking, retrieval and reranking |
| Data management | Compatible | DVC or Git LFS; SigMF for datasets | Version the corpus and measured datasets |
| Documentation | Compatible | Corpus manifest, evidence maps | Parameter traceability |
| Evaluation | Optional | Question sets with expected sources | Measure retrieval quality |

---

## 5. Methodology 3 — Agentic SDR Design

**Alias:** Agentic SDR Design · **Role:** Autonomous workflow orchestrator · **Level:** Design-time

### Overview

- **Type:** Workflow-orchestration methodology (tool-using agent)
- **Abstraction:** Process level — orchestrates other methods as tools
- **Domain:** Iterative SDR design with tool APIs (GNU Radio, Python/MATLAB, SDR hardware)
- **Origin:** Golden Reference: the transition from "LLM assistant" to "engineering agent" [GR]. Closest external evidence: a GRCon 2025 study using LLMs through MCP as high-level controllers of GNU Radio [1], and open-source agents that build, run and verify GNU Radio receivers [8].
- **Purpose:** Automate the design iteration: interpret requirements, retrieve literature, choose an architecture, generate a flowgraph, simulate, inspect PSD/SNR/BER, diagnose failures, modify parameters, repeat. [GR]
- **Scope:** Requirements through testing, carried out through tools. [GR]

### Process

- **Lifecycle:** Agent loop over tools: literature search → hardware datasheets → Python/MATLAB → GNU Radio → SDR hardware → measurement/analysis. [GR]
- **Planning:** The agent plans steps from requirements; humans set goals, budgets and permissions. [GR] [INT]
- **Iteration:** Explicit and automated — an identified failure drives parameter changes and re-execution. [GR]
- **Experimentation:** Built in — simulations and experiments are agent actions. [GR]
- **Validation:** Automated metric inspection (PSD, SNR, BER) inside the loop; a final human sign-off is advisable. [GR] [INT]
- **Documentation:** Plans, tool-call traces, intermediate flowgraphs and decision logs are needed to audit what the agent did. [INT]
- **Automation:** Highest among the design-time methodologies. [INT]
- **Reproducibility:** At risk from nondeterministic planning; requires recording tool calls, versions and seeds. [INT]
- **Deployment:** Produces validated candidates. Deployment stays a human decision. [GR]
- **Governance:** Tool permissions (especially for transmit-capable hardware), stopping criteria and cost budgets. [INT]

### Philosophy & Principles

AI as an engineering agent that acts, observes and corrects, rather than a text generator. [GR]

1. Check every claim by running something [INT]
2. Let tools, not free text, carry domain rules such as validated connections and typed parameters [9]
3. Humans own goals, guardrails and final acceptance [INT]

### Inputs & Outputs

- **Inputs:** SDR requirements and tool access: literature search, hardware datasheets, Python/MATLAB, GNU Radio, SDR hardware, measurement/analysis. [GR]
- **Outputs:** Flowgraphs, PSD/SNR/BER metrics, a decision log and a candidate design. [GR] [INT]
- **Feedback:** Failure → modify parameters → repeat the experiment. [GR]
- **Key Decisions:** Failure identified? Budget left? Escalate to a human? [GR] [INT]

### Fit & Limitations

- **Best fit:** Receiver prototyping in simulation, flowgraph debugging, and parameter exploration where each step can be checked. [INT]
- **Not a fit:** Tasks without machine-checkable success criteria, unsupervised transmit experiments, and hard real-time control. [INT]
- **Strengths:** Closes the loop between generation and evidence, and scales exploration. [INT]
- **Limitations:** Errors compound across steps; cost; dependence on tool quality; harder auditing. [INT]
- **Constraints:** LLMs are computationally expensive and ill-suited to latency-sensitive or embedded environments [2]; the agent stays at the design/orchestration level. [GR]

### Workflow Diagram

```
SDR Requirements (goal + acceptance criteria)
    │
    ▼
Interpret Requirements (agent plans the work)
    │
    ├───── Knowledge Tools (literature search, datasheets)
    ▼
Retrieve Literature (papers, datasheets)
    │
    ▼
Determine Architecture (signal-processing chain)
    │
    ▼
Generate Flowgraph (GNU Radio via tools)  ◄──── Modify Parameters (then repeat)
    │                                                    ▲
    ▼                                                    │ yes
Run Simulation (or hardware experiment)                  │
    │            ╌╌╌╌ Execution Tools (Python/MATLAB, GNU Radio, SDR)
    ▼                                                    │
Inspect Metrics (PSD, SNR, BER)                          │
    │                                                    │
    ▼                                                    │
Failure Identified? ─────────────────────────────────────╯
    │ no: criteria met
    ▼
Candidate Design (for human review)
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Marconi (yoelbassin/gr-mcp) | Strong | Agent plugin that surveys spectrum, builds a receiver, debugs and runs closed-loop experiments. v1.0 is simulation-only. |
| GRCon 2025: LLMs as GNU Radio controllers | Explicit | Used LLMs through MCP to dynamically manage GNU Radio signal-processing chains. Concluded LLMs are strong at orchestration but sample-inefficient. |
| gnuradio-mcp | Related | 80+ MCP tools to build, validate, run and export flowgraphs. Runs flowgraphs in Docker containers. |
| WirelessAgent (HKUST) | Conceptual | LLM agents with perception, memory, planning and action modules for network slicing. |

### Project Layout

```
agentic-sdr/
├── README.md
├── agent/
│   ├── agent_config.yaml
│   ├── system_prompt.md
│   └── tools/
│       ├── literature_search.py
│       ├── datasheet_lookup.py
│       ├── gnuradio_builder.py
│       ├── simulate.py
│       ├── sdr_control.py
│       └── measure_psd_snr_ber.py
├── specs/
│   └── requirements.yaml
├── guardrails/
│   ├── tool_permissions.yaml
│   └── tx_limits.yaml
├── runs/
│   └── run_0007/
│       ├── plan.md
│       ├── trace.jsonl
│       ├── flowgraph_iter03.grc
│       ├── metrics_iter03.json
│       └── decision_log.md
├── artifacts/
│   └── captures/
│       ├── fm_iter03.sigmf-data
│       └── fm_iter03.sigmf-meta
└── review/
    └── human_signoff.md
```

---

## 6. Methodology 4 — AI-in-the-Loop Optimization

**Alias:** AI-in-the-Loop Optimization · **Role:** Design optimizer · **Level:** Design-time

### Overview

- **Type:** Parameter-optimization methodology
- **Abstraction:** Parameter level; architecture held fixed
- **Domain:** Tuning detectors, spectrum sensing and receiver chains
- **Origin:** Golden Reference formulation θ* = arg max J(θ) [GR]. Precedents: genetic-algorithm cognitive engines that evolve SDR parameter "chromosomes" [20], design-of-experiments optimisation of SDR configurations [18], and Bayesian optimisation of radio parameters [17].
- **Purpose:** Search the configurable parameter vector θ for the best trade-off defined by an objective J(θ). [GR]
- **Scope:** Parameter tuning of a mostly known architecture. [GR]

### Process

- **Lifecycle:** Define θ and J → propose θ_k → evaluate → update → converge to θ*. [GR] [INT]
- **Planning:** Explicit and upfront: parameter ranges, objective weights w₁…w₄, constraints and an evaluation budget. [GR] [INT]
- **Iteration:** Numerical loop bounded by a budget or a convergence test. [INT]
- **Experimentation:** Each evaluation of θ is an experiment, run in simulation or on hardware. [INT]
- **Validation:** Objective and constraint satisfaction; re-test θ* on held-out scenarios so it does not overfit one condition. [INT]
- **Documentation:** Search space, objective definition and weights, full trial history, Pareto front and chosen θ*. [INT]
- **Collaboration:** Engineers negotiate the weights: J(θ) = w₁P_D − w₂P_FA − w₃T_latency − w₄C_CPU encodes priorities. [GR] [INT]
- **Automation:** High inside the loop; objective design stays human. [INT]
- **Reproducibility:** Good with fixed seeds and versioned evaluators; noisy hardware evaluations reduce it. [INT]

### Philosophy & Principles

When the structure is known, let an optimizer search the parameter space. [GR]

1. Make the objective explicit [GR]
2. Respect constraints during the search [INT]
3. Prefer sample-efficient methods when evaluations are expensive [INT]

### Inputs & Outputs

- **Inputs:** Known architecture; θ = [f_s, N_FFT, N_Welch, G, P_FA, λ]; objective J(θ); an evaluator. [GR]
- **Outputs:** θ*, trial history and a trade-off analysis. [GR] [INT]
- **Feedback:** The optimizer updates its model or population after each evaluation. [INT]
- **Key Decision:** Converged, or budget exhausted? [INT]
- **Techniques:** Bayesian optimization, genetic algorithms, reinforcement learning, Gaussian processes, evolutionary optimization, AutoML [GR]; libraries such as Optuna [16].

### Fit & Limitations

- **Best fit:** Detector or receiver tuning with measurable P_D, P_FA, latency and CPU cost. [GR]
- **Not a fit:** Architecture still undecided, or objectives that cannot be quantified. [INT]
- **Strengths:** Systematic, measurable and reproducible trade-offs. [INT]
- **Limitations:** Objective misspecification, evaluator fidelity, and no ability to fix a wrong architecture. [INT]

### Workflow Diagram

```
Known Architecture + θ (f_s, N_FFT, N_Welch, G, P_FA, λ)
    │
    ▼
Define J(θ) (weights + constraints)
    │
    ▼                     Trial History
Propose θ_k ◄──────────── (surrogate or population)
    │
    ▼                     Evaluator
Evaluate θ_k ╌╌╌╌╌╌╌╌╌╌▶ (Simulation M6 / HIL M7)
    │
    ▼
Measure → J (P_D, P_FA, latency, CPU)
    │
    ▼
Converged? ──── no: next θ ──▶ [back to Propose]
    │ yes
    ▼
θ* = arg max J(θ) (plus trade-off record)
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Virginia Tech CWT genetic-algorithm cognitive engine | Strong | Encodes radio parameters as chromosome genes and lets a GA find the optimal set. Reported hardware and simulation results. |
| Design of experiments for SDR configurations | Related | Classical statistical optimisation of SDR parameters — a baseline for AI optimizers. |
| Bayesian optimisation for radio resource management | Related | GP Bayesian optimisation tunes radio parameters online while limiting performance drops. |

### Project Layout

```
sdr-param-optimization/
├── README.md
├── search_space.yaml
├── objective.py
├── constraints.yaml
├── configs/
│   └── baseline_params.yaml
├── evaluator/
│   ├── sim_evaluator.py
│   └── hil_evaluator.py
├── studies/
│   └── study_2026-09_cfar/
│       ├── trials.db
│       ├── pareto_front.csv
│       └── best_params.yaml
├── notebooks/
│   └── sensitivity_analysis.ipynb
└── reports/
    └── optimization_report.md
```

---

## 7. Methodology 5 — ML-Assisted DSP Design

**Alias:** ML-Assisted Signal Processing · **Role:** DSP component · **Level:** Runtime

### Overview

- **Type:** Component-level technique (AI inside the SDR)
- **Abstraction:** Block level inside the signal chain
- **Domain:** Automatic modulation classification (AMC), occupancy, signal detection, channel estimation
- **Origin:** The traditional meaning of AI + SDR according to the Golden Reference [GR]. Foundational work: deep learning for the physical layer [22] and over-the-air radio signal classification [21].
- **Purpose:** Replace or augment a specific DSP block with a learned model: IQ → ML model → decision. [GR]
- **Scope:** One block's function and its data pipeline. [GR]

### Process

- **Lifecycle:** Data → representation → training → evaluation against a baseline → integration → runtime inference. [INT]
- **Planning:** Dataset design (signal classes, SNR range, impairments) is the main plan. [INT]
- **Iteration:** The usual ML loop over data, representation and model. [INT]
- **Experimentation:** Heavy — training runs, SNR sweeps, impairment studies, and synthetic vs. over-the-air comparisons [21].
- **Validation:** Compare with a conventional baseline under matched conditions — O'Shea et al. compared against higher-order moments with boosted trees [21] — and check the latency budget. [INT]
- **Deployment:** The model runs inside the SDR chain under real-time and compute constraints. [GR] [INT]

### Philosophy & Principles

AI inside SDR ≠ AI designing SDR — although a complete methodology can contain both. [GR]

1. Replace a block only when it beats the baseline [INT]
2. Test on the data distribution met on air [INT]
3. Respect the latency budget of the chain [INT]

### Inputs & Outputs

- **Inputs:** IQ, PSD, spectrogram or x[k]; labelled datasets. [GR]
- **Outputs:** Decisions: modulation class, occupancy, detections, channel estimates. [GR]
- **Key Decision:** Does the model beat the baseline within the latency budget? [INT]

### Fit & Limitations

- **Best fit:** AMC, wideband detection and occupancy estimation when representative data is available. [INT]
- **Not a fit:** Tasks with closed-form optimal solutions and tight compute budgets, or no representative data. [INT]
- **Strengths:** Learns features directly from IQ and copes with complex impairments [21].
- **Limitations:** Distribution shift from synthetic to over-the-air data [21]; edge compute cost; explainability. [INT]

### Workflow Diagram

```
Target Block & Task (AMC, occupancy, detection, channel est.)
    │
    │                    Datasets (synthetic + OTA IQ)
    ▼                        │
Representation ◄─────────────╯
(IQ, PSD, spectrogram)
    │
    ▼
Train Model (CNN, NN, Transformer)
    │
    ▼                    Conventional Baseline
Evaluate vs Baseline ◄──── (e.g. moments + boosted trees)
    │
    ▼
Beats Baseline? ──── no: revise ──▶ [back to Representation]
    │ yes
    ▼
Integrate as SDR Block (replace or augment)
    │
    ▼
IQ → ML Model → Decision (runtime inference)
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Over-the-air deep learning radio signal classification (O'Shea et al.) | Explicit | IQ → CNN/ResNet → modulation class, compared against higher-order-moments baseline. Studied CFO, symbol rate and multipath. |
| TorchSig | Explicit | Toolkit and datasets (Sig53, WidebandSig53) for training RF classification and detection models. |
| DeepSig RadioML datasets | Explicit | Public labelled IQ datasets widely used to benchmark AMC models. |
| Sionna PHY neural components | Related | Link-level simulator where neural blocks replace conventional PHY blocks. |

### Project Layout

```
ml-dsp-block/
├── README.md
├── model_card.md
├── data/
│   ├── synthetic/
│   ├── ota_captures/
│   └── splits.yaml
├── baselines/
│   └── cumulant_classifier.py
├── models/
│   ├── resnet_amc.py
│   └── checkpoints/
├── training/
│   ├── train.py
│   └── config_amc.yaml
├── evaluation/
│   ├── accuracy_vs_snr.csv
│   └── latency_profile.json
└── integration/
    ├── gr_amc_block.py
    └── amc_rx.grc
```

---

## 8. Methodology 6 — Simulation-in-the-Loop

**Alias:** Simulation/Digital-Twin Assisted Design · **Role:** Model evaluator · **Level:** Evaluation

### Overview

- **Type:** Verification methodology (automated evaluation loop)
- **Abstraction:** Process level — an evaluation stage for any design methodology
- **Domain:** Detector and receiver design; pre-hardware comparison of architectures
- **Origin:** Golden Reference: AI proposal → simulation → metrics → AI evaluation → new proposal [GR]. Tooling examples: Sionna for link-level and system-level simulation and ray tracing [27]; GNU Radio simulations driven by agents [8].
- **Purpose:** Verify AI-generated designs empirically before accepting them, in a controlled and cheap environment. [GR]
- **Scope:** The proposal–simulation–metrics loop. [GR]

### Process

- **Lifecycle:** AI proposal → simulation → metrics → AI evaluation → new proposal. [GR]
- **Planning:** Scenario design: channel models, SNR grid, signal mix, metrics and targets. [INT]
- **Iteration:** Automated loop until targets are met. [GR]
- **Experimentation:** Controlled Monte Carlo experiments and parameter sweeps. [INT]
- **Validation:** Metrics such as P_D and P_FA under AWGN (the Golden Reference example) or richer channel models. [GR]
- **Reproducibility:** Highest of all methodologies when seeds and versions are fixed. [INT]
- **Deployment:** Acts as a gate before hardware. [GR] [INT]

### Philosophy & Principles

Do not accept generated DSP code without empirical verification. [GR]

1. Define metrics and targets before simulating [INT]
2. Fix seeds and versions [INT]
3. State what the model leaves out [INT]

### Inputs & Outputs

- **Inputs:** A candidate design (from an LLM, agent or engineer), scenarios and targets. [GR]
- **Outputs:** Metrics (P_D, P_FA, BER…), a verdict and evidence for or against the design. [GR]
- **Feedback:** Metrics drive the next AI proposal. [GR]
- **Key Decision:** Does the design meet targets in the simulated scenarios? [GR]

### Fit & Limitations

- **Best fit:** Detector design, receiver comparisons and early architecture selection. [GR]
- **Not a fit:** Final evidence for effects models capture poorly, such as front-end non-linearities or site-specific interference. [INT]
- **Strengths:** Cheap, safe, repeatable and parallelisable. [INT]
- **Limitations:** The sim-to-real gap. [GR] [INT]

### Workflow Diagram

```
Candidate Design (from LLM, agent or engineer)
    │
    ▼
AI Proposal (e.g. generate detector)  ◄──── no: new proposal ────╮
    │                                                              │
    ▼                      Simulation Evaluators                   │
Simulation ◄╌╌╌╌╌╌╌╌╌╌╌╌ (Python, MATLAB, GNU Radio, RF sims)   │
(e.g. AWGN channel)                                                │
    │                                                              │
    ▼                                                              │
Metrics (P_D, P_FA, BER)                                          │
    │                                                              │
    ▼                                                              │
AI Evaluation (targets met?) ──────────────────────────────────────╯
    │ yes
    ▼
Simulation-Verified Design (ready for HIL) ╌╌╌▶ [M7: Hardware-in-the-Loop]
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Sionna (NVIDIA) | Related | GPU-accelerated link-level and system-level simulation plus ray tracing. Native ML integration. |
| Marconi simulate-scene and tx-experiment | Strong | Agent registers a simulated device and runs closed-loop experiments in simulation. v1.0 is simulation-only. |
| GrGym simulated environments | Related | RL agents trained against GNU Radio programs in simulated environments before real testbeds. |
| Colosseum as a digital twin | Related | Digital-twin replicas of real environments through channel emulation. Bridges this methodology and HIL. |

### Project Layout

```
sim-in-the-loop/
├── README.md
├── seeds.yaml
├── scenarios/
│   ├── awgn_snr_sweep.yaml
│   └── multipath_fading.yaml
├── channel_models/
│   └── awgn.py
├── candidates/
│   ├── detector_v1.py
│   └── detector_v2.py
├── harness/
│   ├── run_monte_carlo.py
│   └── metrics.py
├── results/
│   └── detector_v2/
│       ├── pd_pfa.csv
│       └── roc.png
└── evaluation_log.md
```

---

## 9. Methodology 7 — Hardware-in-the-Loop (HIL)

**Alias:** Hardware-in-the-Loop AI Design · **Role:** Experimental optimizer · **Level:** Evaluation

### Overview

- **Type:** Experimental evaluation and optimization methodology
- **Abstraction:** Process level — the physical evaluation stage
- **Domain:** Final tuning and validation on target hardware
- **Origin:** Golden Reference: AI → SDR configuration → USRP/HackRF/RTL-SDR → IQ → metrics → AI [GR]. External practice: Colosseum supports AI/ML experimentation with HIL at scale [29]; GrGym connects RL agents to GNU Radio programs on real testbeds [32].
- **Purpose:** Optimize and validate against the physical RF chain, including effects that simulation may omit. [GR]
- **Scope:** Configuration → device → IQ → metrics → AI. [GR]

### Process

- **Lifecycle:** Configure, acquire, measure, adjust — e.g. G = 20 dB → SNR = 7 dB, then G = 25 dB → SNR = 11 dB — until an objective or constraint is satisfied. [GR]
- **Planning:** Bench setup, calibration, safe parameter limits and scheduling of shared equipment. [INT]
- **Iteration:** Automated measurement loop, slower than simulation. [GR] [INT]
- **Validation:** Measured metrics with repeated trials and calibration references. [INT]
- **Automation:** Device control through driver APIs such as UHD [35] and SoapySDR, which GR4 integrates [4].
- **Reproducibility:** Lower than simulation because the RF environment varies; improved by cabled setups, channel emulators and complete metadata. [INT]

### Philosophy & Principles

Optimize against the physical RF chain, not only its model. [GR]

1. Enforce safe limits before the AI can act [INT]
2. Record every capture with its configuration [INT]
3. Repeat measurements to separate effect from noise [INT]

### Inputs & Outputs

- **Inputs:** An objective or constraint, an SDR device (USRP, HackRF, RTL-SDR) and an initial configuration. [GR]
- **Outputs:** A hardware-validated configuration, IQ captures and measured metrics. [GR] [INT]
- **Feedback:** Metrics feed the AI, which proposes the next configuration. [GR]
- **Key Decision:** Objective or constraint satisfied? Safe limits respected? [GR] [INT]

### Fit & Limitations

- **Best fit:** Gain and bandwidth tuning, front-end-specific calibration and final acceptance. [INT]
- **Not a fit:** Early exploration of many architectures (too slow), and unsupervised transmission in licensed bands. [INT]
- **Strengths:** Captures real front-end and environment effects. [GR]
- **Limitations:** Slow, environment-dependent and equipment-bound. [INT]

### Workflow Diagram

```
Objective / Constraint (e.g. SNR ≥ target)
    │
    ▼
AI Selects Configuration (e.g. G = 20 dB)  ◄──── no: e.g. G → 25 dB ────╮
    │                                                                       │
    ▼                                                                       │
Apply to SDR (driver API) ╌╌╌▶ Physical RF Chain (USRP, HackRF, RTL-SDR) │
                                       │                                    │
                                       ▼                                    │
                    Acquire IQ (timed capture)                               │
                    │              │                                         │
                    ▼              ╰──▶ IQ Recordings (SigMF)              │
                Compute Metrics (SNR, P_D, P_FA)                            │
                    │                                                       │
                    ▼                                                       │
                Satisfied? ─────────────────────────────────────────────────╯
                    │ yes
                    ▼
                Hardware-Validated Configuration (with IQ evidence)
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| Colosseum (Northeastern University, PAWR) | Explicit | World's largest wireless network emulator with HIL: 256 USRP X310 radios, with AI/ML support. Channels are emulated by FPGA filters. |
| GrGym (TU Berlin) | Explicit | Exposes any GNU Radio program's state and control knobs to an RL agent, in simulation or on real SDR testbeds. |
| VT CWT cognitive engine on hardware | Strong | GA-driven radio parameter search with experimental results on a hardware platform in 2004. |
| Over-the-air evaluation of ML classifiers | Related | Over-the-air measurements with software radios used to evaluate models trained on synthetic data. |

### Project Layout

```
hil-sdr/
├── README.md
├── lab_notebook.md
├── bench/
│   ├── bench_setup.md
│   ├── device_inventory.yaml
│   └── calibration/
│       └── noise_floor_2026-09-20.json
├── control/
│   ├── sdr_config.py
│   └── safe_limits.yaml
├── captures/
│   └── run_0042/
│       ├── cap_g20.sigmf-data
│       ├── cap_g20.sigmf-meta
│       ├── cap_g25.sigmf-data
│       └── cap_g25.sigmf-meta
├── metrics/
│   └── snr_estimator.py
├── optimizer/
│   └── hil_loop.py
└── results/
    └── run_0042_summary.csv
```

---

## 10. Methodology 8 — Closed-Loop Autonomous SDR

**Alias:** Closed-Loop / Self-Adaptive SDR · **Role:** Runtime controller · **Level:** Runtime

### Overview

- **Type:** Runtime control methodology (cognitive-radio pattern)
- **Abstraction:** System operation level
- **Domain:** Dynamic spectrum access, spectrum sharing, adaptive links
- **Origin:** Golden Reference: Sense → Analyze → Decide → Reconfigure → Sense, beginning to resemble a cognitive radio [GR]. Foundations: cognitive radio as introduced by Mitola and Maguire [38] and framed by Haykin as an SDR-based system that senses, learns and adapts [37].
- **Purpose:** Continuously adapt {f_c, B, G, waveform, detector} to the observed RF environment. [GR]
- **Scope:** Runtime sensing, reasoning and reconfiguration. [GR]

### Process

- **Lifecycle:** Continuous cycle: Sense → Analyze → Decide → Reconfigure → Sense. [GR]
- **Planning:** Define the action space, policy envelope and fallback behaviour before deployment. [INT]
- **Iteration:** Continuous during operation. [GR]
- **Experimentation:** Online exploration must be bounded; controllers are usually trained and tested in emulation first, e.g. OpenRAN Gym xApps on Colosseum [39].
- **Validation:** Runtime KPIs plus pre-deployment testing in emulators and replayed scenarios. [INT]
- **Automation:** Fully automated at runtime. [GR]
- **Reproducibility:** Hard, because environments do not repeat; recorded telemetry and replay help. [INT]
- **Governance:** The strongest need of all: regulatory compliance, human override and safe fallback. [INT]

### Philosophy & Principles

The radio reasons about its environment and reconfigures itself. [GR]

1. Bound autonomy with a policy envelope [INT]
2. Log every decision [INT]
3. Fall back safely when sensing is uncertain [INT]

### Inputs & Outputs

- **Inputs:** The RF environment (spectrum occupancy) and policy constraints. [GR] [INT]
- **Outputs:** A reconfigured SDR: f_c, B, G, waveform, detector. [GR]
- **Feedback:** Each reconfiguration changes what is sensed next. [GR]

### Fit & Limitations

- **Best fit:** Spectrum sharing and dynamic access — the setting of DARPA's Spectrum Collaboration Challenge [41].
- **Not a fit:** Static links, and certification contexts that forbid runtime changes. [INT]
- **Strengths:** Adapts without human delay. [INT]
- **Limitations:** Verification is difficult; emergent behaviour among radios [37]; interference risk. [INT]

### Workflow Diagram

```
RF Environment ──observed──▶ Sense (spectrum occupancy)
(occupancy, interference)         │
                                  ▼
                            Analyze (environment state)
                                  │
    Policy Envelope ──constrains──▼
    (limits, human override)  Decide (AI controller)
                                  │
                                  ▼
                            Reconfigure ({f_c, B, G, waveform, detector})
                                  │         │
                                  │         ╰──record──▶ Decision Telemetry
                                  │                      (for replay & audit)
                                  ╰── sense again ──▶ [back to Sense]
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| DARPA Spectrum Collaboration Challenge (2016–2019) | Explicit | Teams built radio networks that autonomously decide, moment to moment, how to share spectrum. |
| OpenRAN Gym xApps | Explicit | AI/ML closed-loop control of a softwarized cellular RAN through the O-RAN near-real-time RIC, tested on Colosseum. |
| VT CWT cognitive engine | Explicit | Defines "knobs and meters" between an AI engine and an SDR and adapts the radio with a genetic algorithm using GNU Radio. |
| Hebbian Cellular Automata (GRCon 2025) | Conceptual | Proposed as a lighter learning framework for control and latency-sensitive tasks after LLMs proved ill-suited to low-level interaction. |

### Project Layout

```
closed-loop-sdr/
├── README.md
├── sensing/
│   ├── psd_monitor.grc
│   └── occupancy_detector.py
├── analysis/
│   └── environment_state.py
├── controller/
│   ├── policy.py
│   ├── action_space.yaml
│   └── model/
├── reconfiguration/
│   └── apply_config.py
├── policy_envelope/
│   ├── regulatory_limits.yaml
│   └── human_override.md
├── telemetry/
│   └── decisions_2026-09-25.jsonl
└── tests/
    └── replay_scenarios/
        └── crowded_band_evening.yaml
```

---

## 11. Integrated AI-Assisted SDR Design Framework (∑)

**Alias:** Golden Reference: "A methodology I would use for a research project" · **Role:** Composite · **Level:** Integrated

### Overview

- **Type:** Integrated lifecycle methodology (hybrid)
- **Abstraction:** Lifecycle level; composes methodologies 1–4, 6, 7 and a HITL gate
- **Domain:** Research projects and lab-to-field SDR development
- **Origin:** Proposed in the Golden Reference as the combination of the previous approaches [GR]. Timeliness argument: GR4 exposes block metadata and reflection that support automated construction and validation [3] [4], and GNU Radio frames GR4 around AI-enabled SDR development [5].
- **Purpose:** A human-supervised, evidence-grounded, closed-loop engineering methodology in which AI synthesizes, implements, evaluates and iteratively improves SDR architectures using simulation and physical RF measurements as feedback. [GR]
- **Scope:** Ten steps from requirements to human approval and deployment, with an iterative Design → Execute → Measure → Evaluate → Redesign loop. [GR]

### The Ten Steps

| Step | Name | Description |
|---|---|---|
| 1 | Requirements | Measurable acceptance criteria (band, hardware, P_D/P_FA, latency) |
| 2 | Knowledge Retrieval / RAG | Ground the design in evidence — retrieve manuals, standards, papers, previous experiments (Methodology 2) |
| 3 | AI Architecture Synthesis | Propose the DSP architecture via copilot or agent (Methodologies 1, 3) |
| 4 | Code/Flowgraph Generation | Make the architecture executable — GNU Radio, Python, C++ |
| 5 | Static + Constraint Validation | Check types, connections, parameter ranges against hardware limits and regulatory constraints before anything runs |
| 6 | Simulation-in-the-Loop | First empirical evidence — run scenario simulations (Methodology 6) |
| 7 | Hardware-in-the-Loop | Physical RF evidence — configure SDR, acquire IQ, measure (Methodology 7) |
| 8 | Metric Evaluation | Judge the evidence against requirements; note sim-versus-hardware gaps |
| 9 | AI Optimization | Tune θ, or request an architectural change when tuning cannot meet requirements (Methodology 4) |
| 10 | Human Approval / Deployment | HITL gate — review evidence and sign off, or reject |

### Process

- **Lifecycle:** Steps 1–10 [GR], with the iteration loop: Design → Execute → Measure → Evaluate → Redesign. [GR]
- **Planning:** Requirements and acceptance criteria come first and are traced through every step. [GR] [INT]
- **Experimentation:** Two evaluation stages: simulation, then physical hardware. [GR]
- **Validation:** Layered: static and constraint checks, simulation, HIL, metric evaluation, human approval. [GR]
- **Documentation:** A traceability matrix linking requirements, evidence, design decisions, tests and approvals. [INT]
- **Collaboration:** Humans own requirements and approval; AI owns synthesis, execution and optimization. [GR] [INT]
- **Automation:** Steps 2–9 can be automated; step 10 is deliberately human. [GR] [INT]
- **Deployment:** Only after human approval. [GR]
- **Governance:** A formal HITL gate at the end and constraint validation before any execution. [GR]

### Philosophy & Principles

Evidence-grounded, human-supervised, closed-loop. [GR]

1. Ground before generating [GR]
2. Validate statically before executing [GR]
3. Simulate before hardware; measure before approving [GR]
4. Humans approve deployment [GR]

### Fit & Limitations

- **Best fit:** Thesis or lab programmes that combine LLMs/RAG/agents, DSP, GNU Radio and SDR hardware. [GR]
- **Not a fit:** Quick one-off prototypes where the infrastructure cost outweighs the benefit. [INT]
- **Strengths:** Covers the gaps the individual methodologies leave open at design time. [INT]
- **Limitations:** Heavy setup and long iterations when HIL is involved; it does not by itself cover runtime adaptation (M8) or ML blocks (M5). [INT]

### Workflow Diagram

```
1. Requirements ──▶ 2. Knowledge/RAG ──▶ 3. AI Architecture ──▶ 4. Code/Flowgraph ──▶ 5. Static Validation
                                                                                            │
                                               ╭── fail: regenerate ◄───────────────────────╯
                                               │                                            │ pass
                                               │    ╭── redesign ◄── 9. AI Optimization ◄──╯
                                               │    │                        │
                                               │    ▼                        │
                                               │  3. AI Architecture    8. Metric Evaluation
                                               │                        ▲        ▲
                                               │                        │        │
                                               │                  7. HIL ◄── 6. Simulation
                                               │
                                    10. Human Approval ──── rejected: revise ──▶ [1. Requirements]
                                               │ approved
                                               ▼
                                         Deployed SDR
```

### Example Projects

| Project | Evidence | Description |
|---|---|---|
| GR4 reflection and metadata | Conceptual | Enabler for steps 4–5: block parameters, ports and constraints can be extracted programmatically. GR4 reached RC1 in March 2026. |
| Marconi | Related | Covers steps 4–6 (build, validate, run, verify in simulation). No hardware stage or formal approval step yet. |
| Colosseum as a prototyping stage | Conceptual | Repeatable emulation before experimenting "in the wild" — the same simulate-emulate-field staging as steps 6–7. |
| OpenRAN Gym workflow | Conceptual | End-to-end design, data collection and testing workflow for AI controllers, from emulation to over-the-air deployment. |

### Project Layout

```
ai-assisted-sdr-framework/
├── README.md
├── traceability_matrix.csv
├── 01_requirements/
│   └── requirements.md
├── 02_knowledge/
│   ├── corpus_manifest.yaml
│   └── index/
├── 03_architecture/
│   └── architecture_v4.md
├── 04_generation/
│   ├── flowgraphs/
│   └── src/
├── 05_validation/
│   ├── constraints.yaml
│   └── static_check_report.md
├── 06_simulation/
│   ├── scenarios/
│   └── results/
├── 07_hil/
│   ├── safe_limits.yaml
│   └── captures/
├── 08_metrics/
│   └── evaluation_report.md
├── 09_optimization/
│   └── studies/
├── 10_approval/
│   ├── approval_record.md
│   └── deployment_checklist.md
└── iterations/
    └── iter_004/
        └── iteration_log.md
```

---

## 12. Relationship Map

### Relationship Types

| Type | Meaning |
|---|---|
| **Extends** | Overlapping scope, adds capability |
| **Orchestrates** | Upstream controller that invokes the other |
| **Uses as evaluator/resource** | Complementary |
| **Feeds** | Upstream → downstream |
| **Overlaps** | Overlapping responsibilities |
| **Orthogonal** | Different axis |
| **Integrates as a step** | Lifecycle integration |

### All Relationships

| From | To | Type | Source | Description |
|---|---|---|---|---|
| M2 | M1 | Extends | GR | Adds a knowledge base and retrieval to LLM generation; the Golden Reference calls it considerably stronger for engineering work. |
| M3 | M1 | Orchestrates | GR | The agent uses LLM generation as one of its steps. |
| M3 | M2 | Orchestrates | GR | Literature search and hardware datasheets are agent tools. |
| M3 | M6 | Orchestrates | GR | "Run a simulation" is an agent step. |
| M3 | M7 | Orchestrates | GR | SDR hardware is in the agent's tool chain. |
| M4 | M6 | Uses | INT | Simulation as a cheap evaluator of J(θ). |
| M4 | M7 | Uses | INT | Hardware as a faithful but slow evaluator of J(θ). |
| M6 | M7 | Feeds | INT | Simulation-verified designs move to hardware, where effects simulation omits are tested. |
| M7 | M8 | Feeds | INT | A HIL loop that runs continuously in the field approaches closed-loop operation. |
| M4 | M8 | Overlaps | INT | Same knobs: design-time search versus runtime adaptation. |
| M3 | M4 | Overlaps | INT | Both change parameters; ownership must be explicit. |
| M5 | M3 | Orthogonal | GR | AI inside SDR ≠ AI designing SDR, although a complete methodology can contain both. |
| M8 | M5 | Uses | INT | Sensing and analysis stages may be ML blocks. |
| M5 | M6 | Uses | INT | Synthetic datasets and SNR sweeps come from simulation. |
| M1 | M6 | Feeds | GR | Generated code should be verified empirically before acceptance. |
| ∑ | M2 | Integrates | GR | Step 2: Knowledge Retrieval / RAG. |
| ∑ | M3 | Integrates | INT | Steps 3–4 by an agent. |
| ∑ | M1 | Integrates | INT | Steps 3–4 by a copilot. |
| ∑ | M6 | Integrates | GR | Step 6: Simulation-in-the-Loop. |
| ∑ | M7 | Integrates | GR | Step 7: Hardware-in-the-Loop. |
| ∑ | M4 | Integrates | GR | Step 9: AI Optimization. |

---

## 13. Hybrid Workflows

### Hybrid 1: Evidence-Grounded Generation with Staged Verification

**Chain:** M2 (Retrieve evidence) → M1 (Generate architecture and code) → M6 (Verify in simulation) → M7 (Confirm on hardware)

- **Rationale:** Each stage removes a different failure mode: retrieval reduces unsupported parameters, simulation catches functional errors cheaply, and hardware exposes effects that simulation omits. [GR]
- **Overlap:** Simulation and HIL judge the same design against the same metrics. Agree beforehand which thresholds apply at each stage. [INT]
- **Complement:** RAG supplies parameters with provenance, which makes simulation failures easier to diagnose: was the evidence wrong, or the implementation? [INT]
- **Conflict:** Settings that are optimal in simulation may not transfer. When the two disagree, hardware evidence should win, and the discrepancy should be written back into the knowledge base. [INT]

### Hybrid 2: Agent with a Delegated Optimizer

**Chain:** M3 (Agent designs the architecture) → M4 (Optimizer tunes θ) → M6 (Evaluate J(θ) in simulation) → M7 (Confirm θ* on hardware)

- **Rationale:** The Golden Reference gives architecture reasoning to agents and numerical search to optimizers; combining them uses each where it is strongest. [GR] [INT]
- **Overlap:** Both the agent and the optimizer can change parameters. Give the agent ownership of topology and discrete choices, and the optimizer ownership of continuous θ. [INT]
- **Complement:** The agent can recognise when a failure needs a structural change rather than more tuning. [INT]
- **Conflict:** Nested loops multiply evaluation cost, and an agent that edits the objective during a study invalidates earlier trials. [INT]

### Hybrid 3: An ML Block Engineered Through the Design Loop

**Chain:** M5 (Specify the block and its baseline) → M6 (Synthetic data and SNR sweeps) → M7 (Over-the-air captures and tests)

- **Rationale:** An ML block is still a DSP component and needs the same empirical verification as any design. O'Shea et al. evaluated simulated impairments and over-the-air measurements with software radios [21].
- **Overlap:** Simulation produces both training data and evaluation data; keep them separate to avoid leakage. [INT]
- **Complement:** Synthetic data gives labels at scale; over-the-air data exposes hardware and channel effects [21].
- **Conflict:** Models trained on synthetic data can lose accuracy over the air, so plan for fine-tuning or retraining [21]. [INT]

### Hybrid 4: From Approved Design to Adaptive Operation

**Chain:** ∑ (Design, test and approve) → M5 (ML blocks for sensing) → M7 (HIL test of the controller) → M8 (Closed-loop deployment)

- **Rationale:** The integrated framework ends at human-approved deployment and closed-loop operation begins there [GR]. Linking them lets the approval step define what the runtime controller may do. [INT]
- **Overlap:** Both apply a measure → evaluate → change pattern, at different timescales and with different authority. [INT]
- **Complement:** Design-time evidence bounds runtime autonomy, and runtime telemetry feeds the next design iteration. [INT]
- **Conflict:** Runtime reconfiguration can move the radio outside the configurations a human approved. An explicit policy envelope, approved at step 10, resolves this. [INT]

---

## 14. Evidence Notes & Discrepancies

### Discrepancies

**Two names for most methodologies.** The Golden Reference's taxonomy table and its section headings use different names for six entries (e.g. table "ML-Assisted Signal Processing" vs heading "ML-Assisted DSP Design"; "Simulation/Digital-Twin Assisted Design" vs "Simulation-in-the-Loop"; "Closed-Loop / Self-Adaptive SDR" vs "Closed-Loop Autonomous SDR"). The atlas uses the section heading as the primary name and shows the table name as the taxonomy name. Nothing was renamed. [GR]

**The integrated framework has no short name.** The Golden Reference introduces it as "a methodology I would use for a research project" and "a rigorous AI-Assisted SDR Design framework." The label "Integrated AI-Assisted SDR Design framework" is descriptive, not a new definition. [GR]

### Ambiguities

**One GRCon 2025 paper or two?** The Golden Reference cites work that fine-tuned LLMs for GNU Radio flowgraph construction and, separately, an experiment using LLMs as high-level controllers. Research found both in a single GRCon 2025 contribution, "Powering Cognitive Radios with LLMs" [1] [2]. The reference may point to other work not located; this is unresolved.

**Where does a "digital twin" belong?** The Golden Reference's table mentions digital twins under methodology 6, and its HIL section uses physical RF measurements. Colosseum calls itself a digital twin yet uses real radios with emulated channels [30] [29]. The atlas keeps the Golden Reference's split (6 = simulated signals, 7 = physical RF chain) and marks Colosseum as bridging both.

**Reinforcement learning spans two methodologies.** The Golden Reference lists RL under AI-in-the-Loop Optimization. External RL-on-SDR work such as GrGym often adapts parameters during operation [32], which overlaps with Closed-Loop Autonomous SDR. This is overlap, not contradiction.

### Caveats

**"Unsuitable for sample-by-sample processing" — nuance.** The Golden Reference says LLMs are unsuitable for sample-by-sample, latency-critical processing. The GRCon paper says LLMs are powerful for orchestration but sample-inefficient, ill-suited to low-level interactions and edge devices, and computationally expensive for latency-sensitive settings [1] [2]. The substance agrees; note that "sample-inefficient" there refers to learning efficiency, not per-sample DSP.

**"GR4 as a foundation for AI-enabled SDR workflows."** GNU Radio's own framing describes a future "where AI-enabled SDR development feels effortless" [5]. This matches the Golden Reference in spirit but is aspirational language; no specific AI tooling inside GR4 core was verified.

**What the evidence levels mean here.** No external project labels itself with the Golden Reference's taxonomy. Evidence levels rate how directly a project documents the practice a methodology describes, not use of the label. "Explicitly documented" never means "adopted this taxonomy."

**How sources were checked.** Every link was found through web search on 26 September 2026 and its page content was read as returned by the search tool; the Marconi repository was opened directly. arXiv links point to the abstract page of the identifier shown in the source. Tool names in the tools tables without a citation (e.g. MLflow, DVC, BoTorch) are listed as commonly compatible options, not as verified claims about the methodology.

### Confirmations

**GR4 metadata and reflection — consistent.** External sources confirm GR4's reflection system exposes block metadata, ports and parameters at runtime [3], and that metadata can drive validation, UI generation and automated system construction [4]. As of the sources consulted, GR4 was at release-candidate stage (RC1, March 2026); later releases were not checked.

**Cognitive radio lineage — consistent.** The Golden Reference says closed-loop SDR begins to resemble a cognitive radio. Foundational definitions agree: a cognitive radio is built on an SDR, aware of its environment and adapts by learning [37] [38].

---

## 15. Sources & References

All sources accessed 2026-09-26.

| # | Title | Publication | Type | URL |
|---|---|---|---|---|
| 1 | Powering Cognitive Radios with LLMs (GRCon 2025 contribution page) | GNU Radio Events, GRCon 2025 | Conference abstract | https://events.gnuradio.org/event/26/contributions/750/ |
| 2 | Powering Cognitive Radio with AI — GRCon 2025 paper (Paul David) | GNU Radio Conference 2025 | Conference paper | https://events.gnuradio.org/event/26/contributions/750/attachments/221/598/grcon25_paper_paul_david.pdf |
| 3 | How GR4 Can Transform Your SDR Workflows (17 Dec 2025) | GNU Radio project | Official article | https://www.gnuradio.org/news/2025-12-17-gr4-transform-sdr-workflows/ |
| 4 | First Release Candidate for GR4 (22 Mar 2026) | GNU Radio project | Release notes | https://www.gnuradio.org/news/2026-03-22-gr4-release-candidate-1/ |
| 5 | GR4 category page, incl. "A New Chapter for GNU Radio" | GNU Radio project | Official site | https://www.gnuradio.org/categories/gr4/ |
| 6 | gnuradio/gnuradio4 — GNU Radio 4 framework | GitHub (GNU Radio) | Repository | https://github.com/gnuradio/gnuradio4 |
| 7 | GR4 tutorial: blocks 101 | GNU Radio Wiki | Tutorial | https://wiki.gnuradio.org/index.php?title=GR4_tutorial:_blocks_101 |
| 8 | yoelbassin/gr-mcp — Marconi, LLM-driven RF for Claude Code | GitHub | Repository | https://github.com/yoelbassin/gr-mcp |
| 9 | GR-MCP by Dollarhyde — MCP server for GNU Radio | Glama MCP directory | Directory listing | https://glama.ai/mcp/servers/Dollarhyde/gr-mcp |
| 10 | gnuradio-mcp — MCP server for GNU Radio | PyPI | Package page | https://pypi.org/project/gnuradio-mcp/ |
| 11 | Model Context Protocol specification | modelcontextprotocol.io | Official spec | https://modelcontextprotocol.io/specification/latest |
| 12 | Lewis et al. (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | NeurIPS 2020 / arXiv:2005.11401 | Foundational paper | https://arxiv.org/abs/2005.11401 |
| 13 | Bornea et al. (2024) Telco-RAG | arXiv:2404.15939 | Preprint | https://arxiv.org/abs/2404.15939 |
| 14 | netop-team/telco-rag — open-source Telco-RAG framework | GitHub | Repository | https://github.com/netop-team/telco-rag |
| 15 | Tong et al. WirelessAgent | arXiv:2409.07964 (HKUST) | Preprint | https://arxiv.org/abs/2409.07964 |
| 16 | Akiba et al. (2019) Optuna | KDD 2019 / arXiv:1907.10902 | Foundational paper | https://arxiv.org/abs/1907.10902 |
| 17 | Bayesian Optimization for Radio Resource Management | arXiv:2012.08469 | Preprint | https://arxiv.org/abs/2012.08469 |
| 18 | Amanna et al. (2012) Parametric optimization of SDR configurations using DOE | Analog Integrated Circuits & Signal Processing 73, Springer | Peer-reviewed | https://link.springer.com/article/10.1007/s10470-012-9934-4 |
| 19 | Optuna documentation | optuna.readthedocs.io | Official docs | https://optuna.readthedocs.io/ |
| 20 | Rondeau, Le, Rieser, Bostian (2004) Cognitive Radios with Genetic Algorithms | SDR Forum Technical Conference 2004 | Conference paper | https://www.wirelessinnovation.org/assets/Proceedings/2004/2004-sdr04-1-5-1-rondeau.pdf |
| 21 | O'Shea, Roy, Clancy (2018) Over-the-Air Deep Learning Based Radio Signal Classification | IEEE JSTSP 12(1) / arXiv:1712.04578 | Peer-reviewed | https://arxiv.org/abs/1712.04578 |
| 22 | O'Shea, Hoydis (2017) An Introduction to Deep Learning for the Physical Layer | arXiv:1702.00832 | Foundational paper | https://arxiv.org/abs/1702.00832 |
| 23 | TorchDSP/torchsig — signal processing ML toolkit | GitHub | Repository | https://github.com/TorchDSP/torchsig |
| 24 | DeepSig RadioML datasets | DeepSig Inc. | Dataset portal | https://www.deepsig.ai/datasets |
| 25 | radioML/dataset — GNU Radio-based signal generation | GitHub | Repository | https://github.com/radioML/dataset |
| 26 | Rondeau et al. Cognitive Radio Formulation and Implementation | CrownCom 2006, IEEE / EUDL | Conference paper | https://eudl.eu/doi/10.1109/crowncom.2006.363476 |
| 27 | NVlabs/sionna — Sionna 2.0 (PHY, SYS, RT) | GitHub (NVIDIA) | Repository | https://github.com/NVlabs/sionna |
| 28 | Sionna: An Open-Source Library for Next-Generation PHY Research | arXiv:2203.11854 | Preprint | https://arxiv.org/abs/2203.11854 |
| 29 | Bonati et al. Colosseum: Large-Scale Wireless Experimentation Through HIL Network Emulation | arXiv:2110.10617 | Preprint | https://arxiv.org/abs/2110.10617 |
| 30 | Colosseum as a Digital Twin | arXiv:2303.17063 | Preprint | https://arxiv.org/abs/2303.17063 |
| 31 | Colosseum — The Open RAN Digital Twin (overview) | Northeastern University | Official site | https://northeastern.edu/colosseum/master-class2021 |
| 32 | Zubow et al. (2021) GrGym: When GNU Radio goes to (AI) Gym | ACM HotMobile 2021 | Conference paper | https://doi.org/10.1145/3446382.3448358 |
| 33 | tkn-tub/gr-gym — OpenAI Gym + GNU Radio framework | GitHub (TU Berlin) | Repository | https://github.com/tkn-tub/gr-gym |
| 34 | sigmf/SigMF — Signal Metadata Format specification | GitHub | Specification | https://github.com/sigmf/SigMF |
| 35 | EttusResearch/uhd — USRP Hardware Driver | GitHub (Ettus Research) | Repository | https://github.com/EttusResearch/uhd |
| 36 | UHD and USRP Manual | Ettus Research | Official docs | http://files.ettus.com/manual/ |
| 37 | Haykin (2005) Cognitive Radio: Brain-Empowered Wireless Communications | IEEE JSAC 23(2) | Peer-reviewed | https://doi.org/10.1109/JSAC.2004.839380 |
| 38 | Mitola, Maguire (1999) Cognitive radio: making software radios more personal | IEEE Personal Communications 6(4) | Peer-reviewed | https://doi.org/10.1109/98.788210 |
| 39 | Bonati et al. (2022) Intelligent Closed-loop RAN Control with xApps in OpenRAN Gym | arXiv:2208.14877 | Preprint | https://arxiv.org/abs/2208.14877 |
| 40 | OpenRAN Gym and related O-RAN control-loop projects | Northeastern WiNES Lab | Project page | https://ece.northeastern.edu/wineslab/openran.php |
| 41 | DARPA Spectrum Collaboration Challenge (10 Sep 2019) | DARPA | News release | https://www.darpa.mil/news/2019/spectrum-collaboration-challenge-event |
| 42 | PySDR: A Guide to SDR and DSP using Python | Marc Lichtman (pysdr.org) | Online textbook | https://pysdr.org |

---

## Global Comparison Matrix

| Criterion | M1 LLM-Assisted | M2 RAG-Grounded | M3 Agentic | M4 Optimization | M5 ML-DSP | M6 Simulation | M7 HIL | M8 Closed-Loop | ∑ Integrated |
|---|---|---|---|---|---|---|---|---|---|
| **Iteration** | Manual re-prompting | Re-query or expand corpus | Explicit and automated | Numerical loop with budget | Usual ML loop | Automated until targets met | Automated, slower than sim | Continuous during operation | Design→Execute→Measure→Evaluate→Redesign |
| **Experimentation** | None built in | Not inherent; draws on stored experiments | Built in via tool actions | Each evaluation is an experiment | Heavy: training runs, SNR sweeps | Controlled Monte Carlo | Real measurements | Online, bounded | Two stages: sim then hardware |
| **Validation** | Engineer HITL | Parameter traceability | Automated metric inspection + human sign-off | Objective + constraint satisfaction | Compare vs conventional baseline | Metrics vs targets | Measured metrics with repeated trials | Runtime KPIs + pre-deployment testing | Layered: static, sim, HIL, metric eval, human approval |
| **Automation** | Generation auto; eval manual | Retrieval auto; curation manual | Highest among design-time | High inside loop; objective design human | Training scriptable | High; fully scriptable | High via driver APIs | Fully automated at runtime | Steps 2–9 automatable; step 10 human |
| **Reproducibility** | Weak by default | Better with versioned corpus | At risk from nondeterministic planning | Good with fixed seeds | Needs versioned datasets and seeds | Highest when seeds fixed | Lower due to RF variation | Hard; telemetry + replay help | Designed in if artifacts versioned |
| **Governance** | Engineer accountability | Source licences and freshness | Tool permissions, budgets | Weights are value judgements | Dataset licences, model drift | Declare model-fidelity assumptions | Transmit safety, regulatory limits | Strongest need: compliance, override, fallback | Formal HITL gate + constraint validation |

---

*This atlas was compiled from the Golden Reference text and external sources accessed on 26 September 2026. Provenance marks throughout indicate the origin of each claim.*