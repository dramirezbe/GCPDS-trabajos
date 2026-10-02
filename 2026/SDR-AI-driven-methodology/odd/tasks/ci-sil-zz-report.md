# CI+SIL × zz_Workflow Report

## Objective
Create `reports/1_CI-SIL_+_zz/1_CI-SIL_+_zz.tex` — a LaTeX report that extracts and synthesizes the arguments, ideas, and references for WHY the CI+SIL × zz_Workflow methodology is the right approach for AI-assisted SDR prototyping.

## Problem
The golden context (7 artifacts) builds a layered argument from "how LLMs work" up to "how to couple zz_Workflow with CI+SIL", but that argument exists only as scattered HTML/MD artifacts. It needs to be consolidated into a single, compilable, citable LaTeX document with TikZ diagrams.

## Why
This is the first formal report of the methodology project — it crystallizes the rationale behind selecting CI+SIL as the integration strategy for the zz_Workflow prototype engine, with traceable 2026 references.

## Scope
- **In**: Extract arguments + references from the 7 golden context artifacts. Write LaTeX. TikZ diagrams for: (a) the 6-phase validation flow, (b) the zz_Workflow stages, (c) the CI+SIL coupling. Compile with `latexmk -pdf -outdir=...`. Clean with `-c`.
- **Out**: No new research. No artifact creation. No HTML. The `.tex` file IS the deliverable.

## Constraints
- Follow M2 `.tex` style (palette, packages, tcolorbox, fancyhdr) for visual consistency
- TikZ diagrams via tikz-figure-skill patterns
- All references from the golden context artifacts (2025–2026 sources)
- Compile cleanly with pdflatex (no lualatex dependency)

## Acceptance criteria
- [x] `latexmk -pdf` compiles without errors — 7 pages, no warnings
- [x] All 7 context sources reflected in the argument structure
- [x] 4 TikZ diagrams: agent-hardware loop, 6-phase flow, zz_Workflow stages, CI+SIL coupling
- [x] Bibliography with 17 references from the golden context
- [x] `latexmk -c` cleans auxiliary files — only .tex + .pdf remain

## Route
Delegated direct — single writer for 1 substantial file with multiple TikZ diagrams.

## Document structure plan

### Title page
CI+SIL × zz_Workflow: Rationale for Coupling Continuous Integration with the LLM-Assisted Prototype Engine

### §1 Introduction — Why this document
- The SDR FM sensor project uses an LLM-assisted workflow (zz_Workflow)
- Need: a robust integration strategy for verification downstream
- Thesis: CI+SIL is the natural, high-fit coupling

### §2 Foundation — How current AI agents work (from ctx 1)
- LLMs are probabilistic, not deterministic
- ~16-step degradation ceiling (arXiv:2609.01660)
- Structure beats instructions (Nerd Level Tech 2026)
- Context engineering > prompt engineering (Anthropic 2026)
- Hallucination incentivized by accuracy metrics (Nature 2026, Kalai et al.)

### §3 Hardware control via tools (from ctx 2)
- Tool/function calling + MCP pattern
- Closed-loop: agent → tool → actuator → sensor → feedback
- Two-level control: LLM plans, deterministic controller executes
- Schema-level physical limits as first defense
- **TikZ diagram**: closed-loop agent–hardware interaction

### §4 Validation methodologies compared (from ctx 3)
- 5 approaches: V-model, CI+SIL, HIL benches, LLM-as-judge+HITL, MCDA
- Shared principles: verify ≠ validate, cheap→expensive, external oracle, human at extremes
- **TikZ diagram**: 6-phase validation flow (Encuadre → Modelo → SW/CI → HIL → Juicio → MCDA)

### §5 Validating hypotheses with AI (from ctx 4)
- Code: generate → execute → test → repair loop
- LLM-as-judge with calibration (Cohen's κ)
- HIL when restriction is hardware
- HITL: review and arbitrage
- "Validated in simulation" ≠ "validated" — the gap

### §6 The zz_Workflow engine (from ctx 5)
- 4 stages (Ctx 00–03), gated by human
- Model never self-approves
- 6 governing artifacts
- Output: Verified Prototype + Evidence
- **TikZ diagram**: zz_Workflow stages with gates and feedback R0–R3

### §7 Coupling analysis — Why CI+SIL (from ctx 6)
- Coupling matrix: CI+SIL = HIGH/clean, V&V = HIGH-MEDIUM, others lower
- Artefact 3 already IS a regression harness
- "Verified" ≠ "validated" — precise and honest naming
- LLM-as-judge omission is a strength (arXiv:2609.02246)
- Coverage of 6-phase structure: phases 1,3,5 covered; 2 partial; 4,6 gaps

### §8 The refined coupling (from ctx 7)
- Evidence seeds baseline regression
- Trigger = model/prompt/Ctx change (not just code commits)
- Deterministic acceptance gates preserved
- R0–R3 defect routing reused
- HIL vivo remains downstream gap
- **TikZ diagram**: CI+SIL × zz_Workflow coupled block diagram

### §9 Conclusion
- CI+SIL is the natural first integration tier
- zz_Workflow's gated, traceable, regression-ready design makes it high-fit
- Downstream work: HIL bench, MCDA for multi-hypothesis selection

### References
~13 entries from golden context, bibtex

## Tasks

- [x] T1: Explore context and plan — this document
- [x] T2–T5: Write full .tex (§1–9, 4 TikZ figs, 17 refs), compile, clean — delegated writer
- [x] T6: Verify compilation, review PDF output — inline spot-check

## TDD mode
Disabled (LaTeX document, not software). Functional check = `latexmk -pdf` compiles.

## Delivery
Single work-unit commit. ~400 line heuristic advisory only — LaTeX documents are naturally longer.

## Forecast
~500–700 authored lines (LaTeX + TikZ). Single file, single commit.
