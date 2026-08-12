# MAT099 Enhanced AI Racing Project Handbook

An industry-style wiki document for a dissertation project about creating,
testing, comparing, and explaining autonomous racing agents in TORCS.

## Purpose

This handbook supports two related goals:

1. Create and research autonomous racing agents that can drive around a TORCS
   track using rule-based control, racing-line logic, and reinforcement learning.
2. Present the system in a way that helps people who are not familiar with AI
   understand what an agent is doing, why it behaves that way, and how its
   performance can be measured.

## Project Scope

The project is a solo dissertation codebase, not a general-purpose racing
framework. It is organised to support reproducibility, explanation, and future
extension.

## Key Workflows

- Launch the GUI: `python src\gui\app.py`
- Launch the console menu: `python src\main.py`
- Run tests: `python -m unittest discover -s tests`
- Train Dyna-Q: `python scripts\train_dyna_q_policy.py --episodes 5`
- Evaluate a saved policy: `python scripts\evaluate_dyna_q_policy.py --policy-path data\policies\dyna_q_g_track_3_best.json`

## Main Sections

1. Executive Summary
2. Project Purpose and Scope
3. System Overview
4. AI Concepts for New Learners
5. Agent Guide
6. Running the Project
7. Training and Evaluation
8. Telemetry and Results
9. Reproducibility Workflow
10. Troubleshooting
11. Glossary
12. Code Map

The Word version contains the full formatted handbook, command reference,
troubleshooting table, glossary, and architecture diagram.
