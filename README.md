# CSE691 Project

This repository contains the implementation for the CSE691 course.
It explores **LLM driven planning** for grid-based environments, integrating:
- One-step lookahead
- Rollout evaluation
- Dynamic programming for subtask execution

---

## Overview

The system uses:
- **`env.py`** — Defines and initializes the environment (should be modified if you want to test different minigrid envs).
- **`gpt_utils.py`** — Utility functions for interacting with a GPT model.
- **`llm_planner.py`** — Generates a high-level plan based on mission descriptions.
- **`llm_rollout.py`** — Evaluates subtasks using simulated rollouts (main file that should be run)
- **`lookahead.py`** — Implements one-step lookahead to choose optimal subtasks.
- **Visual assets** — `initial_observation.png`, `sequential_init.png` showing environment states.

The goal is to test whether **LLMs + rollout strategies** can produce efficient, generalizable task completion plans across different MiniGrid environments.

---

## 🛠 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

---

## 🛠 Configure API access 
# For OpenAI
os.environ["OPENAI_API_KEY"] = "your-api-key"


