# CSE691 Project

This repository contains the implementation for the CSE691 course project by **Rohitkumar Arasanipalai**.  
It explores **language-model-driven planning** for grid-based environments, integrating:
- One-step lookahead
- Rollout evaluation
- Dynamic programming for subtask execution

---

## 📌 Overview

The system uses:
- **`env.py`** — Defines and initializes the environment.
- **`gpt_utils.py`** — Utility functions for interacting with a GPT model.
- **`llm_planner.py`** — Generates a high-level plan based on mission descriptions.
- **`llm_rollout.py`** — Evaluates subtasks using simulated rollouts.
- **`lookahead.py`** — Implements one-step lookahead to choose optimal subtasks.
- **Visual assets** — `initial_observation.png`, `sequential_init.png` showing environment states.

The goal is to test whether **LLMs + rollout strategies** can produce efficient, generalizable task completion plans across different MiniGrid environments.

---

## 🛠 Installation

Install the required dependencies:

```bash
# Core dependencies
pip install numpy matplotlib imageio

# Environment
pip install gym minigrid

# LLM API access (choose one)
pip install openai  # For OpenAI GPT models
pip install google-generativeai  # For Google Gemini models

# Optional dependencies
pip install tqdm  # For progress bars
pip install matplotlib  # For visualization and plots

---

## 🛠 Configure API access 
# For OpenAI
os.environ["OPENAI_API_KEY"] = "your-api-key"


