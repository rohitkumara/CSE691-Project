from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan
import matplotlib.pyplot as plt
import copy
import imageio
import os
import re
import heapq
from gpt_utils import client

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def save_plan_text(plan, save_path):
    with open(save_path, 'w') as f:
        for step in plan:
            f.write(step + "\n")

def generate_environment_description(agent_pos, objects, goal_pos, width=10, height=10):
    desc = []
    desc.append(f"Environment Overview:")
    desc.append(f"- The environment is a {width}x{height} grid.")
    desc.append(f"- The agent starts at position {agent_pos}.")

    desc.append("Objects:")
    for obj in objects:
        desc.append(f"- A {obj['color']} {obj['type']} is located at {obj['pos']}.")

    desc.append(f"Goal:")
    desc.append(f"- The goal is located at {goal_pos}.")

    desc.append("Important Notes:")
    desc.append("- Some doors may be locked and block movement.")
    desc.append("- To open a locked door, the agent must first pick up the key of matching color.")
    desc.append("- The agent can only Move, Pick up, Drop, or Toggle objects.")
    desc.append("- The agent must deliver the box to the goal by picking it up and dropping it adjacent to the goal.")
    desc.append("- Shorter paths are preferred.")

    return "\n".join(desc)

def get_gpt_subtasks(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": "List multiple valid subtasks using only allowed actions (Move, Pick up, Drop, Toggle)."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256
    )
    plan = response.choices[0].message.content
    subtasks = [line.strip().lstrip("1234567890.- ") for line in plan.split("\n") if line.strip()]
    return subtasks

def filter_valid_subtasks(subtasks):
    valid_keywords = ["move to", "pick up", "drop", "toggle"]
    forbidden_phrases = ["left", "right", "up", "down", "step", "move to (", "move one", "move a step"]
    filtered = []
    for task in subtasks:
        lower_task = task.lower()
        if any(kw in lower_task for kw in valid_keywords) and not any(fp in lower_task for fp in forbidden_phrases):
            filtered.append(task)
    return filtered


def get_subtasks(agent_pos, objects, goal, mission, prior_chain=None, width=10, height=10, num_subtasks=3):
    env_desc = generate_environment_description(agent_pos, objects, goal, width, height)
    if prior_chain is None:
        prompt = f"""{env_desc}

Mission:
- {mission}

Guidelines for suggesting subtasks:
- ONLY suggest high-level actions based on objects like keys, doors, box, and goal.
- DO NOT suggest simple movements like "move left", "move right", "move to (x,y)", or "move one step".
- Each subtask must involve a meaningful interaction with an object (e.g., move to yellow key, pick up box, toggle red door, drop box at goal).
- Allowed actions: Move to (object), Pick up (object), Drop (object), Toggle (object).

Suggest around {num_subtasks} valid subtasks."""
    else:
        prompt = f"""{env_desc}

The agent has already completed: {prior_chain}.
Mission:
- {mission}

Guidelines for suggesting subtasks:
- ONLY suggest high-level actions based on objects like keys, doors, box, and goal.
- DO NOT suggest simple movements like "move left", "move right", "move to (x,y)", or "move one step".
- Each subtask must involve a meaningful interaction with an object (e.g., move to yellow key, pick up box, toggle red door, drop box at goal).
- Allowed actions: Move to (object), Pick up (object), Drop (object), Toggle (object).

Suggest around {num_subtasks} next valid subtasks."""

    print("\n[LLM Prompt for Subtasks]\n", prompt)
    subtasks = get_gpt_subtasks(prompt)
    subtasks = filter_valid_subtasks(subtasks)
    return subtasks


def is_plan_valid(plan_steps):
    forbidden_phrases = ["push", "signal", "completion", "celebrate"]
    for step in plan_steps:
        for phrase in forbidden_phrases:
            if phrase in step.lower():
                return False
    return True

def is_box_at_goal(env):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    goal_x, goal_y = env.goal
    for dx, dy in directions + [(0,0)]:
        x, y = goal_x + dx, goal_y + dy
        if 0 <= x < env.grid.width and 0 <= y < env.grid.height:
            obj = env.grid.get(x, y)
            if obj and obj.type == 'box':
                return True
    return False

def execute_plan_with_gif(env, plan, path, gif_name="rollout.gif"):
    print("Executing plan and saving GIF...")
    timesteps = 0
    frames = []
    unchanged_counter = 0
    from gpt_utils import get_target_position

    last_pos = tuple(env.agent_pos)

    for step in plan:
        objects = extract_objects(env.grid)
        print("-", step)
        target_pos = get_target_position(step, objects)
        if target_pos is None:
            continue

        if isinstance(target_pos[0], str):
            action_map = {
                'pickup': 4,
                'drop': 5,
                'toggle': 6
            }
            action = action_map.get(target_pos[0])
            obs, _, _, _, _ = env.step(action, target_pos[1])
            frames.append(obs["image"])
            timesteps += 1

            if tuple(env.agent_pos) == last_pos:
                unchanged_counter += 1
            else:
                unchanged_counter = 0
            last_pos = tuple(env.agent_pos)

            if unchanged_counter >= 5:
                print("[Plan execution aborted: agent stuck]")
                imageio.mimsave(gif_name, frames, fps=2)
                return float('inf')

        elif isinstance(target_pos, tuple):
            path.set_goal(target_pos)
            while path.manhattan(env.agent_pos, target_pos) > 1:
                if unchanged_counter >= 5:
                    print("[Plan execution aborted: agent stuck]")
                    imageio.mimsave(gif_name, frames, fps=2)
                    return float('inf')
                action = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _, _ = env.step(action)
                frames.append(obs["image"])
                timesteps += 1

                if tuple(env.agent_pos) == last_pos:
                    unchanged_counter += 1
                else:
                    unchanged_counter = 0
                last_pos = tuple(env.agent_pos)

    imageio.mimsave(gif_name, frames, fps=2)

    if not is_box_at_goal(env):
        print("[Rollout failed: box not delivered to goal]")
        return float('inf')

    return timesteps

def generate_plan_from_chain(subtask_chain, agent_pos, objects, goal, mission, width=10, height=10):
    env_desc = generate_environment_description(agent_pos, objects, goal, width, height)
    prompt = f"""{env_desc}

The agent must first perform: {subtask_chain}.
Then continue generating steps to complete the full mission: {mission}.
Only allowed actions: Move, Pick up, Drop, Toggle.
"""
    prompt += format_env_state_for_gpt(agent_pos, objects, goal, mission)
    print("\n[LLM Prompt for Full Plan Generation]\n", prompt)
    return get_gpt_plan(prompt)

def best_first_fortified_rollout_planner():
    env = SimpleEnv()
    obs, _ = env.reset()
    objects = extract_objects(env.grid)  # ✅ Extract once
    goal = obs["goal"]
    mission = obs["mission"]
    agent_pos = tuple(env.agent_pos)

    best_plan = []
    best_score = float('inf')
    best_chain = []

    queue = []
    iteration = 0

    initial_subtasks = get_subtasks(agent_pos, objects, goal, mission, prior_chain=None, num_subtasks=8)
    for subtask in initial_subtasks:
        queue.append((0, [subtask]))

    heapq.heapify(queue)

    while queue:
        current_cost, current_chain = heapq.heappop(queue)

        if current_cost > best_score:
            print(f"Skipping expansion: current cost {current_cost} worse than best score {best_score}")
            continue

        iteration += 1
        candidate_id = 0
        os.makedirs(f"plans_iteration_{iteration}", exist_ok=True)

        # 🌟 Dynamic number of subtasks based on depth
        if len(current_chain) == 0:
            num_subtasks = 8
        elif len(current_chain) == 1:
            num_subtasks = 6
        else:
            num_subtasks = 3

        subtasks = get_subtasks(agent_pos, objects, goal, mission, prior_chain=current_chain, num_subtasks=num_subtasks)

        if not subtasks:
            continue

        for subtask in subtasks:
            candidate_id += 1
            candidate_chain = current_chain + [subtask]
            plan = generate_plan_from_chain(candidate_chain, agent_pos, objects, goal, mission)

            if not is_plan_valid(plan):
                print("[Plan rejected: contains forbidden actions]")
                continue

            save_path = f"plans_iteration_{iteration}/plan_candidate_{candidate_id}.txt"
            save_plan_text(plan, save_path)

            env_copy = copy.deepcopy(env)
            path = PathPlanner(env_copy, None)
            rollout_cost = execute_plan_with_gif(env_copy, plan, path, gif_name=f"plans_iteration_{iteration}/rollout_candidate_{candidate_id}.gif")
            if rollout_cost == float('inf'):
                continue

            if rollout_cost < best_score:
                best_score = rollout_cost
                best_plan = plan
                best_chain = candidate_chain

            heapq.heappush(queue, (rollout_cost, candidate_chain))

    print("\n[Best Subtask Chain Found]:", best_chain)
    print("[Best Cost]:", best_score)

    if not best_plan:
        print("[No valid plan found. Skipping final execution.]")
        return

    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    print("\n[Executing Final Best Plan:]")
    final_steps = execute_plan_with_gif(final_env, best_plan, final_path, gif_name="final_best_plan.gif")
    print("[Total Steps]:", final_steps)

if __name__ == "__main__":
    best_first_fortified_rollout_planner()
