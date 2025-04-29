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

def save_plan_text(plan_steps, save_path, depth=None, child=None, mission=None):
    with open(save_path, 'w') as f:
        f.write("=== PLAN DETAILS ===\n")
        if depth is not None and child is not None:
            f.write(f"Depth: {depth} | Child: {child}\n")
        if mission:
            f.write(f"Mission: {mission}\n")
        f.write("\n=== Steps ===\n")
        for idx, step in enumerate(plan_steps, 1):
            f.write(f"{idx}. {step}\n")
        f.write("\n=== End of Plan ===\n")

def save_plan_summary(summary_path, depth, child, rollout_cost, plan_path, gif_path):
    with open(summary_path, 'a') as f:
        if rollout_cost == float('inf'):
            cost_str = "INF (Failed)"
        else:
            cost_str = f"{rollout_cost:.2f}"  # 2 decimals
        f.write(f"Depth {depth} | Child {child} | Cost: {cost_str}\n")
        f.write(f"Plan File: {plan_path}\n")
        f.write(f"GIF File: {gif_path}\n")
        f.write("------------------------------------------------------\n")

# def apply_subtask_to_env(env, subtask):
#     """Applies a high-level subtask like Move to object, Pick up, Drop, Toggle to the real environment."""
#     subtask = subtask.lower()

#     from gpt_utils import get_target_position  # or your parsing logic
#     objects = extract_objects(env.grid)
#     parsed = get_target_position(subtask, objects)

#     if parsed is None:
#         print(f"[Warning] Could not parse subtask: {subtask}")
#         return

#     if isinstance(parsed, tuple):
#         # Move to target position
#         target_pos = parsed
#         path = PathPlanner(env, None)
#         path.set_goal(target_pos)

#         while path.manhattan(env.agent_pos, target_pos) > 1:
#             action = path.one_step_lookahead_with_rollout(path.extract_state(env))
#             obs, _, _, _, _ = env.step(action)
#     else:
#         action_type, obj_type = parsed
#         action_map = {
#             'pickup': 4,
#             'drop': 5,
#             'toggle': 6
#         }
#         action = action_map.get(action_type)
#         if action is not None:
#             obs, _, _, _, _ = env.step(action, obj_type)


def apply_subtask_to_env(env, subtask):
    """Applies a high-level subtask like Move to object, Pick up, Drop, Toggle to the real environment."""
    subtask = subtask.lower()

    from gpt_utils import get_target_position
    objects = extract_objects(env.grid)
    parsed = get_target_position(subtask, objects)

    if parsed is None:
        print(f"[Warning] Could not parse subtask: {subtask}")
        return

    if isinstance(parsed, tuple):
        # 🌟 If parsed as a real (x, y) position → Move
        target_pos = parsed
        path = PathPlanner(env, None)
        path.set_goal(target_pos)

        while path.manhattan(env.agent_pos, target_pos) > 1:
            action = path.one_step_lookahead_with_rollout(path.extract_state(env))
            if action is None:
                print("[Warning] Agent stuck, cannot move to target.")
                return
            obs, _, _, _, _ = env.step(action)
    elif isinstance(parsed, list) or isinstance(parsed, tuple):
        # If parsing returned a list (e.g., ["pickup", "yellow key"]) — action + object
        action_type, obj_type = parsed
        action_map = {
            'pickup': 4,
            'drop': 5,
            'toggle': 6
        }
        action = action_map.get(action_type)
        if action is not None:
            obs, _, _, _, _ = env.step(action, obj_type)
    else:
        print(f"[Warning] Unknown parsed format: {parsed}")

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
    current_env = copy.deepcopy(env)

    agent_pos = tuple(current_env.agent_pos)
    objects = extract_objects(current_env.grid)
    goal = obs["goal"]
    mission = obs["mission"]

    current_chain = []
    best_plan = []
    best_score = float('inf')

    iteration = 0
    MAX_DEPTH = 100
    COST_TOLERANCE = 3
    done = False

    os.makedirs("rollouts", exist_ok=True)
    summary_path = "rollouts/plans_summary.txt"
    open(summary_path, 'w').close()

    print("\n[Starting Final Corrected One-Step Rollout Search]\n")

    while not done and iteration < MAX_DEPTH:
        # 🌟 REGENERATE updated environment description
        num_subtasks = 8 if iteration == 0 else (6 if iteration == 1 else 3)

        subtasks = get_subtasks(
            agent_pos=agent_pos,
            objects=objects,
            goal=goal,
            mission=mission,
            prior_chain=current_chain,
            width=current_env.width,
            height=current_env.height,
            num_subtasks=num_subtasks,
        )

        if not subtasks:
            print("[No more subtasks to suggest. Stopping.]")
            break

        best_subtask = None
        best_rollout_cost = float('inf')
        child_id = 0

        for subtask in subtasks:
            child_id += 1
            candidate_chain = current_chain + [subtask]
            plan = generate_plan_from_chain(candidate_chain, agent_pos, objects, goal, mission)

            if not is_plan_valid(plan):
                continue

            env_copy = copy.deepcopy(current_env)
            path = PathPlanner(env_copy, None)

            rollout_gif_path = f"rollouts/rollout_depth{iteration}_child{child_id}.gif"
            rollout_cost = execute_plan_with_gif(env_copy, plan, path, gif_name=rollout_gif_path)

            plan_text_path = f"rollouts/plan_depth{iteration}_child{child_id}.txt"
            save_plan_text(plan, plan_text_path, depth=iteration, child=child_id, mission=mission)
            save_plan_summary(summary_path, iteration, child_id, rollout_cost, plan_text_path, rollout_gif_path)

            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_subtask = subtask

        if best_subtask is None:
            print("[No valid subtask led to improvement. Stopping.]")
            break

        # 🚨 Cost Tolerance Check
        if best_score != float('inf') and best_rollout_cost > best_score + COST_TOLERANCE:
            print(f"[Stopping: rollout cost worsened from {best_score} to {best_rollout_cost}]")
            break

        if best_rollout_cost < best_score or best_score == float('inf'):
            best_score = best_rollout_cost

        # ✅ Apply best subtask to real environment
        print(f"[Step {iteration}] Executing best subtask: {best_subtask}")
        current_chain.append(best_subtask)
        apply_subtask_to_env(current_env, best_subtask)

        # ✅ 🌟 UPDATE environment state properly
        agent_pos = tuple(current_env.agent_pos)
        objects = extract_objects(current_env.grid)  # 🌟 UPDATE objects list after subtask execution

        if is_box_at_goal(current_env):
            print("[Box delivered successfully!]")
            done = True

        iteration += 1

    print("\n[Final Best Plan Found]:", current_chain)
    print("[Best Cost]:", best_score)

    if not current_chain:
        print("[No valid plan found. Skipping final execution.]")
        return

    print("\n[Replaying Final Best Plan on Fresh Environment]\n")
    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    final_steps = execute_plan_with_gif(final_env, current_chain, final_path, gif_name="rollouts/final_best_plan.gif")
    print("[Total Steps Taken]:", final_steps)

if __name__ == "__main__":
    best_first_fortified_rollout_planner()
