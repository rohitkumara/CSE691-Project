
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
    desc.append("- The agent can only Move to, Pick up, Drop, or Toggle objects.")
    desc.append("- The agent must deliver the box to the goal by picking it up and dropping it adjacent to the goal.")
    desc.append("- Shorter paths are preferred.")

    return "\n".join(desc)

def get_gpt_subtasks(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": "List multiple valid subtasks using only allowed actions (Move to, Pick up, Drop, Toggle)."},
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
    filtered = []
    for task in subtasks:
        if any(kw in task.lower() for kw in valid_keywords):
            filtered.append(task)
    return filtered

def is_plan_valid(plan_steps):
    forbidden_phrases = ["push", "move the box", "move box", "signal", "completion", "celebrate"]
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

def best_first_fortified_rollout_planner_multi_start():
    env = SimpleEnv()
    obs, _ = env.reset()
    objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]
    agent_pos = tuple(env.agent_pos)

    best_global_plan = []
    best_global_score = float('inf')

    initial_starts = ["Pick up the purple box", "Pick up the blue key"]
    width = env.width
    height = env.height

    for start_task in initial_starts:
        queue = []
        iteration = 0

        queue.append((0, [start_task]))
        heapq.heapify(queue)

        best_local_plan = []
        best_local_score = float('inf')

        while queue:
            current_cost, current_chain = heapq.heappop(queue)

            if current_cost > best_local_score:
                continue

            iteration += 1
            candidate_id = 0
            os.makedirs(f"plans_{sanitize_filename(start_task)}_iter_{iteration}", exist_ok=True)

            env_desc = generate_environment_description(agent_pos, objects, goal, width, height)
            prompt = f"""{env_desc}

The agent has already completed: {current_chain}.
Mission:
- {mission}

Suggest next valid subtasks."""
            subtasks = filter_valid_subtasks(get_gpt_subtasks(prompt))
            if not subtasks:
                continue

            for subtask in subtasks:
                candidate_id += 1
                candidate_chain = current_chain + [subtask]
                plan = get_gpt_subtasks(prompt)

                if not is_plan_valid(plan):
                    continue

                save_path = f"plans_{sanitize_filename(start_task)}_iter_{iteration}/plan_candidate_{candidate_id}.txt"
                save_plan_text(plan, save_path)

                env_copy = copy.deepcopy(env)
                path = PathPlanner(env_copy, None)
                rollout_cost = execute_plan_with_gif(env_copy, plan, path, gif_name=f"plans_{sanitize_filename(start_task)}_iter_{iteration}/rollout_candidate_{candidate_id}.gif")
                if rollout_cost == float('inf'):
                    continue

                if rollout_cost < best_local_score:
                    best_local_score = rollout_cost
                    best_local_plan = plan

                heapq.heappush(queue, (rollout_cost, candidate_chain))

        if best_local_plan and best_local_score < best_global_score:
            best_global_score = best_local_score
            best_global_plan = best_local_plan

    if not best_global_plan:
        print("[No valid plan found. Skipping final execution.]")
        return

    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    print("\n[Executing Final Best Plan:]")
    final_steps = execute_plan_with_gif(final_env, best_global_plan, final_path, gif_name="final_best_plan.gif")
    print("[Total Steps]:", final_steps)

if __name__ == "__main__":
    best_first_fortified_rollout_planner_multi_start()
