import csv
from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan, get_target_position
import matplotlib.pyplot as plt
import copy
import imageio
import os
import re
import heapq
from gpt_utils import client
#import sys

# ========== Utility Functions ==========

def save_plan_text(plan_steps, save_path, depth=None, child=None, mission=None, prior_chain=None):
    with open(save_path, 'w') as f:
        f.write("=== PLAN DETAILS ===\n")
        if depth is not None and child is not None:
            f.write(f"Depth: {depth} | Child: {child}\n")
        if mission:
            f.write(f"Mission: {mission}\n")
        
        # 🌟 Log already completed subtasks (prior_chain)
        if prior_chain:
            f.write("\nCompleted Subtasks So Far:\n")
            for i, sub in enumerate(prior_chain, 1):
                f.write(f"- {i}. {sub}\n")

        f.write("\n=== Steps ===\n")
        for idx, step in enumerate(plan_steps, 1):
            f.write(f"{idx}. {step}\n")
        f.write("\n=== End of Plan ===\n")


def save_plan_summary(summary_path, depth, child, rollout_cost, plan_path, gif_path):
    with open(summary_path, 'a') as f:
        if rollout_cost == float('inf'):
            cost_str = "INF (Failed)"
        else:
            cost_str = f"{rollout_cost:.2f}"
        f.write(f"Depth {depth} | Child {child} | Cost: {cost_str}\n")
        f.write(f"Plan File: {plan_path}\n")
        f.write(f"GIF File: {gif_path}\n")
        f.write("------------------------------------------------------\n")

def apply_subtask_to_env(env, subtask):
    """Applies a high-level subtask like Move to object, Pick up, Drop, Toggle to the real environment."""
    subtask = subtask.lower()
    objects = extract_objects(env.grid)
    parsed = get_target_position(subtask, objects)

    if parsed is None:
        print(f"[Warning] Could not parse subtask: {subtask}")
        return
    
    if isinstance(parsed[0], str):
        # Action on object: pickup/drop/toggle
        action_type, obj_type = parsed
        action_map = {
            'pickup': 4,
            'drop': 5,
            'toggle': 6
        }
        action = action_map.get(action_type)
        if action is not None:
            obs, _, _, _, _ = env.step(action, obj_type)
    elif isinstance(parsed, tuple):
        
        target_pos = parsed
        path = PathPlanner(env, None)
        path.set_goal(target_pos)

        unchanged_counter = 0
        last_pos = tuple(env.agent_pos)

        while path.manhattan(env.agent_pos, target_pos) > 1:
            action = path.one_step_lookahead_with_rollout(path.extract_state(env))
            if action is None:
                print("[Warning] Agent stuck, cannot move to target.")
                return
            obs, _, _, _, _ = env.step(action)
            current_pos = tuple(env.agent_pos)
            if current_pos == last_pos:
                unchanged_counter += 1
            else:
                unchanged_counter = 0
            last_pos = current_pos

            if unchanged_counter >= 5:
                print(f"[Stuck] Agent did not move while trying to reach {target_pos}. Aborting subtask.")
                return "stuck"
    else:
        print(f"[Warning] Unknown parsed format: {parsed}")

def generate_environment_description(env, agent_pos, objects, goal_pos, width=10, height=10):
    desc = []
    desc.append(f"Environment Overview:")
    desc.append(f"- The environment is a {width}x{height} grid.")
    desc.append(f"- The agent starts at position {agent_pos}.")
    
    if env.carrying is not None:
        desc.append(f"- The agent is currently carrying a {env.carrying.color} {env.carrying.type}.")
    else:
        desc.append("- The agent is currently not carrying anything.")
    
    desc.append("Objects:")
    for obj in objects:
        if obj['type'] == 'door':
            if hasattr(obj["obj"], 'is_locked'):
                state = "locked" if obj["obj"].is_locked else "unlocked"
                desc.append(f"- A {state} {obj['color']} door is located at {obj['pos']}.")
        else:
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


def get_subtasks(env, agent_pos, objects, goal, mission, prior_chain=None, width=10, height=10, num_subtasks=3):
    print("SUBTASKS COUNT:", num_subtasks)
    env_desc = generate_environment_description(env, agent_pos, objects, goal, width, height)
    if prior_chain is None:
        prompt = f"""{env_desc}

Mission:
- {mission}

Instructions:
- List only high-level subtasks needed to complete the mission.
- Use only these actions: `Move to`, `Pick up`, `Drop`, `Toggle`
- Subtasks must describe *intentional transitions* (e.g., moving to a key, unlocking a door, reaching the goal).
- Each subtask must be a single line.
- Do not include object descriptions, locations, colors, or explanations.
- DO NOT suggest simple movements like "move left", "move right", "move to (x,y)", or "move one step".
- Do not include more than necessary — only key transitions.

---

Output format:
Subtasks:
- Move to
- Pick up
- Toggle
...

Suggest around {num_subtasks} valid subtasks.
Begin now."""
    else:
        prompt = f"""{env_desc} 

The agent has already completed: {prior_chain}.
Mission:
- {mission}

Instructions:
- List only high-level subtasks needed to complete the mission.
- Use only the following actions: `Move to`, `Pick up`, `Toggle`, `Drop`
- Use object names (e.g., "blue key", "green door", "box", "goal") as targets.
- Do not include object coordinates or explain anything.
- Only include subtasks required to unlock doors or make progress.
- Assume the agent cannot pass through locked doors unless they are toggled.
- Subtasks must appear in a valid order based on reachability.
- To open a locked door, the agent must first pick up the key of matching color.


---

Output format:
Subtasks:
- [Action]
- [Action]
...

Suggest around {num_subtasks} next valid subtasks.
Begin now."""

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
    count = 0
    for goal in env.goal:
        goal_x, goal_y = goal[0]
        color = goal[1]
        x, y = goal_x, goal_y
        obj = env.grid.get(x, y)
        if obj and obj.type == 'box' and obj.color == color:
            count +=1
        
    return count == len(env.goal)

def execute_plan_with_gif(env, plan, path, gif_name="rollout.gif"):
    print("Executing plan and saving GIF...")
    timesteps = 0
    frames = []
    unchanged_counter = 0
    last_pos = tuple(env.agent_pos)

    for step in plan:
        objects = extract_objects(env.grid)
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
                objects = extract_objects(env.grid)
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

def generate_plan_from_chain(env, subtask_chain, agent_pos, objects, goal, mission, width=10, height=10):
    env_desc = generate_environment_description(env, agent_pos, objects, goal, width, height)
    prompt = f"""{env_desc}

            The agent has already completed: {subtask_chain}.
            Then continue generating steps to complete the full mission: {mission}.
            Only allowed actions: Move, Pick up, Drop, Toggle.
            """
    # === Environment Description ===
    env_lines = []
    # === Rules & Guidelines ===
    env_lines.append("=== Instructions ===")
    env_lines.append("- Generate a high-level plan using only: Move to, Pick up, Drop, Toggle.")

    env_lines.append("- The agent can carry only one object at a time.")
    env_lines.append("- Once the goal is reachable and agent is holding the box, use: 'Move to goal' and 'Drop box'.")
    env_lines.append("- Some doors may be locked and block movement.")
    env_lines.append("- To open a locked door, the agent must first pick up the key of matching color.")
    env_lines.append("- The box must be delivered to the goal of the same color. For example, a purple box must be delivered to a purple goal.")
    env_lines.append("- The agent can only Move, Pick up, Drop, or Toggle objects.")
    env_lines.append("- The agent must deliver the box to the goal by picking it up and dropping it adjacent to the goal.")
    env_lines.append("- Shorter paths are preferred.")

    # === Output Style ===
    env_lines.append("=== Format ===")
    env_lines.append("Numbered step-by-step plan from current state to mission completion.")
    env_lines.append("Use only 1 line per action. No explanations.")
    env_lines.append("")

    example = (
        """Example:
            Agent starts at (2, 2)  
            - A cyan box is at (3, 1)  
            - A green key is at (3, 6)  
            - A locked green door is at (5, 6)  
            - The purple goal is at (7, 5)
            - The cyan goal is at (4, 5)

            Example: Possible subtasks:
            - Move to the box
            - Move to the green key
            - Move to the green door

            Mission: Deliver the box to the goal.

            Steps:
            1. Move to purple box  
            2. Pick up the purple box  
            3. Move to key
            4. Drop the purple box
            5. Pick up the key  
            6. Move to door 
            7. Open the door  
            8. Move to purple box  
            9. Drop the key  
            10. Pick up the purple box  
            11. Move to the purple goal 
            12. Drop the purple box

            ---

            """
        
        "Plan the steps the agent should take to solve the task."
    )

    # === Final Prompt ===
    
    full_prompt = prompt + "\n".join(env_lines) + example

    #full_prompt = full_prompt

    print("\n[LLM Prompt for Full Plan Generation]\n", full_prompt)
    return get_gpt_plan(full_prompt)

#def generate_plan_from_chain(env, subtask_chain, agent_pos, objects, goal, mission, width=10, height=10):
    env_desc = generate_environment_description(env, agent_pos, objects, goal, width, height)
    prompt = f"""{env_desc}

The agent already performed: {subtask_chain}.
Then continue generating steps to complete the full mission: {mission}.
Only allowed actions: Move, Pick up, Drop, Toggle.
"""
    prompt += format_env_state_for_gpt(agent_pos, objects, goal, mission)
    print("\n[LLM Prompt for Full Plan Generation]\n", prompt)
    return get_gpt_plan(prompt)

# ========== Main Planner ==========
def best_first_fortified_rollout_planner():
    env = SimpleEnv()
    obs, _ = env.reset()
    
    plt.imshow(obs["image"])
    plt.title("Initial Observation")
    plt.imsave("initial_observation.png", obs["image"])
    current_env = copy.deepcopy(env)

    agent_pos = tuple(current_env.agent_pos)
    objects = extract_objects(current_env.grid)
    goal = obs["goal"]
    mission = obs["mission"]

    current_chain = []
    best_score = float('inf')

    iteration = 0
    MAX_DEPTH = 30
    COST_TOLERANCE = 3
    done = False

    cost_evolution = []
    csv_log = []

    os.makedirs("rollouts", exist_ok=True)
    summary_path = "rollouts/plans_summary.txt"
    open(summary_path, 'w').close()

    print("\n[Starting Corrected Rollout Planning]\n")

    env_desc = generate_environment_description(env,
            agent_pos,
            objects,
            goal,
            width=current_env.width,
            height=current_env.height
        )

    while not done and iteration < MAX_DEPTH:
        num_subtasks = 8 if iteration == 0 else (6 if iteration == 1 else 3)


        # Always regenerate fresh environment description
        env_desc = generate_environment_description(current_env,
            agent_pos,
            objects,
            goal,
            width=current_env.width,
            height=current_env.height
        )

        subtasks = get_subtasks(current_env,
            agent_pos=agent_pos,
            objects=objects,
            goal=goal,
            mission=mission,
            prior_chain=current_chain,
            width=current_env.width,
            height=current_env.height,
            num_subtasks=num_subtasks
        )


        if not subtasks:
            print("[No subtasks found. Stopping.]")
            break

        best_subtask = None
        best_rollout_cost = float('inf')
        child_id = 0

        print("Potential subtasks generated: ", subtasks)

        for subtask in subtasks:
            child_id += 1
            print(child_id, "Current chosen subtask for rollout:", subtask)
            subtask_env = copy.deepcopy(current_env)
            h = ""
            h = apply_subtask_to_env(subtask_env, subtask)
            if h == "stuck":
                continue
            agent_pos = subtask_env.agent_pos
            objects = extract_objects(subtask_env.grid)

            candidate_chain = current_chain + [subtask]
            agent_pos = (int(agent_pos[0]), int(agent_pos[1]))

            plan = generate_plan_from_chain(subtask_env, candidate_chain, agent_pos, objects, goal, mission)


            if not is_plan_valid(plan):
                continue

            env_copy = copy.deepcopy(subtask_env)
            path = PathPlanner(env_copy, None)

            rollout_gif_path = f"rollouts/rollout_depth{iteration}_child{child_id}.gif"
            rollout_cost = execute_plan_with_gif(env_copy, plan, path, gif_name=rollout_gif_path)

            print("Plan:", plan, "cost:", rollout_cost)
            print()
            plan_text_path = f"rollouts/plan_depth{iteration}_child{child_id}.txt"
            save_plan_text(plan, plan_text_path, depth=iteration, child=child_id, mission=mission, prior_chain=current_chain)
            save_plan_summary(summary_path, iteration, child_id, rollout_cost, plan_text_path, rollout_gif_path)

            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                print(subtask)
                best_subtask = subtask
                best_plan = [best_subtask] + plan


        if best_subtask is None:
            print("[No valid subtask selected. Stopping.]")
            break

        if best_score != float('inf') and best_rollout_cost > best_score + COST_TOLERANCE:
            print(f"[Stopping: rollout cost worsened from {best_score} to {best_rollout_cost}]")
            break

        if best_rollout_cost < best_score or best_score == float('inf'):
            best_score = best_rollout_cost
            final_best_plan = best_plan

        cost_evolution.append(best_score)
        csv_log.append((iteration, best_subtask, best_score))

        print(f"[Step {iteration}] Executing best subtask: {best_subtask}")
        current_chain.append(best_subtask)

        # Apply subtask and immediately refresh environment
        apply_subtask_to_env(current_env, best_subtask)
        agent_pos = tuple(current_env.agent_pos)
        objects = extract_objects(current_env.grid)

        # Fresh environment description is generated each loop

        if is_box_at_goal(subtask_env):
            print("[Box delivered successfully!]")
            done = True

        iteration += 1

    print("\n[Final Subtask Chain Completed]:", current_chain)
    print("[Best Cost So Far]:", best_score)

    # === FINAL FULL PLAN GENERATION ===
    print("\n[Generating Final Full Plan from Final World State]\n")

    final_objects = extract_objects(current_env.grid)
    final_agent_pos = tuple(current_env.agent_pos)
    final_goal = current_env.goal


    # final_best_plan = ['Move to purple box', 'Pick up the purple box', 'Move to the green key', 'Drop the purple box', 'Pick up the green key', 'Move to the green door', 'Toggle the green door', 'Move to the purple box', 'Pick up the purple box', 'Move to the purple goal', 'Drop the purple box']
    final_plan_text_path = "rollouts/final_best_plan.txt"
    save_plan_text(final_best_plan, final_plan_text_path, mission=mission, prior_chain=current_chain)
    print(f"[Saved Final Best Plan to {final_plan_text_path}]")

    # Save CSV log
    csv_path = "rollouts/plans_cost_log.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Depth", "Subtask", "Cost"])
        for depth, subtask, cost in csv_log:
            writer.writerow([depth, subtask, f"{cost:.2f}"])
    print(f"[Saved Planning Log to {csv_path}]")

    # Plot cost evolution
    plt.figure(figsize=(8,5))
    plt.plot(range(len(cost_evolution)), cost_evolution, marker='o')
    plt.title('Rollout Cost vs Planning Depth')
    plt.xlabel('Planning Step (Depth)')
    plt.ylabel('Rollout Cost')
    plt.grid(True)
    plt.savefig("rollouts/cost_evolution_plot.png")
    plt.close()
    print("[Saved Cost Evolution Plot to rollouts/cost_evolution_plot.png]")

    # Replay Final Best Plan
    print("\n[Replaying Final Best Plan on Fresh Environment]\n")
    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    final_steps = execute_plan_with_gif(final_env, final_best_plan, final_path, gif_name="rollouts/final_best_plan.gif")
    print("[Total Steps Taken]:", final_steps)

# ========== End ==========


if __name__ == "__main__":
    best_first_fortified_rollout_planner()
