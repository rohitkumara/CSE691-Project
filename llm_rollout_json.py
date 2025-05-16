import json
import os
import copy
import time
import csv
import imageio
import matplotlib.pyplot as plt
from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import get_action_type, extract_objects
from llm_planner import LLMPlanner

def save_plan_text(plan_steps, save_path, depth=None, child=None, mission=None, prior_chain=None):
    with open(save_path, 'w') as f:
        f.write("=== PLAN DETAILS ===\n")
        if depth is not None and child is not None:
            f.write(f"Depth: {depth} | Child: {child}\n")
        if mission:
            f.write(f"Mission: {mission}\n")
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
        cost_str = f"{rollout_cost:.2f}" if rollout_cost != float('inf') else "INF (Failed)"
        f.write(f"Depth {depth} | Child {child} | Cost: {cost_str}\n")
        f.write(f"Plan File: {plan_path}\n")
        f.write(f"GIF File: {gif_path}\n")
        f.write("------------------------------------------------------\n")

def parse_and_execute_plan(env, plan_json, gif_name="rollouts/final_execution.gif"):
    path = PathPlanner(env, None)
    frames = []
    timesteps = 0
    stuck_counter = 0
    unchanged_counter = 0
    last_pos = tuple(env.agent_pos)
    start_time = time.time()

    for step in plan_json:
        action = get_action_type(step["action"])
        target_name = step["target"].lower()
        objects = extract_objects(env.grid)

        target = None
        for obj in objects:
            if obj["type"] in target_name and obj["color"] in target_name:
                target = obj
                break
        if "goal" in target_name:
            for gpos, gcolor in env.goal:
                if gcolor in target_name:
                    target = gpos
                    break

        if isinstance(target, dict):
            action_map = {"pickup": 4, "drop": 5, "toggle": 6}
            act_id = action_map[action]
            obs, _, _, _, _ = env.step(act_id, target)
        elif isinstance(target, tuple):
            path.set_goal(target)
            while path.manhattan(env.agent_pos, target) > 1:
                if unchanged_counter >= 5:
                    print("[Agent stuck. Aborting subtask execution.]")
                    stuck_counter += 1
                    imageio.mimsave(gif_name, frames, fps=2)
                    return timesteps, stuck_counter, time.time() - start_time
                act = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _, _ = env.step(act)
                frames.append(obs["image"])
                timesteps += 1
                if tuple(env.agent_pos) == last_pos:
                    unchanged_counter += 1
                else:
                    unchanged_counter = 0
                last_pos = tuple(env.agent_pos)

        obs = env.gen_obs()
        frames.append(obs["image"])
        timesteps += 1

    imageio.mimsave(gif_name, frames, fps=2)
    return timesteps, stuck_counter, time.time() - start_time

def best_first_rollout():
    os.makedirs("rollouts", exist_ok=True)
    env = SimpleEnv()
    obs, _ = env.reset()
    planner = LLMPlanner(env)
    plt.imsave("rollouts/initial_obs.png", obs["image"])

    current_env = copy.deepcopy(env)
    current_chain = []
    best_score = float('inf')
    final_best_plan = None
    iteration = 0
    max_depth = 30
    cost_tolerance = 3

    cost_evolution = []
    csv_log = []
    summary_path = "rollouts/plans_summary.txt"
    open(summary_path, 'w').close()

    while iteration < max_depth:
        num_subtasks = 8 if iteration == 0 else (6 if iteration == 1 else 3)
        subtasks_json = planner.generate("subtasks", num_subtasks=num_subtasks, prior_chain=current_chain)
        try:
            subtasks_data = json.loads(subtasks_json)
            subtasks = subtasks_data.get("subtasks", [])
        except Exception as e:
            print("[Error parsing subtasks JSON]", e)
            break

        best_rollout_cost = float('inf')
        best_subtask = None
        best_plan = None
        best_metrics = (0, 0, 0)

        for i, subtask in enumerate(subtasks):
            trial_env = copy.deepcopy(current_env)
            planner.env = trial_env
            planner._apply_subtask(subtask)
            full_plan_json = planner.generate("actions", prior_chain=current_chain + [subtask])
            try:
                full_plan_data = json.loads(full_plan_json)
                full_plan = full_plan_data.get("actions", [])
            except Exception as e:
                print("[Error parsing full plan JSON]", e)
                continue

            trial_env_copy = copy.deepcopy(trial_env)
            path = PathPlanner(trial_env_copy, None)
            rollout_gif = f"rollouts/rollout_iter{iteration}_child{i+1}.gif"
            cost, stucks, runtime = parse_and_execute_plan(trial_env_copy, full_plan, gif_name=rollout_gif)

            if cost < best_rollout_cost:
                best_rollout_cost = cost
                best_subtask = subtask
                best_plan = full_plan
                best_metrics = (stucks, runtime, cost)

            save_plan_summary(summary_path, iteration, i+1, cost, f"plan_iter{iteration}_child{i+1}.txt", rollout_gif)

        if best_subtask is None:
            print("[No valid subtask found. Terminating.]")
            break

        if best_score != float('inf') and best_rollout_cost > best_score + cost_tolerance:
            print(f"[Stopping: cost worsened from {best_score} to {best_rollout_cost}]")
            break

        if best_rollout_cost < best_score:
            best_score = best_rollout_cost
            final_best_plan = current_chain + best_plan

        current_chain.append(best_subtask)
        planner._apply_subtask(current_env, best_subtask)
        cost_evolution.append(best_score)
        csv_log.append((iteration, best_subtask, *best_metrics))

        if planner._box_at_goal(current_env):
            print("[Goal reached successfully!]")
            break

        iteration += 1

    if final_best_plan:
        print("\n[Executing Final Best Plan]\n")
        env_replay = SimpleEnv()
        env_replay.reset()
        parse_and_execute_plan(env_replay, final_best_plan, gif_name="rollouts/final_best_plan.gif")

    with open("rollouts/plans_cost_log.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Depth", "Subtask", "StuckCount", "Runtime", "Cost"])
        for row in csv_log:
            writer.writerow(row)

    plt.figure(figsize=(8,5))
    plt.plot(range(len(cost_evolution)), cost_evolution, marker='o')
    plt.title('Rollout Cost vs Planning Depth')
    plt.xlabel('Planning Step (Depth)')
    plt.ylabel('Rollout Cost')
    plt.grid(True)
    plt.savefig("rollouts/cost_evolution_plot.png")
    plt.close()

if __name__ == "__main__":
    best_first_rollout()
