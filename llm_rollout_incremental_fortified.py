
from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan
import matplotlib.pyplot as plt
import copy
import imageio
import re
from gpt_utils import client

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def get_gpt_subtasks(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": "List multiple subtasks clearly. One subtask per line."},
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

def get_subtasks(agent_pos, objects, goal, mission, prior_chain=None):
    if prior_chain is None:
        prompt = f"""You are an expert planner.
The agent is currently at {agent_pos}.
Objects: {[f"{obj['color']} {obj['type']} at {obj['pos']}" for obj in objects]}.
Goal: {goal}.
Mission: {mission}.

List at least 5 useful first subtasks the agent could attempt to achieve the mission.
Each subtask should be a short phrase like 'Pick up red key' or 'Move to blue box'.
"""
    else:
        prompt = f"""The agent has already completed the following subtasks: {prior_chain}.
Given the updated environment, suggest next subtasks to continue solving the mission: {mission}.
List at least 3 next subtasks, one per line.
"""
    print("\n[LLM Prompt for Subtasks]\n", prompt)
    return get_gpt_subtasks(prompt)

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
    return timesteps

def generate_plan_from_chain(subtask_chain, agent_pos, objects, goal, mission):
    prompt = f"""The agent must first perform these subtasks in order: {subtask_chain}.
Then continue generating steps to complete the full mission: {mission}.
Create a full high-level plan starting with the given subtasks.
"""
    prompt += format_env_state_for_gpt(agent_pos, objects, goal, mission)
    print("\n[LLM Prompt for Full Plan Generation]\n", prompt)
    return get_gpt_plan(prompt)

def fortified_incremental_rollout_planner():
    env = SimpleEnv()
    obs, _ = env.reset()
    objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]
    agent_pos = tuple(env.agent_pos)

    best_subtask_chain = []
    best_cost = float('inf')
    best_plan = []

    while True:
        subtasks = get_subtasks(agent_pos, objects, goal, mission, prior_chain=best_subtask_chain if best_subtask_chain else None)
        if not subtasks:
            break

        best_local_score = float('inf')
        best_local_subtask = None
        best_local_plan = None

        for subtask in subtasks:
            candidate_chain = best_subtask_chain + [subtask]
            plan = generate_plan_from_chain(candidate_chain, agent_pos, objects, goal, mission)
            env_copy = copy.deepcopy(env)
            path = PathPlanner(env_copy, None)
            rollout_cost = execute_plan_with_gif(env_copy, plan, path, gif_name=f"rollout_{sanitize_filename('_'.join(candidate_chain))}.gif")
            if rollout_cost == float('inf'):
                continue
            if rollout_cost < best_local_score:
                best_local_score = rollout_cost
                best_local_subtask = subtask
                best_local_plan = plan

        if best_local_plan and best_local_score < best_cost:
            best_subtask_chain.append(best_local_subtask)
            best_cost = best_local_score
            best_plan = best_local_plan
            print(f"\n[Accepted Extension]: {best_local_subtask}")
            print(f"[Current Best Subtask Chain]: {best_subtask_chain}")
            print(f"[Current Best Cost]: {best_cost}")
        else:
            print("\n[No better plan found. Ending expansion.]")
            break

    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    print("\n[Executing Final Best Plan:]")
    final_steps = execute_plan_with_gif(final_env, best_plan, final_path, gif_name="final_best_plan.gif")
    print("[Total Steps]:", final_steps)

if __name__ == "__main__":
    fortified_incremental_rollout_planner()
