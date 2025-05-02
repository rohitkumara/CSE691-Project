
from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan, execute_plan
import matplotlib.pyplot as plt
import copy

def display_step_with_text(image, step_desc):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(step_desc, fontsize=10)
    plt.axis('off')
    plt.show()

def get_subtasks(agent_pos, objects, goal, mission):
    prompt = f"""Given the agent is at {agent_pos}, objects: {[f"{obj['color']} {obj['type']}" for obj in objects]}, and the goal at {goal}, 
    list all possible subtasks the agent can do first to solve the mission: {mission}.
    Example subtasks: 'Pick up red key', 'Go to green box', 'Open the door', etc.
    Just list the subtasks.
    """
    print("\n[LLM Prompt for Subtasks]\n", prompt)
    subtask_list = get_gpt_plan(prompt)
    print("[Returned Subtasks]:", subtask_list)
    return subtask_list

def evaluate_subtask_policies(env, subtasks, objects, goal, mission):
    best_score = float('inf')
    best_policy = None
    best_subtask = None

    for subtask in subtasks:
        prompt = f"""The following subtask should be attempted first: {subtask}\n"""
        prompt += format_env_state_for_gpt(tuple(env.agent_pos), objects, goal, mission)
        print("\n[LLM Prompt for Policy Generation - Subtask:]", subtask)
        print(prompt)
        plan = get_gpt_plan(prompt)
        print("[Plan Generated]:", plan)
        env_copy = copy.deepcopy(env)
        path = PathPlanner(env_copy, None)
        score = execute_plan_with_overlay(env_copy, plan, path)
        if score < best_score:
            best_score = score
            best_policy = plan
            best_subtask = subtask
    return best_subtask, best_policy, best_score

def execute_plan_with_overlay(env, plan, path):
    print("Executing plan with overlay...")
    timesteps = 0
    from gpt_utils import get_target_position
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
            timesteps += 1
            display_step_with_text(obs["image"], f"Step: {step}")
        elif isinstance(target_pos, tuple):
            path.set_goal(target_pos)
            while path.manhattan(env.agent_pos, target_pos) > 1:
                action = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _, _ = env.step(action)
                timesteps += 1
                display_step_with_text(obs["image"], f"Step: {step}")
    return timesteps

def fortified_rollout_planner():
    env = SimpleEnv()
    obs, _ = env.reset()
    objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]
    agent_pos = tuple(env.agent_pos)

    subtasks = get_subtasks(agent_pos, objects, goal, mission)

    overall_best_score = float('inf')
    overall_best_plan = []

    while subtasks:
        best_subtask, candidate_plan, candidate_score = evaluate_subtask_policies(env, subtasks, objects, goal, mission)

        if candidate_score < overall_best_score:
            overall_best_score = candidate_score
            overall_best_plan = candidate_plan

        subtasks.remove(best_subtask)

        print(f"\n[Subtask Selected]: {best_subtask}")
        print(f"[Current Best Plan]: {overall_best_plan}")
        print(f"[Best Score]: {overall_best_score}")

    # Execute the best plan
    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    print("\n[Executing Final Best Plan:]")
    final_steps = execute_plan_with_overlay(final_env, overall_best_plan, final_path)
    print("[Total Steps]:", final_steps)

if __name__ == "__main__":
    fortified_rollout_planner()
