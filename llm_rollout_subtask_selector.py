
from env import SimpleEnv
from lookahead import PathPlanner
from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan, execute_plan
import matplotlib.pyplot as plt
import copy

def get_subtasks(agent_pos, objects, goal, mission):
    prompt = f"""Given the agent is at {agent_pos}, objects: {[f"{obj['color']} {obj['type']}" for obj in objects]}, and the goal at {goal}, 
    list all possible subtasks the agent can do first to solve the mission: {mission}.
    Example subtasks: 'Pick up red key', 'Go to green box', 'Open the door', etc.
    Just list the subtasks.
    """
    subtask_list = get_gpt_plan(prompt)
    return subtask_list

def evaluate_subtask_policies(env, subtasks, objects, goal, mission):
    best_score = float('inf')
    best_policy = None
    best_subtask = None

    for subtask in subtasks:
        prompt = f"""The following subtask should be attempted first: {subtask}\n"""
        prompt += format_env_state_for_gpt(tuple(env.agent_pos), objects, goal, mission)
        plan = get_gpt_plan(prompt)
        env_copy = copy.deepcopy(env)
        path = PathPlanner(env_copy, None)
        score = execute_plan(env_copy, plan, path, render=False)
        if score < best_score:
            best_score = score
            best_policy = plan
            best_subtask = subtask
    return best_subtask, best_policy, best_score

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

        print(f"Selected Subtask: {best_subtask}")
        print(f"Current Best Plan: {overall_best_plan}")
        print(f"Score: {overall_best_score}")

    # Execute the best plan
    final_env = SimpleEnv()
    final_env.reset()
    final_path = PathPlanner(final_env, None)
    print("\nExecuting Final Best Plan:")
    final_steps = execute_plan(final_env, overall_best_plan, final_path)
    print("Total steps:", final_steps)

if __name__ == "__main__":
    fortified_rollout_planner()
