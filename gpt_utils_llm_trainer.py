# Enhanced version of gpt_utils.py with multi-plan support and debug printing

from openai import OpenAI
from env import SimpleEnv
from lookahead import PathPlanner
import matplotlib.pyplot as plt
from minigrid.core.constants import COLOR_NAMES
import copy

client = OpenAI(
    api_key="AIzaSyBQnPr3MhWnS5gx_QFHzNBu66GE9DgpJs0",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def extract_objects(grid):
    objects = []
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj:
                objects.append({"type": obj.type, "color": obj.color, "pos": (x, y)})
    return objects

def format_env_state_for_gpt(agent_pos, objects, goal_pos, mission):
    object_descriptions = "\n".join([
        f"- A {obj['color']} {obj['type']} is at {obj['pos']}" for obj in objects
    ])
    return (
        """You are an expert task planner for a grid world agent.
        
            Rules:
            - The agent can carry only one object at a time.
            - Only 1 object can be held; to pick up a new one, the current one must be dropped.
            - Actions must be 1-step executable (Move to (x, y), Pick up, Drop, Toggle).
            - No need to mention coordinates of each object in the plan.
            - Once the agent has the correct object and the goal is reachable, DO NOT list every tile step — just say:  \n            → “Move to the goal”

            ---
            """
        f"The agent is at position {agent_pos}.\n"
        f"{object_descriptions}\n"
        f"The goal is at {goal_pos}.\n"
        f"The agent's mission is {mission}.\n"
        "Plan the steps the agent should take to solve the task."
    )

def get_gpt_plan(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": (
            "You are a high-level planning assistant for a robot navigating a maze. "
            "Generate a clear step-by-step plan. Use short and clear instructions like 'Go to the key', 'Pick up the key'."
        )},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256
    )
    plan = response.choices[0].message.content
    return [line.strip().lstrip("1234567890. ") for line in plan.split("\n") if line.strip()]

def get_multiple_gpt_plans(prompt, model="gemini-2.0-flash", num_variants=5):
    messages = [
        {"role": "system", "content": (
            "You are a high-level planner. Generate a unique and valid plan that solves the agent's mission. "
            "Be concise and provide short action steps (e.g., 'Move to the box', 'Pick up the box')."
        )},
        {"role": "user", "content": prompt}
    ]

    responses = []
    for _ in range(num_variants):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=256
        )
        plan = response.choices[0].message.content
        steps = [line.strip().lstrip("1234567890. ") for line in plan.split("\n") if line.strip()]
        responses.append(steps)
    return responses

def get_action_type(step_string):
    step_lower = step_string.lower()
    if "pick up" in step_lower:
        return "pickup"
    elif "put down" in step_lower or "drop" in step_lower:
        return "drop"
    elif any(word in step_lower for word in ["open", "close", "toggle"]):
        return "toggle"
    else:
        return "move"

def get_target_position(plan_step, objects):
    for obj in objects:
        action_type = get_action_type(plan_step)
        if action_type == "drop":
            return (action_type, "")

        color_match = False
        for color in COLOR_NAMES:
            if color.lower() in plan_step.lower() and color.lower() == obj['color'].lower():
                color_match = True
                break
        if not any(color.lower() in plan_step.lower() for color in COLOR_NAMES):
            color_match = True

        if obj['type'] in plan_step.lower() and color_match and action_type != "move":
            return (action_type, obj['type'])
        if obj['type'] in plan_step.lower() and color_match:
            return obj['pos']
    return None

def execute_plan(env, plan, path):
    print("Executing high-level plan:")
    timesteps = 0
    for step in plan:
        objects = extract_objects(env.grid)
        print("-", step)
        target_pos = get_target_position(step, objects)
        print("  → Target:", target_pos)

        if target_pos is None:
            print("  [!] No valid target for:", step)
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
            plt.imshow(obs["image"])
            plt.show()
        elif isinstance(target_pos, tuple):
            path.set_goal(target_pos)
            while path.manhattan(env.agent_pos, target_pos) > 1:
                action = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _, _ = env.step(action)
                timesteps += 1
                plt.imshow(obs["image"])
                plt.show()
    return timesteps

def evaluate_plan(env, plan, path):
    env_copy = copy.deepcopy(env)
    try:
        return execute_plan(env_copy, plan, path)
    except:
        return 1e6

def select_best_plan(env, prompt, path, num_variants=5):
    plans = get_multiple_gpt_plans(prompt, num_variants=num_variants)
    best_score = float('inf')
    best_plan = None
    for i, plan in enumerate(plans):
        print(f"\nEvaluating Plan {i+1}: {plan}")
        score = evaluate_plan(env, plan, path)
        print(f" → Score: {score} steps")
        if score < best_score:
            best_score = score
            best_plan = plan
    return best_plan

if __name__ == "__main__":
    env = SimpleEnv()
    obs, _ = env.reset()
    plt.imshow(obs["image"])
    plt.title("Initial Observation")
    plt.show()
    path = PathPlanner(env, None)
    state = {
        'agent_pos': tuple(env.agent_pos),
        'agent_dir': env.agent_dir
    }
    objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]
    prompt = format_env_state_for_gpt(state['agent_pos'], objects, goal, mission)
    best_plan = select_best_plan(env, prompt, path, num_variants=5)
    print("\nBest Selected Plan:")
    for step in best_plan:
        print("-", step)
    print("\nExecuting Best Plan...")
    final_timesteps = execute_plan(env, best_plan, path)
    print("Final execution took", final_timesteps, "timesteps.")
