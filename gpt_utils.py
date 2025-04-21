from openai import OpenAI
from env import SimpleEnv
from lookahead import PathPlanner
import matplotlib.pyplot as plt

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
                objects.append({"type": obj.type, "pos": (x, y)})
    return objects

def format_env_state_for_gpt(agent_pos, objects, goal_pos, mission):
    object_descriptions = "\n".join([
        f"- A {obj['type']} is at {obj['pos']}" for obj in objects
    ])
    return (
        "Agent and Environment Constraints: Agent can only carry one object at a time.\n"
        f"The agent is at position {agent_pos}.\n"
        f"{object_descriptions}\n"
        f"The goal is at {goal_pos}.\n"
        f"The agents mission is {mission}.\n"
        "Plan the steps the agent should take to solve the task."
    )

def get_gpt_plan(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": (
            "You are a high-level planning assistant for a robot navigating a maze. "
            "Given the agent's position, the position of objects (like keys, doors, boxes, diamonds), the goal, and the mission, "
            "generate a clear step-by-step plan that the agent should follow to reach the goal. "
            "Use short and clear instructions, e.g., 'Go to the key', 'Pick up the key', 'Open the door'."
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

def get_action_type(step_string):
    step_lower = step_string.lower()
    if "pick up" in step_lower:
        return "pickup"
    elif "put down" in step_lower or "drop" in step_lower:
        return "drop"
    elif "open" in step_lower:
        return "toggle"
    elif "close" in step_lower:
        return "toggle"
    else:
        return "move"

def get_target_position(plan_step, objects):
    for obj in objects:
        action_type = get_action_type(plan_step)
        if action_type == "drop":
            return (action_type, "")
        if obj['type'] in plan_step.lower() and action_type != "move":
            return (action_type, obj['type'])
        if obj['type'] in plan_step.lower():
            return obj['pos']
    return None

def execute_plan(env, plan, path):
    print("Executing high-level plan from GPT-4:")
    for step in plan:
        objects = extract_objects(env.grid)
        print("-", step)
        print("\t", env.carrying)
        print(env.actions.pickup)
        target_pos = get_target_position(step, objects)
        if target_pos is None:
            print("no path", target_pos)
        if isinstance(target_pos[0], str):
            action_map = {
                    'pickup': 4,
                    'drop': 5,
                    'toggle': 6
                }
            action = action_map.get(target_pos[0])
            print(action, target_pos[1])
            obs, _, _, _ , _ = env.step(action, target_pos[1])
            plt.imshow(obs["image"])
            plt.show()
        elif isinstance(target_pos, tuple):
            path.set_goal(target_pos)
            while(path.manhattan(env.agent_pos, target_pos) > 1):
                action = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _ , _ = env.step(action)
                plt.imshow(obs["image"])
                plt.show()

if __name__ == "__main__":
    env = SimpleEnv()
    obs, _ = env.reset()
    plt.imshow(obs["image"])
    plt.title("Initial Observation")
    # plt.show()
    path = PathPlanner(env, None)
    state = {
        'agent_pos': tuple(env.agent_pos),
        'agent_dir': env.agent_dir
    }
    grid_objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]

    prompt = format_env_state_for_gpt(state["agent_pos"], grid_objects, goal, mission)
    # print(prompt)
    plan = get_gpt_plan(prompt)
    print(plan)
    # plan = ["Okay, here's a plan for the agent to deliver the box to the goal:", 'Go to the box at (3, 1).', 'Pick up the box.', 'Go to the key at (3, 6).', 'Put down the box.', 'Pick up the key.', 'Go to the door at (5, 6).', 'Open the door.', 'Go to the box at (3, 6).', 'Pick up the box.', 'Go to the goal at (7, 5).', 'Put down the box.']
    # plan = plan[1:]
    # execute_plan(env, plan, path)
