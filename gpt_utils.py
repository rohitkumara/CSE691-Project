import openai
from openai import OpenAI
from env import SimpleEnv
from lookahead import PathPlanner
import matplotlib.pyplot as plt
from minigrid.core.constants import COLOR_NAMES  # Add this import at the top


client = OpenAI(
    api_key="AIzaSyBWH0YSvquwUH3PFrh8cbjw1zUjnDGHaKU",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# client = OpenAI(
#     api_key="api_key",
# )


def extract_objects(grid):
    objects = []
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj:
                objects.append({"type": obj.type, "color": obj.color, "pos": (x, y), "obj": obj})
    return objects

def format_env_state_for_gpt(agent_pos, objects, goal_pos, mission):
    return (
        """Example:
            Agent starts at (2, 2)  
            - A box is at (3, 1)  
            - A green key is at (3, 6)  
            - A locked green door is at (5, 6)  
            - The goal is at (7, 5)

            Example: Possible subtasks:
            - Move to the box
            - Move to the green key
            - Move to the green door
            

            Mission: Deliver the box to the goal.

            Steps:
            1. Move to box  
            2. Pick up the box  
            3. Move to key
            4. Drop the box
            5. Pick up the key  
            6. Move to door 
            7. Open the door  
            8. Move to box  
            9. Drop the key  
            10. Pick up the box  
            11. Move to the goal 
            12. Drop the box

            ---

            """
        
        "Plan the steps the agent should take to solve the task."
    )

def get_gpt_plan(prompt, model="gemini-2.0-flash"):
    messages = [
        {"role": "system", "content": (
            "You are a high-level planning assistant for a robot navigating a maze. "
            "Given the agent's position, the position of objects (like keys, doors, boxes, diamonds), the goal, and the mission, "
            "generate a clear step-by-step plan that the agent should follow to reach the goal. "
            "Use short and clear instructions, e.g., 'Go to the key', 'Pick up the key', 'Open the door'."
            "Do not repeat the instruction. Just output the subtasks"
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
    elif "toggle" in step_lower:
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
            if color.lower() in plan_step.lower():
                # If color specified in command matches object color
                if color.lower() == obj['color'].lower():
                    color_match = True
                    break
                else:
                    continue
        if not any(color.lower() in plan_step.lower() for color in COLOR_NAMES):
            color_match = True
        if obj['type'] in plan_step.lower() and color_match and action_type != "move":
            return [action_type, obj]
        if obj['type'] in plan_step.lower() and color_match:
            return obj['pos']
    return None

def execute_plan(env, plan, path):
    print("Executing high-level plan from Gemini:")
    timesteps = 0
    for step in plan:
        objects = extract_objects(env.grid)
        print("-", step)
        print("\t", env.carrying)
        print(env.actions.pickup)
        target_pos = get_target_position(step, objects)
        print(target_pos)
        if target_pos is None:
            print("no path", target_pos)
        if isinstance(target_pos[0], str):
            action_map = {
                    'pickup': 4,
                    'drop': 5,
                    'toggle': 6
                }
            action = action_map.get(target_pos[0])
            # print(action, target_pos[1])
            obs, _, _, _ , _ = env.step(action, target_pos[1])
            timesteps += 1
            plt.imshow(obs["image"])
            plt.show()
        elif isinstance(target_pos, tuple):
            path.set_goal(target_pos)
            while(path.manhattan(env.agent_pos, target_pos) > 1):
                action = path.one_step_lookahead_with_rollout(path.extract_state(env))
                obs, _, _, _ , _ = env.step(action)
                timesteps += 1
                plt.imshow(obs["image"])
                plt.show()
    return timesteps

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
    grid_objects = extract_objects(env.grid)
    goal = obs["goal"]
    mission = obs["mission"]
    print(COLOR_NAMES)
    prompt = format_env_state_for_gpt(state["agent_pos"], grid_objects, goal, mission)
    # print(prompt)
    plan = get_gpt_plan(prompt)
    print(plan)
    # plan = ['Move to the box', 'Pick up the box', 'Move to the green key', 'Drop the box', 'Pick up the green key', 'Move to the green door', 'Open the door', 'Move to the box', 'Pick up the box', 'Move to the goal', 'Drop the box']
    plan = [i for i in plan if ":" not in i]
    timesteps = execute_plan(env, plan, path)
    print("timesteps:", timesteps)
