
from env import SimpleEnv
from lookahead import extract_state, ACTIONS, manhattan, simulate_action
import matplotlib.pyplot as plt

def extract_objects(grid):
    objects = []
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj:
                objects.append({"type": obj.type, "pos": (x, y)})
    return objects

def get_mock_gpt_plan():
    return [
        "Go to the key",
        "Pick up the key",
        "Go to the door",
        "Open the door",
        "Go to the goal"
    ]

def get_target_position(plan_step, objects):
    for obj in objects:
        if obj['type'] in plan_step.lower():
            return obj['pos']
    return None

def greedy_action_to_target(env, state, target_pos):
    best_action = None
    min_dist = float('inf')
    for action in ACTIONS:
        next_state, _, fail = simulate_action(env, state, action)
        if fail or next_state is None:
            continue
        dist = manhattan(next_state['agent_pos'], target_pos)
        if dist < min_dist:
            min_dist = dist
            best_action = action
    return best_action

def execute_plan(env, plan, objects, goal_pos):
    print("Executing mock high-level plan:")
    for step in plan:
        print("-", step)
        target_pos = get_target_position(step, objects)
        if "goal" in step.lower():
            target_pos = goal_pos
        if target_pos:
            for _ in range(15):  # Limit steps per sub-task
                state = extract_state(env)
                if manhattan(state['agent_pos'], target_pos) == 0:
                    break
                action = greedy_action_to_target(env, state, target_pos)
                if action is None:
                    print("No valid path to", target_pos)
                    break
                obs, _, _, _, _ = env.step(action)
                plt.imshow(obs["image"])
                plt.title(step)
                plt.show()

if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    obs, _ = env.reset()
    plt.imshow(obs["image"])
    plt.title("Initial Observation")
    plt.show()

    grid_objects = extract_objects(env.grid)
    goal = obs["goal"]

    plan = get_mock_gpt_plan()
    execute_plan(env, plan, grid_objects, goal)
