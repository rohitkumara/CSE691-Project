from env import SimpleEnv, ACTIONS
import time, copy
import matplotlib.pyplot as plt

# TODO:
# implement 1 step lookahead - with rollout (heuristic manhattan distance)
    # fix reward when required

env = SimpleEnv()
obs, info = env.reset()

print(env.agent_pos)
plt.imshow(obs["image"])
plt.show()

door_pos = (5, 6)
key_pos = (3, 6)
box_pos = (3, 1)
goal_pos = obs["goal"]


# 1 step lookahead with rollout
def simulate_action(env, state, action):
    sim_env = copy.deepcopy(env)
    sim_env.agent_pos = state['agent_pos']
    sim_env.agent_dir = state['agent_dir']
    sim_env.carrying = state['carrying']
    sim_env.grid = copy.deepcopy(state['grid'])

    obs, _, terminated, truncated, _ = sim_env.step(action)
    if terminated or truncated:
        return None, float('inf'), True
    new_state = extract_state(sim_env)
    return new_state, manhattan(new_state["agent_pos"], find_goal(env)), False

def extract_state(env):
    return {
        'agent_pos': tuple(env.agent_pos),
        'agent_dir': env.agent_dir,
        'carrying': env.carrying,
        'grid': copy.deepcopy(env.grid),
    }

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def find_goal(env):
    return goal_pos

def greedy_action(env, state):
    best_action = None
    min_dist = float('inf')
    for action in ACTIONS:
        next_state, _, fail = simulate_action(env, state, action)
        if fail or next_state is None:
            continue
        dist = manhattan(next_state['agent_pos'], find_goal(env))
        # print("greedy", action, dist)
        if dist < min_dist:
            min_dist = dist
            best_action = action
    return best_action

def rollout(env, state, depth=15):
    # print("in rollout")
    total_cost = 0
    key_bonus = 0
    for _ in range(depth):
        # print(state['agent_pos'], find_goal(env))
        if manhattan(state['agent_pos'], find_goal(env)) == 0:
            return total_cost
        action = greedy_action(env, state)
        # print("action chosen by greedy:", action)
        if action is None:
            return float('inf')
        state, cost, fail = simulate_action(env, state, action)
        if fail:
            return float('inf')
        if state['carrying'] is not None and state['carrying'].type == 'key':
            print("picked up key")
            key_bonus = 3  # You can adjust this value
        total_cost += cost
        # print(total_cost)
    # print()
    return total_cost

def one_step_lookahead_with_rollout(env, current_state):
    # print("in 1-step lookahead")
    best_action = None
    best_score = float('inf')
    for action in ACTIONS:
        next_state, cost, fail = simulate_action(env, current_state, action)
        if fail or next_state is None:
            continue
        rollout_cost = rollout(env, next_state)
        score = cost + rollout_cost
        # print(action, score)
        if score < best_score:
            best_score = score
            best_action = action
    # print()
    return best_action

for _ in range(10):
    action = one_step_lookahead_with_rollout(env, extract_state(env))
    print(action)
    obs, _, _, _ , _ = env.step(action)
    plt.imshow(obs["image"])
    plt.show()
