from env import SimpleEnv, ACTIONS
import time, copy
import matplotlib.pyplot as plt
from lookahead import PathPlanner

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

pp = PathPlanner(env, goal_pos)

for i in range(10):
    # change goal after 6 and 8 steps
    if(i==6):
        pp.set_goal(key_pos)
    if(i==8):
        pp.set_goal(box_pos)
    action = pp.one_step_lookahead_with_rollout(pp.extract_state(env))
    print(i, "goal pos:", pp.goal_pos)
    obs, _, _, _ , _ = env.step(action)
    plt.imshow(obs["image"])
    plt.show()