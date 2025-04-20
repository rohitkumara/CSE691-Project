import copy

class PathPlanner:
    def __init__(self, env, goal_pos):
        self.env = env
        self.goal_pos = goal_pos
        
    def simulate_action(self, state, action):
        sim_env = copy.deepcopy(self.env)
        sim_env.agent_pos = state['agent_pos']
        sim_env.agent_dir = state['agent_dir']
        sim_env.carrying = state['carrying']
        sim_env.grid = copy.deepcopy(state['grid'])

        obs, _, terminated, truncated, _ = sim_env.step(action)
        if terminated or truncated:
            return None, float('inf'), True
        new_state = self.extract_state(sim_env)
        return new_state, self.manhattan(new_state["agent_pos"], self.goal_pos), False

    @staticmethod
    def extract_state(env):
        return {
            'agent_pos': tuple(env.agent_pos),
            'agent_dir': env.agent_dir,
            'carrying': env.carrying,
            'grid': copy.deepcopy(env.grid),
        }

    @staticmethod
    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def set_goal(self, goal_pos):
        self.goal_pos = goal_pos

    def greedy_action(self, state):
        best_action = None
        min_dist = float('inf')
        for action in self.env.get_actions():
            next_state, _, fail = self.simulate_action(state, action)
            if fail or next_state is None:
                continue
            dist = self.manhattan(next_state['agent_pos'], self.goal_pos)
            if dist < min_dist:
                min_dist = dist
                best_action = action
        return best_action

    def rollout(self, state, depth=15):
        total_cost = 0
        key_bonus = 0
        for _ in range(depth):
            if self.manhattan(state['agent_pos'], self.goal_pos) == 0:
                return total_cost
            action = self.greedy_action(state)
            if action is None:
                return float('inf')
            state, cost, fail = self.simulate_action(state, action)
            if fail:
                return float('inf')
            if state['carrying'] is not None and state['carrying'].type == 'key':
                print("picked up key")
                key_bonus = 3  # You can adjust this value
            total_cost += cost
        return total_cost

    def one_step_lookahead_with_rollout(self, current_state):
        best_action = None
        best_score = float('inf')
        for action in self.env.get_actions():
            next_state, cost, fail = self.simulate_action(current_state, action)
            if fail or next_state is None:
                continue
            rollout_cost = self.rollout(next_state)
            score = cost + rollout_cost
            if score < best_score:
                best_score = score
                best_action = action
        return best_action