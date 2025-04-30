import copy
from collections import defaultdict

class PathPlanner:
    def __init__(self, env, goal_pos):
        self.env = env
        self.goal_pos = goal_pos
        self.distances = defaultdict(lambda: float('inf'))  # For caching distances
        self.next_move = {}  # For caching optimal next moves
        
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

    def extract_state(self, env):
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
        self.distances.clear()
        self.next_move.clear()
        # Compute distances with dynamic programming
        self._compute_distances()

    def _compute_distances(self):
        # print("in compute distance")
        """Compute shortest path distances using dynamic programming"""
        # Initialize goal distance
        self.distances[self.goal_pos] = 0
        queue = [(0, self.goal_pos)]
        
        # Actions: up, right, down, left
        actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        while queue:
            # print(queue)
            dist, pos = queue.pop(0)
            
            # If we found a longer path, skip
            if dist > self.distances[pos]:
                continue
                
            # Check all neighboring cells
            for idx, (dx, dy) in enumerate(actions):
                next_pos = (pos[0] + dx, pos[1] + dy)
                
                # Check if move is valid
                if not self._is_valid_move(next_pos):
                    continue
                    
                new_dist = dist + 1
                
                # If we found a better path
                if new_dist < self.distances[next_pos]:
                    self.distances[next_pos] = new_dist
                    self.next_move[next_pos] = (pos[0] - next_pos[0], pos[1] - next_pos[1])
                    queue.append((new_dist, next_pos))
                    
            # Sort queue to process shorter paths first
            queue.sort()

    def _is_valid_move(self, pos):
        """Check if a position is valid to move to"""
        x, y = pos
        if not (0 <= x < self.env.grid.width and 0 <= y < self.env.grid.height):
            return False
        cell = self.env.grid.get(x, y)
        return cell is None or cell.can_overlap()

    def greedy_action(self, state):
        """Fallback greedy action using manhattan distance"""
        best_action = None
        min_dist = float('inf')
        
        # Try all four directions
        for action in range(4):
            next_state, _, fail = self.simulate_action(state, action)
            if fail or next_state is None:
                continue
            dist = self.manhattan(next_state['agent_pos'], self.goal_pos)
            if dist < min_dist:
                min_dist = dist
                best_action = action
        return best_action

    def one_step_lookahead_with_rollout(self, current_state):
        """Get next move from precomputed optimal policy"""
        current_pos = (int(current_state['agent_pos'][0]), int(current_state['agent_pos'][1]))
        
        # If we don't have a cached move, recompute distances
        if current_pos not in self.next_move:
            self._compute_distances()
        
        # print(current_pos, self.next_move)

        # If still no move found, use manhattan distance as fallback
        if current_pos not in self.next_move:
            return self.greedy_action(current_state)
            
        # Convert movement direction to action number
        dx, dy = self.next_move[current_pos]
        # ACTIONS = [0, 1, 2, 3] # left, right, up, down
        action_map = {
            (0, -1): 2,  # up
            (1, 0): 1,   # right
            (0, 1): 3,   # down
            (-1, 0): 0   # left
        }
        return action_map.get((dx, dy))