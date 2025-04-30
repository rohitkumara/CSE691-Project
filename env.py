from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

ACTIONS = [0, 1, 2, 3] # left, right, up, down

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(2, 2),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.goal = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_pov = False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Deliver the box to the goal"

    def gen_obs(self):
        obs = super().gen_obs()
        rgb_img = self.get_frame(False) # set True to get agent POV as img obs
        return {**obs, "image": rgb_img, "goal": self.goal}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            # if(i==3):
            #     continue
            self.grid.set(4, i, Wall())
        
        # Place the door and key
        self.grid.set(4, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 8, Key(COLOR_NAMES[0]))
        self.grid.set(4, 2, Door(COLOR_NAMES[1], is_locked=True))
        self.grid.set(2, 5, Key(COLOR_NAMES[1]))
        self.grid.set(3, 1, Box(COLOR_NAMES[3]))
        self.grid.set(2, 6, Box(COLOR_NAMES[5]))
        # self.grid.set(1, 1, Ball(COLOR_NAMES[3]))
        # self.grid.set(1, 6, Ball(COLOR_NAMES[5]))

        # Place a goal square in the bottom-right corner
        g = Goal()
        g.color = COLOR_NAMES[5]
        self.put_obj(g, 8, 2)
        self.goal.append(((8, 2), g.color))
        g = Goal()
        g.color = COLOR_NAMES[3]
        self.put_obj(g, 8, 5)
        self.goal.append(((8, 5), g.color))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Deliver the box to the goal"

    def _move_agent(self, action):
        #   3
        # 2 a 0
        #   1
        curr_dir = self.agent_dir
        # left, right, up, down
        if(action == 0):
            # left
            if(curr_dir == 0):
                super().step(0)
                super().step(0)
            if(curr_dir == 1): super().step(1)
            if(curr_dir == 2): pass
            if(curr_dir == 3): super().step(0)
        if(action == 1):
            # right
            if(curr_dir == 0): pass
            if(curr_dir == 1): super().step(0)
            if(curr_dir == 2): 
                super().step(0)
                super().step(0)
            if(curr_dir == 3): super().step(1)
        if(action == 2):
            # up
            if(curr_dir == 0): super().step(0)
            if(curr_dir == 1): 
                super().step(0)
                super().step(0)
            if(curr_dir == 2): super().step(1)
            if(curr_dir == 3): pass
        if(action == 3):
            # down
            if(curr_dir == 0): super().step(1)
            if(curr_dir == 1): pass
            if(curr_dir == 2): super().step(0)
            if(curr_dir == 3): 
                super().step(0)
                super().step(0)
        return super().step(2)
    
    def get_actions(self):
        return ACTIONS

    def _drop(self):
        # Check all adjacent cells for empty space
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        goal_found = False
        for dx, dy in directions:
            nx = self.agent_pos[0] + dx
            ny = self.agent_pos[1] + dy
            cell = self.grid.get(nx, ny)
            if cell and cell.type == 'goal' and cell.color == self.carrying.color:
                self.grid.set(nx, ny, self.carrying)
                self.carrying = None
                goal_found = True
                break
        if not goal_found:
            for dx, dy in directions:
                nx = self.agent_pos[0] + dx
                ny = self.agent_pos[1] + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    cell = self.grid.get(nx, ny)
                    if cell is None:  # Found empty space
                        self.grid.set(nx, ny, self.carrying)
                        self.carrying = None
                        break

    def _toggle(self, obj):
        # print("toggling")
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in directions:
            nx = self.agent_pos[0] + dx
            ny = self.agent_pos[1] + dy
            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                # print(nx, ny)
                cell = self.grid.get(nx, ny)
                if isinstance(cell, Door) and cell.is_locked:
                    if obj and cell.type != obj.get('type'):  # safe access with .get()
                        continue
                    cell.toggle(self, self.agent_pos)
                    break

    def _pickup(self, obj):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if self.carrying is not None:
            # If already carrying something, drop it first
            old_item = self.carrying
            for dx, dy in directions:
                nx = self.agent_pos[0] + dx
                ny = self.agent_pos[1] + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    cell = self.grid.get(nx, ny)
                    if cell and cell.can_pickup() and cell.type == obj['type'] and cell.color == obj['color']:
                        self.carrying = cell
                        self.grid.set(nx, ny, old_item)
                        break
        else:
            for dx, dy in directions:
                nx = self.agent_pos[0] + dx
                ny = self.agent_pos[1] + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    cell = self.grid.get(nx, ny)
                    if cell and cell.can_pickup() and cell.type == obj['type'] and cell.color == obj['color']:
                        self.carrying = cell
                        self.grid.set(nx, ny, None)
                        break
    
    def step(self, action, obj=""):
        if(0 <= action and action <= 3):
            return self._move_agent(action)
        else:
            action = action-1
            if action == self.actions.pickup:
                self._pickup(obj)
            elif action == self.actions.drop:
                if self.carrying is not None:
                    self._drop()
            elif action == self.actions.toggle:
                self._toggle(obj)
            
        obs = self.gen_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = SimpleEnv(render_mode="human")
    env.reset()
    grid = env.grid
    objects = []
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj:
                objects.append(obj.color)
    print(objects)


    