from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

ACTIONS = [0, 1, 2, 3, 4, 5, 6] # left, right, up, down, pickup, drop, toggle

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

        self.goal = (size-3,size-5)

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
        return "grand mission"

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
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))
        self.grid.set(3, 1, Box(COLOR_NAMES[3]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.goal[0], self.goal[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

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

    def step(self, action):
        if(0 <= action and action <= 3):
            return self._move_agent(action)
        else:
            return super().step(action)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = SimpleEnv(render_mode="human")

    action_list = [4, 3, 2, 1, 0]
    obs, info = env.reset()
    plt.imshow(obs["image"])
    plt.show()


    