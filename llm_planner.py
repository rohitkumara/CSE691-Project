from gpt_utils import extract_objects
from env import SimpleEnv
from lookahead import PathPlanner
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
import json
from openai import OpenAI
import matplotlib.pyplot as plt


class LLMPlanner():
    def __init__(self, env, model_name="gpt-4.1-nano"):
        self.env = env
        self.path_planner = PathPlanner(env, None)
        self.model = model_name
        self.client = self._set_client()

    def load_api_key(self, path="api_key.txt"):
        with open(path, "r") as f:
            return f.read().strip()

    def _set_client(self):
        api_key = self.load_api_key()
        return OpenAI(api_key=api_key)

    def _get_objects(self):
        keys, boxes, goals, doors, walls = {}, {}, {}, {}, []
        grid = self.env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                if isinstance(obj, Key):
                    keys[obj.color] = (x, y)
                elif isinstance(obj, Box):
                    boxes[obj.color] = (x, y)
                elif isinstance(obj, Goal):
                    goals[obj.color] = (x, y)
                elif isinstance(obj, Door):
                    doors[obj.color] = {"position": (x, y), "locked": obj.is_locked}
                elif isinstance(obj, Wall):
                    walls.append((x, y))
        return keys, boxes, goals, doors, walls

    def _gen_subtasks_prompt(self, num_subtasks):
        instr = (
            f"Generate a valid, ordered list of {num_subtasks} high-level subtasks to complete the mission. "
            "Subtasks must use only the allowed actions and target named objects. "
            "Output exactly this number of subtasks as a JSON list. Do not include coordinates or explanations. Begin now."
        )
        keys, boxes, goals, doors, walls = self._get_objects()
        prompt_data = {
            "mission": self.env.mission,
            "agent": {
                "position": list(self.env.agent_pos),
                "inventory": self.env.carrying,
            },
            "objects": {
                "keys": {color: list(pos) for color, pos in keys.items()},
                "boxes": {color: list(pos) for color, pos in boxes.items()},
                "goals": {color: list(pos) for color, pos in goals.items()},
                "doors": {
                    color: {"position": list(data["position"]), "locked": data["locked"]}
                    for color, data in doors.items()
                },
                "walls": [list(coord) for coord in walls]
            },
            "rules": {
                "actions": ["Move to", "Pick up", "Toggle", "Drop"],
                "constraints": [
                    "Must toggle locked doors to pass",
                    "Need matching key to toggle locked door",
                    "Boxes must be dropped next to their matching goal"
                ]
            },
            "output_format": {
                "subtasks": [{"action": "Move to", "target": "green key"}]
            },
            "instructions": instr
        }
        return json.dumps(prompt_data, indent=2)

    def _gen_actions_prompt(self, prior_chain):
        instr = (
            "Generate a sequence of actions to complete the mission. "
            "Actions must use only the allowed actions and target named objects. "
            "Do not include coordinates or explanations. Begin now."
        )
        keys, boxes, goals, doors, walls = self._get_objects()
        prompt_data = {
            "mission": self.env.mission,
            "completed_subtasks": prior_chain or [],
            "agent": {
                "position": list(self.env.agent_pos),
                "inventory": self.env.carrying,
            },
            "objects": {
                "keys": {color: list(pos) for color, pos in keys.items()},
                "boxes": {color: list(pos) for color, pos in boxes.items()},
                "goals": {color: list(pos) for color, pos in goals.items()},
                "doors": {
                    color: {"position": list(data["position"]), "locked": data["locked"]}
                    for color, data in doors.items()
                },
                "walls": [list(coord) for coord in walls]
            },
            "rules": {
                "actions": ["Move to", "Pick up", "Toggle", "Drop"],
                "constraints": [
                    "Must toggle locked doors to pass",
                    "Need matching key to toggle locked door",
                    "Boxes must be dropped next to their matching goal"
                ]
            },
            "output_format": {
                "actions": [{"action": "Move to", "target": "green key"}]
            },
            "instructions": instr
        }
        return json.dumps(prompt_data, indent=2)

    def generate(self, task="subtasks", num_subtasks=8, prior_chain=None):
        if task == "subtasks":
            prompt_text = self._gen_subtasks_prompt(num_subtasks)
            content = (
                "You are a high-level planning assistant for a robot navigating a grid world. "
                "Given a JSON object describing the agent's state, objects, and mission, "
                f"generate {num_subtasks} possible immediate subtasks. "
                "Only return valid JSON in this format:\n\n"
                "{\n  \"subtasks\": [\n    {\"action\": \"Move to\", \"target\": \"green key\"},\n    ...\n  ]\n}"
            )
        elif task == "actions":
            prompt_text = self._gen_actions_prompt(prior_chain)
            content = (
                "You are a high-level planning assistant for a robot navigating a grid world. "
                "Given a JSON object describing the agent's state, objects, mission, and prior subtasks, "
                "generate a sequence of actions. "
                "Only return valid JSON in this format:\n\n"
                "{\n  \"actions\": [\n    {\"action\": \"Move to\", \"target\": \"green key\"},\n    ...\n  ]\n}"
            )
        else:
            raise ValueError("Invalid task type. Use 'subtasks' or 'actions'.")

        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": f"```json\n{prompt_text}\n```"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )

        return response.choices[0].message.content.strip()

    def _box_at_goal(self, env):
        for goal in env.goal:
            goal_x, goal_y = goal[0]
            color = goal[1]
            obj = env.grid.get(goal_x, goal_y)
            if not obj or obj.type != "box" or obj.color != color:
                return False
        return True
