from gpt_utils import extract_objects, format_env_state_for_gpt, get_gpt_plan
from env import SimpleEnv
from lookahead import PathPlanner
from minigrid.core.world_object import Door, Goal, Key, Wall, Box
import json
from openai import OpenAI
import matplotlib.pyplot as plt


class LLMPlanner():
    def __init__(self, env, model_name="gpt-4.1-nano"): # 4.1 nano similar results to gpt-4-turbo - much, much cheaper
        self.env = env
        self.path_planner = PathPlanner(env, None)
        self.model = model_name
        self.client = self._set_client()
    
    def load_api_key(self, path="api_key.txt"):
        with open(path, "r") as f:
            return f.read().strip()

    def _set_client(self):
        api_key = self.load_api_key()
        return OpenAI(
            api_key=api_key,
        )

    def _gen_subtasks_prompt(self):
        instr = (
            "Generate a valid, ordered list of high-level subtasks to complete the mission. "
            "Subtasks must use only the allowed actions and target named objects. "
            "Output exactly 8 subtasks as a JSON list in the format shown above. "
            "Do not include coordinates or explanations. Begin now."
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
                "subtask": [{"action": "Move to", "target": "green key"}]
            },
            "instructions": instr
        }
        prompt = json.dumps(prompt_data, indent=2)
        return prompt
    
    def _gen_actions_prompt(self):
        instr = (
            "Generate a sequence of actions to complete the mission. "
            "Actions must use only the allowed actions and target named objects. "
            "Do not include coordinates or explanations. Begin now."
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
                "actions": [{"action": "Move to", "target": "green key"}]
            },
            "instructions": instr
        }
        prompt = json.dumps(prompt_data, indent=2)
        return prompt
    
    def _get_objects(self):
        keys = {}
        boxes = {}
        goals = {}
        doors = {}
        walls = []

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

    def generate(self, task = "subtasks"):
        print("Generating", task)
        if task == "subtasks":
            prompt_text = self._gen_subtasks_prompt()
        elif task == "actions":
            prompt_text = self._gen_actions_prompt()

        if task == "subtasks":
            content = (
                "You are a high-level planning assistant for a robot navigating a grid world. "
                "Given a JSON object describing the agent's state, objects, and mission, "
                "generate 8 possible immediate subtasks from the current state. "
                "Only return valid JSON in this format:\n\n"
                "{\n  \"subtasks\": [\n    {\"action\": \"Move to\", \"target\": \"green key\"},\n    ...\n  ]\n}"
            )

        elif task == "actions":
            content = (
                "You are a high-level planning assistant for a robot navigating a grid world. "
                "Given a JSON object describing the agent's state, objects, and mission, and the completed subtask chain"
                "generate a sequence of actions to complete the mission. "
                "Only return valid JSON in this format:\n\n"
                "{\n  \"actions\": [\n    {\"action\": \"Move to\", \"target\": \"green key\"},\n    ...\n  ]\n}"
            )

        else:
            raise ValueError("Invalid task type. Use 'subtasks' or 'actions'.")

        messages = [
        {
            "role": "system",
            "content": content
        },
        {
            "role": "user",
            "content": f"```json\n{prompt_text}\n```"
        }
        ]
        # print(prompt_text)
        # sys.exit()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=256
        )

        # Extract JSON output safely
        raw_output = response.choices[0].message.content.strip()
        print(raw_output)


if __name__ == "__main__":

    '''
    Sample Output:
    Subtasks:
        {
        "subtasks": [
            {"action": "Move to", "target": "green key"},
            {"action": "Pick up", "target": "green key"},
            ...
        ]
        }

    Actions:
        {
        "actions": [
            {"action": "Move to", "target": "green key"},
            {"action": "Pick up", "target": "green key"},
            ...
        ]
        }
    '''

    env = SimpleEnv()
    obs, _ = env.reset()

    plt.imsave("initial_observation.png", obs["image"])

    llmp = LLMPlanner(env)

    llmp.generate("subtasks")
    llmp.generate("actions")

