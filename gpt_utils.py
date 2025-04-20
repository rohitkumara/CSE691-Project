
# import openai

# overwrite_cache = True
# if overwrite_cache:
#   LLM_CACHE = {}

# #@title LLM Scoring

# def state_to_prompt(state, goal):
#     pos = state['agent_pos']
#     carrying = state['carrying'].type if state['carrying'] else "nothing"
#     return (
#         f"The agent is at position {pos}, facing direction {state['agent_dir']}, "
#         f"and is carrying {carrying}. The goal is at {goal}.\n"
#         "What should the agent do next?\n"
#     )


# def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
#               logprobs=1, echo=False):
#   full_query = ""
#   for p in prompt:
#     full_query += p
#   id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
#   if id in LLM_CACHE.keys():
#     print('cache hit, returning')
#     response = LLM_CACHE[id]
#   else:
#     response = openai.Completion.create(engine=engine, 
#                                         prompt=prompt, 
#                                         max_tokens=max_tokens, 
#                                         temperature=temperature,
#                                         logprobs=logprobs,
#                                         echo=echo)
#     LLM_CACHE[id] = response
#   return response

# def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
#   if limit_num_options:
#     options = options[:limit_num_options]
#   verbose and print("Scoring", len(options), "options")
#   gpt3_prompt_options = [query + option for option in options]
#   response = gpt3_call(
#       engine=engine, 
#       prompt=gpt3_prompt_options, 
#       max_tokens=0,
#       logprobs=1, 
#       temperature=0,
#       echo=True,)
  
#   scores = {}
#   for option, choice in zip(options, response["choices"]):
#     tokens = choice["logprobs"]["tokens"]
#     token_logprobs = choice["logprobs"]["token_logprobs"]

#     total_logprob = 0
#     for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
#       print_tokens and print(token, token_logprob)
#       if option_start is None and not token in option:
#         break
#       if token == option_start:
#         break
#       total_logprob += token_logprob
#     scores[option] = total_logprob

#   for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
#     verbose and print(option[1], "\t", option[0])
#     if i >= 10:
#       break

#   return scores, response

# def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
#   if not pick_targets:
#     pick_targets = PICK_TARGETS
#   if not place_targets:
#     place_targets = PLACE_TARGETS
#   options = []
#   for pick in pick_targets:
#     for place in place_targets:
#       if options_in_api_form:
#         option = "robot.pick_and_place({}, {})".format(pick, place)
#       else:
#         option = "Pick the {} and place it on the {}.".format(pick, place)
#       options.append(option)

#   options.append(termination_string)
#   print("Considering", len(options), "options")
#   return options
     
# query = "To pick the blue block and put it on the red block, I should:\n"
# options = make_options(PICK_TARGETS, PLACE_TARGETS)
# scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)
     


import openai
from env import SimpleEnv
from lookahead import extract_state, ACTIONS, manhattan, simulate_action
import matplotlib.pyplot as plt

openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key

def extract_objects(grid):
    objects = []
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj:
                objects.append({"type": obj.type, "pos": (x, y)})
    return objects

def format_env_state_for_gpt(agent_pos, objects, goal_pos):
    object_descriptions = "\n".join([
        f"- A {obj['type']} is at {obj['pos']}" for obj in objects
    ])
    return (
        f"The agent is at position {agent_pos}.\n"
        f"{object_descriptions}\n"
        f"The goal is at {goal_pos}.\n"
        "Plan the steps the agent should take to solve the task."
    )

def get_gpt_plan(prompt, model="gpt-4"):
    messages = [
        {"role": "system", "content": (
            "You are a high-level planning assistant for a robot navigating a maze. "
            "Given the agent's position, the position of objects (like keys, doors, boxes, diamonds), and the goal, "
            "generate a clear step-by-step plan that the agent should follow to reach the goal. "
            "Use short and clear instructions, e.g., 'Go to the key', 'Pick up the key', 'Open the door'."
        )},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256
    )

    plan = response["choices"][0]["message"]["content"]
    return [line.strip().lstrip("1234567890. ") for line in plan.split("\n") if line.strip()]

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
    print("Executing high-level plan from GPT-4:")
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

    state = {
        'agent_pos': tuple(env.agent_pos),
        'agent_dir': env.agent_dir
    }
    grid_objects = extract_objects(env.grid)
    goal = obs["goal"]

    prompt = format_env_state_for_gpt(state["agent_pos"], grid_objects, goal)
    plan = get_gpt_plan(prompt)
    execute_plan(env, plan, grid_objects, goal)
