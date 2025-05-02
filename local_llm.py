import requests
import json
import ollama

# Set up Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/v1/complete"  # Ollama's default HTTP API URL

def decompose_task(task: str) -> list:
    """
    Sends the task to Ollama and returns the list of subtasks.
    
    Args:
    - task: The full task as a natural language string (e.g., "Pick up the red key and open the blue door.")
    
    Returns:
    - List of subtasks (strings).
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Payload for Ollama's API
    data = {
        "model": "mistral",  # Change to another model if you prefer
        "prompt": f"Break this down into gridworld subtasks:\n{task}\nSubtasks:",
        "max_tokens": 150  # Limit to ensure a concise response
    }
    
    # Sending the POST request to Ollama API
    response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        # Extracting subtasks from the response
        result = response.json()
        subtasks = result.get("completion", "").split("\n")
        return [subtask.strip() for subtask in subtasks if subtask.strip()]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Example Usage
if __name__ == "__main__":
    # task = "Pick up the red key, unlock the blue door, and enter the room."
    # subtasks = decompose_task(task)
    # print("Subtasks:", subtasks)

    result = ollama.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": "Move to the red key, pick it up, and unlock the blue door."}
    ]
)
print(result)