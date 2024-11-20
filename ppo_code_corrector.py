import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

# Initialize the LLM with your setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyCGPrTpCZbQca-IWqabKfBC4pYk9grF4JY"
)

def downloaded_code(download_folder="downloaded_code"):
    for filename in os.listdir(download_folder):
        file_path = os.path.join(download_folder, filename)
        with open(file_path, "r") as f:
            code = f.read()
        return code

def check_solutions_with_llm(code, context, error, solutions):
    template = PromptTemplate(
        input_variables=["code", "context", "error", "solutions"],
        template=(
            "You are given the following code:\n{code}\n\n"
            "This code implements the {context} algorithm and has an error: {error}."
            " Below are proposed solutions to fix this error. For each solution, indicate if it is correct or not."
            " If any solution is correct, return 'Yes' along with the fixed code; otherwise, return 'No'.\n\n"
            "Solutions:\n{solutions}\n"
            "Respond only with 'Yes' or 'No' and provide the corrected code if a solution is correct."
        )
    )

    solutions_text = "\n".join([f"{i+1}. {sol}" for i, sol in enumerate(solutions)])
    chain = template | llm
    response = chain.invoke({"code": code, "context": context, "error": error, "solutions": solutions_text})
    
    return response.content.strip()

class CodeCorrectionEnv(gym.Env):
    def __init__(self, error_data):
        super(CodeCorrectionEnv, self).__init__()
        self.error_data = error_data
        self.error_type = None
        self.current_solutions = None
        self.state = None
        self.done = False  # Track if the current episode is done

        # Observation and action space
        self.error_types = list(self.error_data.keys())
        self.observation_space = spaces.Discrete(len(self.error_types))
        self.action_space = spaces.Discrete(1)

    def reset(self):
        # Choose a random error type
        self.error_type = np.random.choice(self.error_types)
        self.current_solutions = self.error_data[self.error_type][0]['solutions']
        self.state = self.error_types.index(self.error_type)
        self.action_space = spaces.Discrete(len(self.current_solutions))
        self.done = False  # Reset the done flag
        return self.state

    def step(self, action):
        # Only process if the environment is not done
        if self.done:
            return self.state, 0, self.done, {}

        # Fetch the downloaded code
        code = downloaded_code()
        error_context = "Bubble Sort algorithm"
        error_message = "IndexError: list index out of range"
        solutions = self.current_solutions

        # Debug print all solutions
        for i, solution in enumerate(solutions):
            print(f"Solution {i+1}: {solution}\n")

        # Use LLM to check solutions
        result = check_solutions_with_llm(code, error_context, error_message, solutions)

        # Reward mechanism based on LLM's response
        if "yes" in result.lower():
            reward = 1
            self.done = True  # Mark the environment as done
            print(f"Correct solution found:\n{result}")
        else:
            reward = -1
            self.done = True  # Mark the environment as done
            print(f"No correct solution found. Result:\n{result}")

        return self.state, reward, self.done, {}

# Example dataset of error solutions
error_data = {
    "IndexOutOfBoundsError": [
        {
            "code": "for j in range(0, n-i-1):\n    if arr[j] > arr[j+1]:\n        arr[j], arr[j+1] = arr[j+1], arr[j]",
            "error": "IndexError: list index out of range",
            "solutions": [
                "Change loop range to range(0, n-1)",
                "Check if the index j+1 is within bounds",
                "Use n-i-1 instead of n in the loop condition"
            ]
        }
    ]
}
# Initialize the environment
env = CodeCorrectionEnv(error_data)

# Define and initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the PPO model
model.learn(total_timesteps=1000)  # Reduced training time for quick testing

# Reset the environment to get the initial observation
obs = env.reset()

# Predict the first action using the model
action, _states = model.predict(obs)

# Perform the action in the environment (step function will be called here)
obs, reward, done, info = env.step(action)

# Print the results
print(f"Action: {action}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Observation: {obs}")
print(f"Info: {info}")


"""
# Initialize the environment
env = CodeCorrectionEnv(error_data)

# Initialize the PPO model and train it
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, reset_num_timesteps=False)

# Test the trained PPO model
obs = env.reset()
done = False  # Flag to control whether to continue the loop

for _ in range(10):
    if done:
        print("Episode finished, resetting environment.")
        obs = env.reset()  # Reset the environment when done
        done = False  # Reset the done flag

    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    print(f"Action: {action}, Reward: {reward}, Done: {done}")
"""