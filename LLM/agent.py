import openai
import os
from datetime import datetime
from utils import *
import traceback


openai.api_key = "your_own_key"
openai.api_type = "your_own_type"
openai.api_base =  "your_own_base"  
openai.api_version = "your_own version"

class AgentLLM:
    def __init__(self, name: str, model_engine = "ssrgpt4", save_path ="/home/huangyan/.local/share/ov/pkg/isaac_sim-2023.1.1/standalone_examples/Chem_lab/LLM/log"):
        self._name = name
        self._model_engine = model_engine
        
        self.system_prompt = "You are a helpful AI assistant."
        self.conversation_log = []
        self.save_path = save_path
        
    
    def load_system_prompt_from_file(self, filepath: str):
        """
        Load the system prompt from a text file.
        :param filepath: str, the path to the file containing the system prompt.
        """
        with open(filepath, 'r') as f:
            self.system_prompt = f.read()
            
    def append_system_prompt_from_file(self, filepath: str):
        """
        Add new content to the system prompt from a text file.
        This function is mainly use to add functions document.
        """
        with open(filepath, 'r') as f:
            new_str = f.read()
        self.system_prompt += new_str
        
    def get_name(self):
        return self._name
    
    def generate_response(self, prompt, input_messages=None, retry_limit=3):
        attempts = 0
        while attempts < retry_limit:
            try:
                if input_messages is None:
                    input_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    prompt = input_messages[-1]["content"]
                
                response = openai.ChatCompletion.create(
                    engine = self._model_engine,
                    messages = input_messages,
                    max_tokens=1500,
                    temperature=0.7)
                message = response.choices[0].message.content
                self._append_to_log(prompt, message)
                self._save_conversation()
                print(f'{self._name}: Response has been generated successfully.')
                return message.strip()
            except openai.OpenAIError as e:
                attempts += 1
                print(f"Attempt {attempts}: An error occurred - {e}")

        print(f"All {retry_limit} retries failed.")
        return None

        
    def _append_to_log(self, prompt, response):
        """
        Append the prompt and response to the conversation log.
        :param prompt: str, the input text to the AI.
        :param response: str, the AI's response.
        """
        self.conversation_log.append({'prompt': prompt, 'response': response})
        
    def _save_conversation(self):
        """
        Save the conversation log to a text file.
        :param filepath: str, the path to the file where to save the conversation.
        """
        filepath = os.path.join(self.save_path, f"{self._name}_log.txt")
        
        # Clear the file
        with open(filepath, 'w') as f:
            pass
        
        with open(filepath, 'w') as f:
            for exchange in self.conversation_log:
                f.write(f"Prompt: {exchange['prompt']}\n")
                f.write(f"####################\n")
                f.write(f"Response: {exchange['response']}\n")
                f.write("\n")  # Add a newline for better readability
                f.write(f"####################\n\n")
                
    def exec_code(self, code: str, global_dict: dict):
        """
        Executes the given code string.

        Args:
        code (str): The code string to be executed.
        global_dict (dict): A dictionary containing global variables.
        
        Outputs:
        flag (bool): True if the code was executed successfully, False otherwise.
        Error message (str): The error message if the code was not executed successfully.

        If an exception occurs during the execution of the code, it prints the error message and returns None.
        """
        # preprocess code by replacing \\n with \n
        code = code.replace('  \\n  ', '\n').replace(' \\n ', '\n').replace('\\n', '\n')

        # if len(extract_scripts(code)) > 0:
        #     code = extract_scripts(code)[0]
        try:
            print(code)
            exec(code, global_dict)
            return True, ''
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"An error occurred: {e}")
            return False, error_traceback
        
                
                
if __name__ == "__main__":
    agent = AgentLLM("test_agent")
    print(agent.generate_response("Hello"))
