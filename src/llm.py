import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

class GeminiModel:
    """
    Wrapper for Google's Gemini API to serve as the backbone for agents.
    """
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.7):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        
        # Rate limit handling
        self.last_request_time = 0
        # Free tier is ~15 RPM, so 4 seconds per request is safe
        self.min_interval = 4.0

    def generate(self, prompt, system_instruction=None):
        """
        Generate text based on a prompt and optional system instruction.
        Includes retry logic for rate limits.
        """
        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            # Enforce minimum interval
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            try:
                full_prompt = prompt
                if system_instruction:
                    full_prompt = f"System Instruction:\n{system_instruction}\n\nUser Task:\n{prompt}"
                
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature
                    )
                )
                
                self.last_request_time = time.time()
                
                if response.text:
                    return response.text
                else:
                    return "Error: Empty response from model."
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"\n[Warning] Rate limit hit. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    return f"Error generating content: {error_str}"
        
        return "Error: Failed to generate content after multiple retries due to rate limits."

    def generate_chat(self, history, new_message, system_instruction=None):
        pass
