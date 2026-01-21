import yaml
from .llm import GeminiModel

class ConstitutionalAgent:
    def __init__(self, name, role_key, model_provider: GeminiModel, config_path="configs/agents.yaml", constitution_path="configs/constitution.yaml"):
        self.name = name
        self.model = model_provider
        self.lessons_learned = []  # List of strings: "I must be more concise."
        
        # Load Configs
        with open(config_path, 'r') as f:
            all_agents_config = yaml.safe_load(f)
            self.role_config = all_agents_config['agents'][role_key]
            
        with open(constitution_path, 'r') as f:
            self.constitution = yaml.safe_load(f)
            
    def _construct_system_prompt(self):
        """
        Builds the system prompt including Role, Constitution, and Lessons Learned.
        """
        # 1. Role Definition
        prompt = f"You are {self.name}, the {self.role_config['role']}.\n"
        prompt += f"Goal: {self.role_config['goal']}\n"
        prompt += f"Description: {self.role_config['description']}\n\n"
        
        # 2. Constitution (The Law)
        prompt += "CONSTITUTION (Core Principles you must obey):\n"
        for principle, details in self.constitution['principles'].items():
            prompt += f"- {principle.upper()}: {details['definition']}\n"
            
        # 3. Lessons Learned (Emergent Optimization)
        if self.lessons_learned:
            prompt += "\nLESSONS LEARNED (Your self-improvement notes):\n"
            for lesson in self.lessons_learned:
                prompt += f"- {lesson}\n"
                
        return prompt

    def act(self, conversation_history):
        """
        Propose a message based on history.
        """
        system_prompt = self._construct_system_prompt()
        
        # Format history for the agent
        context = "Current Conversation:\n"
        for msg in conversation_history:
            context += f"{msg['sender']}: {msg['content']}\n"
            
        task_prompt = f"{context}\n\n{self.name}, it is your turn. Generate your response. Be concise and helpful."
        
        response = self.model.generate(task_prompt, system_instruction=system_prompt)
        return response

    def critique(self, message_content, sender_name):
        """
        Evaluate another agent's message against the Constitution.
        Returns: {violation: bool, feedback: str}
        """
        # Construct Critique Prompt
        prompt = f"You are a Constitutional Critic. Evaluate the following message from {sender_name}.\n"
        prompt += f"Message: \"{message_content}\"\n\n"
        prompt += "Check against these Principles:\n"
        
        for principle, details in self.constitution['principles'].items():
            prompt += f"- {principle.upper()}: {details['violation_check']}\n"
            
        prompt += "\nTask:\n"
        prompt += "1. Identify if ANY principle is violated.\n"
        prompt += "2. If yes, specify which one and why.\n"
        prompt += "3. If no, say 'Compliant'.\n"
        prompt += "Format: 'VIOLATION: [Principle] - [Reason]' or 'COMPLIANT'."
        
        critique_response = self.model.generate(prompt)
        
        # Simple parsing
        if "VIOLATION:" in critique_response:
            return {"violation": True, "feedback": critique_response}
        else:
            return {"violation": False, "feedback": critique_response}

    def learn(self, feedback):
        """
        Update internal state based on feedback (Prompt Evolution).
        """
        # Summarize the feedback into a lesson
        prompt = f"Summarize this critique into a single short 'lesson learned' for yourself: {feedback}"
        lesson = self.model.generate(prompt)
        self.lessons_learned.append(lesson)
