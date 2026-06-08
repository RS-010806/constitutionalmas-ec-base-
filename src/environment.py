from .agent import ConstitutionalAgent
from .llm import GeminiModel
from .logger import RunLogger
import termcolor


class Environment:
    def __init__(self, gemini_model: GeminiModel, logger: RunLogger | None = None, max_revision_rounds: int = 2):
        self.model = gemini_model
        self.agents = []
        self.history = []
        self.metrics = {
            "total_tokens": 0,
            "violations_detected": 0,
            "revisions_triggered": 0,
        }
        self.logger = logger
        self.max_revision_rounds = max_revision_rounds

    def _log(self, text: str):
        if self.logger:
            self.logger.log_line(text)

    def add_agent(self, name, role_key):
        agent = ConstitutionalAgent(name, role_key, self.model)
        self.agents.append(agent)

    def run_turn(self, agent_idx):
        """
        Executes one turn for the specified agent, including the Critique Loop.
        """
        active_agent = self.agents[agent_idx]
        peers = [a for a in self.agents if a != active_agent]

        header = f"\n--- {active_agent.name} is thinking ---"
        print(termcolor.colored(header, "cyan"))
        self._log(header)

        draft_message = active_agent.act(self.history)
        final_message = draft_message

        for revision_round in range(self.max_revision_rounds + 1):
            proposed_line = f"Proposed: {draft_message[:100]}..."
            print(proposed_line)
            self._log(proposed_line)

            violations = []
            critique_header = "--- Peer Critique in progress ---"
            print(termcolor.colored(critique_header, "yellow"))
            self._log(critique_header)
            for peer in peers:
                critique = peer.critique(draft_message, active_agent.name)
                if critique["violation"]:
                    violation_text = f"{peer.name}: {critique['feedback']}"
                    violations.append(violation_text)
                    alert = f"Violation found by {peer.name}!"
                    print(termcolor.colored(alert, "red"))
                    self._log(alert)

            if not violations:
                final_message = draft_message
                break

            self.metrics["violations_detected"] += len(violations)

            if revision_round >= self.max_revision_rounds:
                final_message = draft_message
                max_line = f"{active_agent.name} reached max revision rounds; committing last draft."
                print(termcolor.colored(max_line, "yellow"))
                self._log(max_line)
                break

            self.metrics["revisions_triggered"] += 1
            feedback_summary = " ".join(violations)

            learn_line = f"{active_agent.name} is learning from feedback..."
            print(termcolor.colored(learn_line, "magenta"))
            self._log(learn_line)
            active_agent.learn(feedback_summary)

            revising_line = f"{active_agent.name} is revising..."
            print(termcolor.colored(revising_line, "cyan"))
            self._log(revising_line)
            draft_message = active_agent.act(self.history)

        # 4. Commit to History
        self.history.append({"sender": active_agent.name, "content": final_message})
        final_header = f"Final Message from {active_agent.name}:"
        print(termcolor.colored(final_header, "green"))
        print(final_message)
        self._log(final_header)
        self._log(final_message)

    def run_simulation(self, task, max_turns=5):
        """
        Main loop.
        """
        # Inject task as user message
        user_line = f"TASK: {task}"
        self.history.append({"sender": "User", "content": user_line})
        self._log(f"User: {user_line}")

        for i in range(max_turns):
            # Simple round-robin for now, or could be dynamic
            agent_idx = i % len(self.agents)
            self.run_turn(agent_idx)

        return self.history, self.metrics
