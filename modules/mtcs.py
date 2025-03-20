import random
import math

# Basic Node class to represent each state in the tree
class Node:
    def __init__(self, prompt, parent=None):
        self.prompt = prompt
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_prompts = self.generate_child_prompts()  # Child prompts to simulate new states

    def generate_child_prompts(self):
        # Function to generate child prompts (based on current prompt or CoT)
        # You would replace this with the actual logic from your external file
        return [self.prompt + f" Step {i+1}" for i in range(5)]  # Example: append steps to simulate CoT

    def is_fully_expanded(self):
        return len(self.untried_prompts) == 0

    def best_child(self, exploration_weight=1.41):
        # UCB1 formula for balancing exploration and exploitation
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def add_child(self, prompt):
        # Create a new child node with the given prompt
        child_node = Node(prompt, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, value):
        # Update node statistics
        self.visits += 1
        self.value += value

# MCTS class
class MCTS:
    def __init__(self, llm_response_function, num_simulations=1000):
        self.num_simulations = num_simulations
        self.llm_response_function = llm_response_function  # LLM function to generate responses

    def select(self, node):
        # Select a node to expand using UCB1
        while not node.is_fully_expanded():
            if len(node.children) < len(node.untried_prompts):
                # There's an untried prompt we can expand
                return self.expand(node)
            else:
                # All prompts are tried, move to the best child
                node = node.best_child()
        return node

    def expand(self, node):
        # Add a new child to the node
        prompt = node.untried_prompts.pop(0)
        return node.add_child(prompt)

    def simulate(self, node):
        # Simulate the outcome of a prompt by querying the LLM
        # Assume the LLM gives a score (e.g., how confident or useful the response is)
        response = self.llm_response_function(node.prompt)
        return response  # Assuming LLM returns a reward score for the simulation

    def backpropagate(self, node, value):
        # Backpropagate the value up the tree
        while node is not None:
            node.update(value)
            node = node.parent

    def run(self, root_prompt):
        root = Node(root_prompt)
        for _ in range(self.num_simulations):
            node = self.select(root)
            value = self.simulate(node)
            self.backpropagate(node, value)

        # After all simulations, return the most visited child node
        return root.best_child(exploration_weight=0)  # Choose based on exploitation only

# Example function to simulate querying an LLM (you can replace with actual LLM API call)
def llm_response_function(prompt):
    # Simulating a random score from the language model
    # In practice, you'd query your LLM here for a response and derive a score
    print(f"Simulating LLM response for prompt: {prompt}")
    return random.random()  # Random score for simplicity

if __name__ == "__main__":
    # Example usage of MCTS with LLM responses
    mcts = MCTS(llm_response_function, num_simulations=100)
    best_action = mcts.run("Initial question or CoT prompt")
    print(f"Best action after MCTS: {best_action.prompt}")