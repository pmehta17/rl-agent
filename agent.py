from utils import Condition

import pickle
from copy import deepcopy
import random


class Agent:
    def __init__(self, game):
        """
        Initializes the agent with the given game instance.

        Args:
            game: An instance of the game that the agent will interact with.
        """
        self.game = game
        self.rows = game.rows
        self.cols = game.cols

    def play(self):
        """
        Executes the game loop for the agent.

        The agent continuously observes the game state, determines the next action,
        and performs the action until the game reaches a terminal condition.

        Returns:
            goal_test (Condition): The final state of the game, indicating whether
                                   the game is still in progress, won, or reveal a bomb.
        """
        raise NotImplementedError()

    def get_neighbors(self, x, y):
        """
        Get the neighboring coordinates of a given cell in a board.

        Args:
            x (int): The x-coordinate of the cell.
            y (int): The y-coordinate of the cell.

        Returns:
            list of tuple: A list of tuples representing the coordinates of the neighboring cells.
                           Only includes neighbors that are within the bounds of the board.
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < self.game.rows and 0 <= y + dy < self.game.cols]


class ManualGuiAgent(Agent):
    def __init__(self, game):
        super().__init__(game)

    def play(self):
        pass


class QLearningAgent(Agent):
    def __init__(self, game, q_table_path="", alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        """
        Initializes the Q-learning agent with the given game instance.
         - alpha: Learning rate
         - gamma: Discount factor
         - epsilon: Exploration rate
         - epsilon_decay: Decay factor
         - min_epsilon: Minimum exploration
        """
        super().__init__(game)
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor
        self.min_epsilon = min_epsilon  # Minimum exploration
        # Initialize Q-table
        self.Q = {}  # Dictionary for state-action values
        if q_table_path:
            with open(q_table_path, "rb") as f:
                self.Q = pickle.load(f)

    def state_to_tuple(self, state):
        """
        Converts the given state to a tuple for hashing in the Q-table.
        """
        def convert(cell):
            return cell.value if hasattr(cell, 'value') else cell
    
        state_tuple = tuple(tuple(convert(cell) for cell in sublist) for sublist in state)
        return state_tuple

    def get_possible_actions(self, state):
        """
        Returns a list of valid actions (cell coordinates) based on the current state.
        This implementation assumes that if a cell is an object with a 'revealed' attribute,
        it is hidden if cell.revealed is False. Otherwise, if the cell is None or equals "H"
        (as an example marker), it is considered hidden.
        """
        actions = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell = state[i][j]
                if hasattr(cell, "revealed"):
                    if not cell.revealed:
                        actions.append((i, j))
                else:
                    if cell is None or (isinstance(cell, str) and cell == "H"):
                        actions.append((i, j))
        return self.game.actions(state)

    def get_action(self, state):
        """
        Chooses an action using epsilon-greedy based on the given state.

        Args:
            state: The current state of the game.
        
        Returns:
            Action: The chosen action.
        """
        # TODO: Implement the epsilon-greedy action selection, using self.epsilon to switch strategy
        possible_actions = self.get_possible_actions(state)
        if not possible_actions:
            return None
        # Exploration: choose a random action with probability epsilon.
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            state_tuple = self.state_to_tuple(state)
            best_action = None
            best_value = -float('inf')
            for action in possible_actions:
                a_key = self.action_to_tuple(action)
                q_value = self.Q.get((state_tuple, a_key), 0)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            return best_action if best_action is not None else random.choice(possible_actions)

    def action_to_tuple(self, action):
        return (action.action_type, action.x, action.y)

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the given state, action, reward, and next state.

        Args:
            state (tuple): The current state of the game.
            action (Action): The action taken in the current state.
            reward (int): The reward received after taking the action.
            next_state (tuple): The resulting state after taking the action.
        """
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)
        # TODO: Implement the Q-learning update rule, using the state_tuple as the key for the Q-table
        a_key = self.action_to_tuple(action)
        current_q = self.Q.get((state_tuple, a_key), 0)
        # Use the game’s actions(state) method for valid next actions.
        possible_actions_next = self.get_possible_actions(next_state)
        if possible_actions_next:
            max_future_q = max(
                self.Q.get((next_state_tuple, self.action_to_tuple(a)), 0)
                for a in possible_actions_next
            )
        else:
            max_future_q = 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.Q[(state_tuple, a_key)] = new_q

    def train(self, episodes, save_path=""):
        """
        Trains the agent using Q-learning.

        Args:
            episodes (int): The number of episodes to train the agent.

        Returns:
            dict: The Q-table containing the state-action
                  values learned during training.
        """
        # Training loop
        print("Training Q-learning agent.")
        for _ in range(episodes):
            state = self.game.reset()
            condition = Condition.IN_PROGRESS
            while condition == Condition.IN_PROGRESS:
                old_state = deepcopy(state)
                action = self.get_action(state)
                next_state, condition, reward = self.game.step(action)
                self.update_q_table(old_state, action, reward, next_state)
                state = next_state
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # Save the learned Q-table to file
        if save_path != "":
            with open(save_path, "wb") as f:
                pickle.dump(self.Q, f)
            print("Training complete! Q-table saved to", save_path)
        return self.Q

    def play(self):
        """
        Plays the game using the learned Q-table.

        Returns:
            Condition: The final state of the game, indicating whether
                       the game is still in progress, won, or reveal a bomb.
        """
        print("Playing Minesweeper using Q-learning agent.")
        state = self.game.reset()
        condition = Condition.IN_PROGRESS
        while condition == Condition.IN_PROGRESS:
            state_tuple = self.state_to_tuple(state)

            # TODO: Implement action selection for testing, choose the best action (do not need exploration)
            print("Playing Minesweeper using Q-learning agent.")
            state = self.game.reset()
            condition = Condition.IN_PROGRESS
            while condition == Condition.IN_PROGRESS:
                state_tuple = self.state_to_tuple(state)
                possible_actions = self.get_possible_actions(state)
                if not possible_actions:
                    break
                best_action = None
                best_value = -float('inf')
                for action in possible_actions:
                    a_key = self.action_to_tuple(action)
                    q_value = self.Q.get((state_tuple, a_key), 0)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                if best_action is None:
                    best_action = random.choice(possible_actions)
                state, condition, _ = self.game.step(best_action)
            return condition

