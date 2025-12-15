import numpy as np
import math
from vgc2.battle_engine import State
from .OfflineSimulator import OfflineSimulator
from vgc2.battle_engine import BattleCommand

class Node:
    def __init__(self, simulator: OfflineSimulator = None, args = None, parent=None, action_taken=None, prior=0):
        self.simulator = simulator
        self.args = args
        self.parent: Node = parent
        self.action_taken = action_taken
        self.children: list[Node] = []
        self.children_actions: list[BattleCommand] = [] # so I don't have to retrieve all child actions every time
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def is_fully_expanded(self, state: State):
        return len(set(self.simulator.get_valid_actions(state)) - set(self.children_actions)) == 0

    def select(self, copied_state: State, reward: int, terminated: bool):
        best_child: Node = None
        best_ucb = -np.inf
        
        # take care to only select children which contain valid move
        # for the given state
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        state, tmp_reward, terminated = self.simulator.step(copied_state, best_child.action_taken)
        reward += tmp_reward

        return best_child, state, reward, terminated
    
    def get_ucb(self, child):
        if child.visit_count  == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count

        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        for action_ind, prob in enumerate(policy):
            if prob > 0:
                action_to_expand_on = self.simulator.possible_action_spaces[self.simulator.max_action_space_key][action_ind]
                child = Node(self.simulator, self.args, self, action_to_expand_on, prob)
                self.children.append(child)
                self.children_actions.append(action_to_expand_on)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)  