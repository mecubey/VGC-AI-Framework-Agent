import numpy as np
import torch
from .Node import Node
from vgc2.util.forward import copy_state
from .OfflineSimulator import OfflineSimulator
from .ResNet import ResNet

class MCTS:
    def __init__(self, simulator: OfflineSimulator, args, model: ResNet):
        self.simulator = simulator
        self.args = args
        self.model = model

    def get_discounted_return(self, value_hist: list[float]):
        R = 0
        for i in range(len(value_hist)):
            R += self.args['gamma']**i * value_hist[i]
        return R

    @torch.no_grad()
    def search(self, state):
        root = Node(self.simulator, self.args)
        root.visit_count = 1
        
        copied_state = copy_state(state)
        self.simulator.fill_hidden_information(copied_state)
        policy, _ = self.model(torch.tensor(self.simulator.get_encoded_state(copied_state)).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1-self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                * np.random.dirichlet([self.args['dirichlet_alpha']] * self.simulator.max_action_size)
        valid_actions = self.simulator.get_valid_actions(copied_state)
        self.simulator.apply_action_mask_to_numpy(policy, valid_actions)
        policy /= np.sum(policy)
        root.expand(policy)

        num_searches = 0
        match len(valid_actions):
            case 64:
                num_searches = 200
            case 16:
                num_searches = 300
            case 8:
                num_searches = 300
            case 4:
                num_searches = 300

        for _ in range(num_searches):
            node = root
            copied_state = copy_state(state)
            self.simulator.fill_hidden_information(copied_state)

            value = 0
            value_history = []
            terminated = False
            while node.is_fully_expanded(copied_state):
                node, copied_state, value, terminated = node.select(copied_state, value, terminated)
                value_history.append(value)
                if terminated:
                    break
            
            if not terminated:
                policy, value = self.model(torch.tensor(self.simulator.get_encoded_state(copied_state)).unsqueeze(0))
                self.simulator.apply_action_mask(policy, self.simulator.get_valid_actions(copied_state))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                value = value.item()
                value_history.append(value)
                node.expand(policy)       

            node.backpropagate(self.get_discounted_return(value_history))    

        masked_actions = self.simulator.get_masked_actions(self.simulator.get_valid_actions(state))
        real_action_probs = np.array([child.value_sum for child in root.children], dtype=np.float32)
        real_action_probs /= np.sum(real_action_probs)
        expanded_action_probs = np.zeros(self.simulator.max_action_size)
        valid_action_ind = 0
        for i in range(self.simulator.max_action_size):
            if masked_actions[i]:
                expanded_action_probs[i] = real_action_probs[valid_action_ind]
                valid_action_ind += 1
            else:
                expanded_action_probs[i] = 0
        return expanded_action_probs / np.sum(expanded_action_probs)