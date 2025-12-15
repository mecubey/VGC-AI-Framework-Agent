import torch
import os
import random
import numpy as np
import torch.nn.functional as F
from MCTS import MCTS
from ResNet import ResNet
from PokemonBattleEnv import PokemonBattleEnv
from OfflineSimulator import OfflineSimulator

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(MAIN_DIR, "saves")

class AlphaWolfe:
    def __init__(self, model: ResNet, optimizer, simulator: OfflineSimulator, args):
        self.model = model
        self.model.load_state_dict(torch.load("C:/Projects/pokemon-vgc-engine/tutorial/AlphaWolfe/saves/model_21.pt"))
        self.optimizer = optimizer
        self.optimizer.load_state_dict(torch.load("C:/Projects/pokemon-vgc-engine/tutorial/AlphaWolfe/saves/optimizer_21.pth"))
        self.simulator = simulator
        self.args = args
        self.mcts = MCTS(self.simulator, self.args, self.model)
        self.game = PokemonBattleEnv()

    def self_play(self):
        memory = []
        state = self.game.reset()
        reward = 0
        terminated = False

        while True:
            action_probs = self.mcts.search(state)
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.simulator.max_action_space_indices, p=temperature_action_probs)
            next_state, reward, terminated = self.game.step(action)            
            memory.append((self.simulator.get_encoded_state(state), action_probs, reward))

            if terminated:
                returnMemory = []
                G = 0
                for hist_encoded_state, hist_action_probs, hist_reward in reversed(memory):
                    G = hist_reward + self.args['gamma'] * G
                    returnMemory.append((hist_encoded_state, hist_action_probs, G))
                return returnMemory
            
            state = next_state

    def train(self, sample: list):
        state, policy_targets, value_targets = zip(*sample)
        state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
        state = torch.tensor(state, dtype=torch.float32)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
        value_targets = torch.tensor(value_targets, dtype=torch.float32)
        
        out_policy, out_value = self.model(state)
        
        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        for iteration in range(22, self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.self_play()

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train([memory[i] for i in np.random.randint(low=len(memory), size=self.args['batch_size'])])

            torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, f'model_{iteration}.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(SAVE_DIR, f'optimizer_{iteration}.pth'))

if __name__ == "__main__":
    game = PokemonBattleEnv()
    num_resBlocks = 5
    model = ResNet(len(game.action_space), num_resBlocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    simulator = OfflineSimulator()
    args = {
        'gamma': 0.95,
        'C': 2,
        'num_searches': 800,
        'num_iterations': 500,
        'num_selfPlay_iterations': 250,
        'num_epochs': 13,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.075,
        'use_dirichlet': True
    }
    alphaWolfe = AlphaWolfe(model, optimizer, simulator, args)
    alphaWolfe.learn()