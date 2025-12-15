import numpy as np
import torch
import os
from typing import Optional
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, TeamView
from .MCTS import MCTS
from .OfflineSimulator import OfflineSimulator
from .ResNet import ResNet

class WolfeBattlePolicy(BattlePolicy):
    def __init__(self):
        self.args = {
            'gamma': 0.83,
            'C': 2,
            'dirichlet_epsilon': 0,
            'dirichlet_alpha': 0.075
        }
        MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
        SAVES_DIR = os.path.join(MAIN_DIR, "saves")
        MODEL_FILE = os.path.join(SAVES_DIR, 'model.pt')
        self.simulator = OfflineSimulator()
        self.model = ResNet(self.simulator.max_action_size, 5)
        self.model.load_state_dict(torch.load(MODEL_FILE))
        self.model.eval()
        self.mcts = MCTS(self.simulator, self.args, self.model)

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        valid_actions = self.simulator.get_valid_actions(state)
        if len(valid_actions) == 64: # MCTS just takes way too long...
            return self.simulator.get_greedy_decision(state)
        cmds: list[BattleCommand] = []
        action_probs = self.mcts.search(state)
        cmds += self.simulator.possible_action_spaces[self.simulator.max_action_space_key][np.argmax(action_probs)]
        return cmds