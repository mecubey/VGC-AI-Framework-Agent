import numpy as np
import torch
from vgc2.agent.battle import RandomBattlePolicy, GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, State, BattleRuleParam, BattleCommand, BattlingMove, BattlingPokemon
from vgc2.battle_engine.constants import Stat
from .custom_encodings import ENCODING_CONSTANTS, EncodeContext, encode_state
from itertools import product
from vgc2.util.generator import gen_move_set

class OfflineSimulator():
    def __init__(self):
        self.ctx = EncodeContext()
        self.params = BattleRuleParam()
        self.greedy_battler = GreedyBattlePolicy()

        max_actions_per_pkm = list(product(range(ENCODING_CONSTANTS.MAX_MOVES_PER_PKM), range(ENCODING_CONSTANTS.MAX_ACT_PKM_PER_TEAM)))
        one_act_opp = list(product(range(ENCODING_CONSTANTS.MAX_MOVES_PER_PKM), range(ENCODING_CONSTANTS.MAX_ACT_PKM_PER_TEAM-1)))
        action_placeholder = [(0, 0)]
        self.max_action_size = 64
        self.max_action_space_key = (2, 2)
        self.possible_action_spaces = {
            (2, 2): list(product(max_actions_per_pkm, max_actions_per_pkm)),
            (2, 1): list(product(one_act_opp, one_act_opp)),
            (1, 2): list(product(max_actions_per_pkm, action_placeholder)),
            (1, 1): list(product(one_act_opp, action_placeholder))
        }
        self.max_action_space_indices = list(range(self.max_action_size))

        self.max_reward_per_turn = 7
        self.win_reward = 4

    def step(self, state: State, action):
        # setup
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve
        reward = 0

        cmds: list[BattleCommand] = []
        cmds += action

        # punish disabled or out-of-pp move
        for i, battle_pkm in enumerate(state.sides[0].team.active):
            if battle_pkm.battling_moves[action[i][0]].disabled or battle_pkm.battling_moves[action[i][0]].pp <= 0:
                reward -= 0.1

        # dead pokemon calc
        dead_opp_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])
        dead_own_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])

        # run turn
        engine = BattleEngine(state, self.params)
        engine.run_turn((cmds, self.greedy_battler.decision(State((state.sides[1], state.sides[0])))))

        # reassign for reward calculation
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve

        # reward if enemy was killed
        dead_opp_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])
        reward += dead_opp_pkm_after_turn-dead_opp_pkm_before_turn # A maximum of 2 pkm can be killed in a single turn

        # punish if own pkm was killed
        dead_own_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])
        reward += dead_own_pkm_before_turn-dead_own_pkm_after_turn

        terminated = engine.state.terminal()

        # reward agent if they won the battle, punish if lost
        if terminated:
            if engine.winning_side == 0:
                reward = self.win_reward
            else:
                reward = -self.win_reward

        return state, reward / self.max_reward_per_turn, terminated
    
    def fill_hidden_information(self, state: State):
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve

        # generate hidden reserve opponent pokemon
        for _ in range(len(state.sides[0].team.reserve)-len(state.sides[1].team.reserve)):
            state.sides[1].team.reserve.append(BattlingPokemon(opp_team[np.random.randint(len(opp_team))].constants))

        # reassign, because opponent team changed
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve

        # if any opponent pokemon has no known moves, we generate movesets randomly
        for i in range(len(opp_team)):
            if len(opp_team[i].battling_moves) == 0:
                opp_team[i].battling_moves = [BattlingMove(move) for move in gen_move_set(ENCODING_CONSTANTS.MAX_MOVES_PER_PKM)]

    def get_greedy_decision(self, state: State):
        return self.greedy_battler.decision(state)

    def get_valid_actions(self, state: State):
        return self.possible_action_spaces[(len(state.sides[0].team.active),
                                            len(state.sides[1].team.active))]
    
    def get_encoded_state(self, state: State):
        return encode_state(np.zeros(ENCODING_CONSTANTS.STATE, dtype=np.float32), state, self.ctx, self.params, self.greedy_battler)

    def get_masked_actions(self, valid_actions):
        allowed_actions = np.full(self.max_action_size, False)
        for i in range(self.max_action_size):
            allowed_actions[i] = self.possible_action_spaces[self.max_action_space_key][i] in valid_actions
        return allowed_actions

    def apply_action_mask(self, policy: torch.Tensor, valid_actions):
        action_mask = self.get_masked_actions(valid_actions)
        action_mask = torch.tensor(action_mask)
        policy.masked_fill_(~action_mask, float('-inf'))

    def apply_action_mask_to_numpy(self, policy: np.array, valid_actions):
        action_mask = self.get_masked_actions(valid_actions)
        for i in range(len(action_mask)):
            if not action_mask[i]:
                policy[i] = 0