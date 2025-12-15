from vgc2.agent import BattlePolicy
from vgc2.agent.battle import RandomBattlePolicy, GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, TeamView, State, StateView, BattleRuleParam, BattleCommand
from vgc2.battle_engine.constants import Stat
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams
from vgc2.util.generator import gen_team
from custom_encodings import ENCODING_CONSTANTS, EncodeContext
from itertools import product

class PokemonBattleEnv():
    def __init__(self, opponent: BattlePolicy = GreedyBattlePolicy()):
        self.ctx = EncodeContext()
        self.params = BattleRuleParam()
        self.engine = None
        self.state_view = None 
        self.opponent = opponent
        max_actions_per_pkm = list(product(range(ENCODING_CONSTANTS.MAX_MOVES_PER_PKM), range(ENCODING_CONSTANTS.MAX_ACT_PKM_PER_TEAM)))
        self.action_space = list(product(max_actions_per_pkm, max_actions_per_pkm))
        self.max_turn_reward = 7
        self.win_reward = 4

    def reset(self):
        team = (gen_team(ENCODING_CONSTANTS.MAX_PKM_PER_TEAM, ENCODING_CONSTANTS.MAX_MOVES_PER_PKM),
                gen_team(ENCODING_CONSTANTS.MAX_PKM_PER_TEAM, ENCODING_CONSTANTS.MAX_MOVES_PER_PKM))
        label_teams(team)
        team_view = TeamView(team[0]), TeamView(team[1])
        state = State(get_battle_teams(team, ENCODING_CONSTANTS.MAX_ACT_PKM_PER_TEAM))
        self.engine = BattleEngine(state, self.params)
        self.state_view = (StateView(state, 0, team_view), StateView(state, 1, team_view))
        return self.state_view[0]

    def step(self, action):
        # Setup for reward calculation
        state = self.state_view[0]
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve
        reward = 0

        # Handle index to action conversion and add to battle commands
        cmds: list[BattleCommand] = []
        action = self.action_space[action]
        cmds += action

        # Punish if agent chose disabled or out-of-pp move
        for i, battle_pkm in enumerate(state.sides[0].team.active):
            if battle_pkm.battling_moves[action[i][0]].disabled or battle_pkm.battling_moves[action[i][0]].pp <= 0:
                reward -= 0.1

        # Amount of dead opp pokemon
        dead_opp_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])

        # Amount of dead own pokemon
        dead_own_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])

        # hp calc
        opp_hp_before_turn = sum([pkm.hp for pkm in opp_team])
        own_hp_before_turn = sum([pkm.hp for pkm in own_team])

        # Run turn
        self.engine.run_turn((cmds, self.opponent.decision(self.state_view[1])))

        # reassign for reward calculation
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve

        # Reward if enemy was killed
        dead_opp_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])
        reward += dead_opp_pkm_after_turn-dead_opp_pkm_before_turn # A maximum of 2 pkm can be killed in a single turn

        # Punish if own pkm was killed
        dead_own_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])
        reward += dead_own_pkm_before_turn-dead_own_pkm_after_turn

        # reward if dmg done
        opp_hp_after_turn = sum([pkm.hp for pkm in opp_team])
        reward += (opp_hp_before_turn-opp_hp_after_turn) / sum([pkm.constants.stats[Stat.MAX_HP] for pkm in opp_team])

        # punish if dmg taken
        own_hp_after_turn = sum([pkm.hp for pkm in own_team])
        reward += (own_hp_after_turn-own_hp_before_turn) / sum([pkm.constants.stats[Stat.MAX_HP] for pkm in own_team])

        terminated = self.engine.state.terminal()

        # Reward agent if they won the battle, punish if lost
        if terminated:
            if self.engine.winning_side == 0:
                win_bonus = self.win_reward
            else:
                win_bonus = -self.win_reward
            reward += win_bonus

        return state, reward / self.max_turn_reward, terminated