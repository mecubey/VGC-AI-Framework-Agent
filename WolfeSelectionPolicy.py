from random import shuffle

from vgc2.agent import SelectionPolicy, SelectionCommand
from vgc2.battle_engine import Team
import numpy as np
from vgc2.battle_engine import BattleRuleParam

class WolfeSelectionPolicy(SelectionPolicy):
    def decision(self,
                 teams: tuple[Team, Team],
                 max_size: int) -> SelectionCommand:
        params = BattleRuleParam()

        my_team = teams[0].members
        scores = [0 for i in range(max_size)]
        opp_team = teams[1].members

        for i, my_pkm in enumerate(my_team):
            for move in my_pkm.moves:
                for opp_pkm in opp_team:
                    for opp_type in opp_pkm.species.types:
                        scores[i] += params.DAMAGE_MULTIPLICATION_ARRAY[move.pkm_type][opp_type] 

        ids = []
        IGNORE = -999
        for i in range(max_size):
            cur_max = max(scores)
            cur_ind = scores.index(cur_max)
            ids.append(cur_ind)
            scores[cur_ind] = IGNORE

        return ids