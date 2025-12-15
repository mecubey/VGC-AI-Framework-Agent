from numpy.random import choice
from vgc2.agent import TeamBuildPolicy, TeamBuildCommand
from vgc2.battle_engine.modifiers import Nature
from vgc2.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Type, Nature

class WolfeTeamBuildPolicy(TeamBuildPolicy):
    def decision(self,
                 roster: Roster,
                 meta: Meta | None,
                 max_team_size: int,
                 max_pkm_moves: int,
                 n_active: int) -> TeamBuildCommand:
        ivs = (31,) * 6
        
        roster.sort(key = lambda x: sum(x.base_stats), reverse=True)

        cmds: TeamBuildCommand = []

        useful_types = [Type.FAIRY, Type.STEEL]

        CUTOFF = 10

        useful_pkm_ids = []
        less_useful_pkm_ids = []

        for i in range(CUTOFF):
            if any(t in useful_types for t in roster[i].types):
                useful_pkm_ids.append(i)
            else:
                less_useful_pkm_ids.append(i)

        if len(useful_pkm_ids) < max_team_size:
            for i in range(max_team_size-len(useful_pkm_ids)):
                useful_pkm_ids.append(less_useful_pkm_ids[i])

        for i in useful_pkm_ids:
            evs = tuple([0, 126, 126, 4, 0, 252])
            nature = Nature(choice(len(Nature), 1, False))
            cmds += [(i, evs, ivs, nature, [0, 1, 2, 3])]
        return cmds
