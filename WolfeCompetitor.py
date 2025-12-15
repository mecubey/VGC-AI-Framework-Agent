from vgc2.agent import BattlePolicy, SelectionPolicy, TeamBuildPolicy
from vgc2.competition import Competitor

from .WolfeBattlePolicy import WolfeBattlePolicy
from .WolfeSelectionPolicy import WolfeSelectionPolicy
from .WolfeTeamBuildPolicy import WolfeTeamBuildPolicy

class WolfeCompetitor(Competitor):

    def __init__(self, name: str = "Wolfe"):
        self.__name = name
        self.__battle_policy = WolfeBattlePolicy()
        self.__selection_policy = WolfeSelectionPolicy()
        self.__team_build_policy = WolfeTeamBuildPolicy()

    @property
    def battle_policy(self) -> BattlePolicy | None:
        return self.__battle_policy

    @property
    def selection_policy(self) -> SelectionPolicy | None:
        return self.__selection_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy | None:
        return self.__team_build_policy

    @property
    def name(self) -> str:
        return self.__name
