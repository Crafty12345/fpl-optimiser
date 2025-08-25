import pandas as pd

from team_solver import TeamSolver

class SimpleTeamSolver(TeamSolver):
    def __init__(self, pHeuristic, pMode, verbose = False, pLabel: str = None):
        super().__init__(pHeuristic, pMode, verbose, pLabel)

    def precalcScores(self, pData: pd.DataFrame, pGameweek: int, pSeason: int):
        pData["score"] = pData[self.score_heuristic] * pData["form"] * pData["starts_per_90"]