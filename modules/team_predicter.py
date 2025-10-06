import pandas as pd
import numpy as np

from modules.team_solver import TeamSolver

class TeamPredicter(TeamSolver):
    def __init__(self, pHeuristic, pMode, verbose = False, pLabel = None):
        super().__init__(pHeuristic, pMode, verbose, pLabel)
        self.xCols = ["id","ict_index", "position", "team", "gameweek", "season", "form", "opposing_team", "play_percent", "fixture_dif", "clean_sheets", "expected_goals", "status"]
        self.categoricalColumns = ["id", "position", "team", "opposing_team", "status"]
        self.yCols = ["points_this_week"]
        self.toDummyColumns = ["team", "opposing_team", "status"]
        self.allCols = self.xCols + self.yCols
        self.allDummyColumns: set[str] = set()
        self.idNameDict: dict[int, str] = dict()
    
    def concatWeeks(self, pCallback = None) -> pd.DataFrame:
        tempDf = pd.DataFrame(columns=self.allCols)
        for datum in self.allData:

            if (pCallback is not None):
                pCallback(datum)

            tempDf = pd.concat([tempDf, datum])
        return tempDf.copy()

    def setDummyCols(self, pDatum: pd.DataFrame):
        playerIds = zip(pDatum["id"].values, pDatum["name"])
        for key, val in playerIds:
            self.idNameDict[key] = val

        currentPlayers = set(pDatum["id"].apply(lambda x: "id_" + str(x)))
        self.allDummyColumns = self.allDummyColumns.union(currentPlayers)

        currentTeams = set(pDatum["team"].apply(lambda x: "team_" + x))
        self.allDummyColumns = self.allDummyColumns.union(currentTeams)

        currentOpposingTeams = set(pDatum["opposing_team"].apply(lambda x: "opposing_team_" + x))
        self.allDummyColumns = self.allDummyColumns.union(currentOpposingTeams)

        currentStatuses = set(pDatum["status"].apply(lambda x: "status_" + x))
        self.allDummyColumns = self.allDummyColumns.union(currentStatuses)

    def setDummies(self, pToDummy: pd.DataFrame) -> pd.DataFrame:
        result = pd.get_dummies(pToDummy, columns=self.toDummyColumns)
        colsToAdd = set()
        xCols: set[str] = set(result.columns)
        for col in self.allDummyColumns:
            if col not in xCols and col not in pToDummy.columns:
                result.insert(len(result.columns), col, 0)
                result = result.copy()
        result = result.sort_index(axis=1)
        if "id" in result.columns:
            result["id"] = result["id"].astype("category")
        return result

    def fixDataTypes(self, pDf: pd.DataFrame) -> pd.DataFrame:
        pDf["points_this_week"] = pDf["points_this_week"].astype(np.float64)
        pDf["gameweek"] = pDf["gameweek"].astype(np.uint16)
        pDf["season"] = pDf["season"].astype(np.uint16)

        return pDf