import pandas as pd
import numpy as np
import line_profiler

from modules.team_solver import TeamSolver

class TeamPredicter(TeamSolver):
    def __init__(self, pHeuristic, pMode, verbose = False, pLabel = None, pFreeHit = False):
        super().__init__(pHeuristic, pMode, verbose, pLabel, pFreeHit=pFreeHit)
        self.xCols = ["id","ict_index", "position", "team", "gameweek", "season", "form", "play_percent", "fixture_dif", "clean_sheets", "expected_goals", "status"]
        self.categoricalColumns = ["id", "position", "team", "status"]
        self.yCols = ["total_points"]
        self.toDummyColumns = ["team", "status"]
        self.allCols = self.xCols + self.yCols
        self.allDummyColumns: set[str] = set()
        self.idNameDict: dict[int, str] = dict()
    
    # TODO: Optimise this method
    def concatWeeks(self, pCallback = None) -> pd.DataFrame:
        # TODO: Add dtype parameter to DataFrame creation
        tempDf = pd.DataFrame(columns=self.allCols)
        toConcat: np.ndarray[pd.DataFrame] = np.ndarray(len(self.allData)+1, dtype=object)
        toConcat[0] = tempDf
        i: int = 1
        for datum in self.allData:
            if (pCallback is not None):
                pCallback(datum)
            toConcat[i] = datum
            i += 1

        assert i == len(toConcat)
        # TODO: Fix some players having multiple indexes
        return pd.concat(toConcat)

    def setDummyCols(self, pDatum: pd.DataFrame):
        playerIds = zip(pDatum["id"].values, pDatum["name"])
        for key, val in playerIds:
            self.idNameDict[key] = val

        currentPlayers = set(pDatum["id"].apply(lambda x: "id_" + str(x)))
        self.allDummyColumns = self.allDummyColumns.union(currentPlayers)

        currentTeams = set(pDatum["team"].apply(lambda x: "team_" + x))
        self.allDummyColumns = self.allDummyColumns.union(currentTeams)

        #currentOpposingTeams = set(pDatum["opposing_team"].apply(lambda x: "opposing_team_" + x))
        #self.allDummyColumns = self.allDummyColumns.union(currentOpposingTeams)

        currentStatuses = set(pDatum["status"].apply(lambda x: "status_" + x))
        self.allDummyColumns = self.allDummyColumns.union(currentStatuses)

    def setDummies(self, pToDummy: pd.DataFrame) -> pd.DataFrame:
        result = pd.get_dummies(pToDummy, columns=self.toDummyColumns)
        colsToAdd = np.ndarray(len(self.allDummyColumns), dtype=object)
        numColsToAdd = 0
        xCols: set[str] = set(result.columns)
        for col in self.allDummyColumns:
            if col not in xCols and col not in pToDummy.columns:
                # TODO: Fix this
                #result.insert(len(result.columns), col, 0)
                #print(col)
                colsToAdd[numColsToAdd] = col
                numColsToAdd += 1

        colsToAdd = colsToAdd[colsToAdd != np.array(None)]
        
        colsDf = pd.DataFrame(columns=colsToAdd)
        result = pd.concat([result, colsDf], axis=1)
        #print(colsDf.columns)
        for col in colsDf.columns:
            assert col in result.columns
        for col in self.allDummyColumns:
            assert col in result.columns

        
        #result = result.copy()
        result = result.sort_index(axis=1)
        if "id" in result.columns:
            result["id"] = result["id"].astype("category")
        return result

    @line_profiler.profile
    def fixDataTypes(self, pDf: pd.DataFrame) -> pd.DataFrame:
        pDf["points_this_week"] = pDf["points_this_week"].astype(np.float64)
        pDf["gameweek"] = pDf["gameweek"].astype(np.uint16)
        pDf["season"] = pDf["season"].astype(np.uint16)
        if "fixture_dif" in pDf.columns:
            pDf["fixture_dif"] = pDf["fixture_dif"].astype(np.float32)

        return pDf