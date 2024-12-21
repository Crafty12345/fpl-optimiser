from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from modules.team_solver import TeamSolver, SolverMode
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix
import config

class TeamPredicter(TeamSolver):
    """
    Class for team solver with linear regression functionality
    """
    def __init__(self, pHeuristic: str, pSolverMode: SolverMode, verbose = False):
        self.score_heuristic = pHeuristic
        self.mode = pSolverMode
        self.verbose = verbose

        self.allDataFiles = glob("./data/player_stats/*.csv")
        self.sampleSize = len(self.allDataFiles)
        ALL_COLUMNS = [
            "id",
            "name",
            "cost",
            "ict_index",
            "total_points",
            "points_per_game",
            "form",
            "status",
            "starts_per_90",
            "position",
            "team"
            ]

        self.allData = []
        if(self.verbose):
            print("[DEBUG]: Reading from data files...")
        for i in range(len(self.allDataFiles)):
            currentGameweek = i+1
            currentFileName = self.allDataFiles[i]
            currentData = pd.read_csv(currentFileName)
            if(pHeuristic == "combined"):
                currentData["combined"] = self.calculateCombinedScore(currentData)
            matrix = FixtureDifficultyMatrix(1.0, currentGameweek, currentGameweek)
            currentData["weight"] = currentData["team"].apply(matrix.getSimpleDifficulty)

            currentData["score"] = currentData[self.score_heuristic] * currentData["weight"]
            self.allData.append(currentData)

        self.data = self.allData[-1].copy()
        uniquePlayers = self.allData[-1]["name"]
        scoreDict = dict()
        if(self.verbose):
            print("[DEBUG]: Done reading data files! Calculating linear regression...")

        for player in uniquePlayers:
            y = []
            for dataFile in self.allData:
                toAppend = dataFile.loc[dataFile["name"]==player]["score"]
                if(len(toAppend) == 0):
                    toAppend = 0.0
                else:
                    toAppend = toAppend.values[0]
                y.append(toAppend)
            x = np.arange(self.sampleSize).reshape((-1, 1))
            model = LinearRegression().fit(x, y)
            xToPredict = np.asarray([self.sampleSize+1]).reshape((1, -1))
            predictedScore = model.predict(xToPredict).reshape(-1)
            playerForm = self.data.loc[self.data["name"] == player, "form"]
            playerStartsPer90 = self.data.loc[self.data["name"] == player, "starts_per_90"]
            predictedWeightedScore = predictedScore * playerForm * playerStartsPer90
            self.data.loc[self.data["name"] == player, "score"] = predictedWeightedScore
        
        if(self.verbose):
            print("Done calculating linear regression!")
        
        self.max_iters = config.MAX_ITERS
        self.log = verbose

        self.registerInstance()
        self.removeOutliers()
        self.start()