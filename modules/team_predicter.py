from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json

from modules.team_solver import TeamSolver, SolverMode
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix
from modules.utils import getDataFilesSorted
import config

class TeamPredicter(TeamSolver):
    """
    Class for team solver with linear regression functionality
    """
    def __init__(self, pHeuristic: str, pSolverMode: SolverMode, verbose = False):
        self.score_heuristic = pHeuristic
        self.mode = pSolverMode
        self.verbose = verbose

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
        
        with open("./data/player_stats.json", "r") as f:
            self.dataJson: dict = json.load(f)

        # Number of gameweeks which have been sampled
        self.sampleSize = 0

        for (season, tempDict) in self.dataJson.items():
            season = int(season)
            for (currentGameweek, playerData) in tempDict.items():
                currentGameweek = int(currentGameweek)

                # Convert playerIDs back to ints

                maxGameweek = min(currentGameweek+2, 39)
                currentData = pd.DataFrame.from_records(playerData)

                if(pHeuristic == "combined"):
                    currentData["combined"] = self.calculateCombinedScore(currentData)
                
                matrix = FixtureDifficultyMatrix(1.0, currentGameweek, currentGameweek, season)
                currentData["weight"] = currentData["team"].apply(matrix.getSimpleDifficulty)
                # Decrease weight as season gets older
                currentData["weight"] = currentData["weight"] / (config.CURRENT_SEASON - season + 0.9)

                currentData["score"] = currentData[self.score_heuristic] * currentData["weight"]
                self.allData.append(currentData)
                self.sampleSize += 1

        self.data = self.allData[-1].copy()
        uniquePlayers = self.allData[-1]["name"]
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