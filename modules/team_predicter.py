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

# TODO
class TeamPredicter(TeamSolver):
	"""
	Class for team solver with linear regression functionality
	"""
	def __init__(self, pHeuristic: str, pSolverMode: SolverMode, verbose = False, pLabel: str = None):
		super().__init__(pHeuristic, pSolverMode, verbose, pLabel)

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
	def precalcScores(self, pData: pd.DataFrame, pGameweek: int, pSeason: int):
		if(self.score_heuristic == "combined"):
			pData["combined"] = self.calculateCombinedScore(pData)
		
		matrix = FixtureDifficultyMatrix(1.0, pGameweek, pGameweek, pSeason)
		pData["weight"] = pData["team"].apply(matrix.getSimpleDifficulty)
		# Decrease weight as season gets older
		pData["weight"] = pData["weight"] / (config.CURRENT_SEASON - pSeason + 0.9)

		pData["score"] = pData[self.score_heuristic] * pData["weight"]