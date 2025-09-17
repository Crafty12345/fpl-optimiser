from glob import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re
import json
from copy import deepcopy

from modules.team_solver import TeamSolver, SolverMode
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix
import config

# TODO
class LinearTeamPredicter(TeamSolver):
	"""
	Class for team solver with linear regression functionality
	"""
	def __init__(self, 
				pHeuristic: str, 
				pSolverMode: SolverMode, 
				verbose = False, 
				pLabel: str = None):
		
		super().__init__(pHeuristic, pSolverMode, verbose, pLabel)

		self.uniquePlayers = self.allData[-1]["name"]
		if(self.verbose):
			print("[DEBUG]: Done reading data files! Calculating linear regression...")
		self.playerModelDict: dict[str, LinearRegression] = dict()

		for player in self.uniquePlayers:
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
			self.playerModelDict[player] = model
			xToPredict = np.asarray([self.sampleSize]).reshape((1, -1))

			predictedWeightedScore = self.predictPlayer(xToPredict, player)
			self.latestData.loc[self.latestData["name"] == player, "score"] = predictedWeightedScore

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

	def getDfIndex(self, pSeason: int, pGameweek: int) -> int:
		# Default to next week
		result: int = self.sampleSize
		for (i, file) in enumerate(self.allData):
			if(file["season"] == pSeason & file["gameweek"] == pGameweek):
				result = i
		return result
		...
	def predictPlayer(self, pX: int, pPlayer: str):
		model = self.playerModelDict[pPlayer]
		xToPredict = np.asarray(pX).reshape((1, -1))
		predictedScore = model.predict(xToPredict).reshape(-1)
		playerForm = self.latestData.loc[self.latestData["name"] == pPlayer, "form"]
		playerStartsPer90 = self.latestData.loc[self.latestData["name"] == pPlayer, "starts_per_90"]
		predictedWeightedScore = predictedScore * playerForm * playerStartsPer90
		return predictedWeightedScore

	def updatePredictionData(self, pSeason: int = None, pGameweek: int = None):
		selectedIndex: int = self.getDfIndex(pSeason, pGameweek)
		xToPredict = np.asarray([selectedIndex]).reshape((1, -1))
		for player in self.playerModelDict.keys():
			newScore = self.predictPlayer(xToPredict, player)
			self.latestData.loc[self.latestData["name"]==player, "score"] = newScore

	def updatePredictionData(self, pSeason: int = None, pGameweek: int = None) -> None:
		selectedDf: pd.DataFrame = None
		if (pSeason is None and pGameweek is None):
			self.latestData: pd.DataFrame = self.allData[-1].copy()
			return
		selectedDf = self.getDfByWeekAndSeason(pSeason, pGameweek)
		if (selectedDf is None):
			print(f"Unable to predict for week {pGameweek} of season {pSeason}")
		elif selectedDf is not None:
			self.latestData = selectedDf.copy()