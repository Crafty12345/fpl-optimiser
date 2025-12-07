from glob import glob
import line_profiler
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
		self.uniquePlayers = self.latestData["name"]
		self.playerModelDict: dict[str, LinearRegression] = dict()

	@line_profiler.profile
	def fit(self) -> None:
		# TODO: Get r2
		if(self.verbose):
			print("[DEBUG]: Done reading data files! Calculating linear regression...")

		nameColIndex = [i for (i, col) in enumerate(self.latestData.columns) if col == "name"][0]
		scoreColIndex = [i for (i, col) in enumerate(self.latestData.columns) if col == "score"][0]

		for playerId in self.latestData.index:
			y = np.ndarray(shape=len(self.allData), dtype=np.float32)
			playerName = self.latestData.iat[playerId, nameColIndex]
			for (i, dataFile) in enumerate(self.allData):
				if (playerId in dataFile.index):
					toAppend = dataFile.iat[playerId, scoreColIndex]
				else:
					toAppend = 0.0
				#toAppend = dataFile.loc[dataFile["name"]==player, "score"]
				# print("toAppendIndex=", toAppend.index)
				# if(len(toAppend.index) > 0):
				# 	print(toAppend.iat[toAppend.index[0]])
				# if (len(toAppend) > 0):
				# 	toAppend = toAppend.item()
				# else:
				# 	toAppend = 0.0

				y[i] = (toAppend)
			x = np.arange(self.sampleSize).reshape((-1, 1))
			model = LinearRegression().fit(x, y)
			self.playerModelDict[playerName] = model
			xToPredict = np.asarray([self.sampleSize]).reshape((1, -1))

			predictedWeightedScore = self.predictPlayer(xToPredict, playerName)

			#print("latestData.index=", self.latestData.loc[self.latestData["name"] == player, "score"].index)
			self.latestData.iat[playerId, scoreColIndex] = predictedWeightedScore.item()

		if(self.verbose):
			print("Done calculating linear regression!")

	def precalcScores(self, pData: pd.DataFrame, pGameweek: int, pSeason: int):
		if(self.score_heuristic == "combined"):
			pData["combined"] = self.calculateCombinedScore(pData)
		
		matrix = FixtureDifficultyMatrix()
		matrix.precomputeFixtureDifficulty(0, pGameweek, 3, pSeason, 1.0)
		pData["weight"] = pData.apply(lambda x: matrix.calcSimpleDifficulty(x["team"], x["opposing_team"]), axis=1)
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
		if (len(self.playerModelDict) == 0):
			raise ValueError("Unable to predict player: Models have not been fitted yet. Remember to call fit()")
		model = self.playerModelDict[pPlayer]
		xToPredict = np.asarray(pX).reshape((1, -1))
		predictedScore = model.predict(xToPredict).reshape(-1)
		playerForm = self.latestData.loc[self.latestData["name"] == pPlayer, "form"]
		chanceOfPlay = self.latestData.loc[self.latestData["name"] == pPlayer, "play_percent"]
		predictedWeightedScore = predictedScore * playerForm * chanceOfPlay
		return predictedWeightedScore

	def updatePredictionData(self, pSeason: int, pGameweek: int = None):
		selectedIndex: int = self.getDfIndex(pSeason, pGameweek)
		xToPredict = np.asarray([selectedIndex]).reshape((1, -1))
		for player in self.playerModelDict.keys():
			newScore = self.predictPlayer(xToPredict, player)
			self.latestData.loc[self.latestData["name"]==player, "score"] = newScore

	def updatePredictionData(self, pRefSeason: int, pTargetSeason: int, pRefWeek: int, pTargetWeek: int) -> None:
		selectedDf: pd.DataFrame = None
		selectedDf = self.getDfByWeekAndSeason(pTargetWeek, pTargetSeason)
		if (selectedDf is None):
			print(f"Unable to update predictions for week {pRefWeek} of season {pRefSeason}")
		elif selectedDf is not None:
			self.latestData = selectedDf.copy()