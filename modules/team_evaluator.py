import json
import pandas as pd
from abc import ABC, abstractmethod
from copy import deepcopy

from modules.team_solver import TeamSolver
from modules.forest_team_predicter import RandomForestRegressor

class TeamEvaluator(TeamSolver):
	def __init__(self):

		self.allData = []
		self.sampleSize = 0

	def evaluateAndTrain(self, pModel: TeamSolver) -> float:

		pModel.train()
		pModel.find_team()
		_sum: float = 0.0
		numSamples: int = 0

		with open("./data/player_stats.json", "r") as f:
			self.dataJson: dict = json.load(f)
		
		for (_season, tempDict) in self.dataJson.items():
			season: int = int(_season)
			for (_gameweek, playerData) in tempDict.items():
				gameweek: int = int(_gameweek)
				currentData = pd.DataFrame.from_records(playerData)
				if(currentData["points_this_week"].notnull().all()):
					currentData["score"] = currentData["points_this_week"]
					self.latestData = currentData
					self.train()
					tempModel = deepcopy(pModel)
					tempModel.updatePredictionData(season, gameweek)
					tempModel.train()
					_sum += self.getTeamDiffs(tempModel.getTeam())
					numSamples += 1
		if (numSamples > 0):
			return _sum / numSamples
		else:
			return 0.0

	def getTeamDiffs(self, pOtherTeam: dict[str, dict[str, pd.DataFrame]]):
		thisPlayers: dict[str, dict[str, pd.DataFrame]] = self.getTeam()
		thisPlayersSet: set[int] = set(thisPlayers["team"]["id"]).union(set(thisPlayers["bench"]["id"]))
		otherPlayersSet: set[int] = set(pOtherTeam["team"]["id"]).union(set(pOtherTeam["bench"]["id"]))
		numCommon: int = len(thisPlayersSet.intersection(otherPlayersSet))
		return numCommon / len(thisPlayersSet)

	def precalcScores(self, pData, pGameweek, pSeason):
		...