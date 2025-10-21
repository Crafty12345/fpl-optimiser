from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass
from collections import deque
from copy import deepcopy

from modules.player import Player, Position

NUM_GOALKEEPERS = 1
NUM_DEFENDERS = 4
NUM_FORWARD = 2
NUM_MID = 4

assert NUM_GOALKEEPERS + NUM_DEFENDERS + NUM_FORWARD + NUM_MID == 11

MAX_POSITION: dict[Position, int] = {
	Position.DEF: NUM_DEFENDERS,
	Position.FWD: NUM_FORWARD,
	Position.GKP: NUM_GOALKEEPERS,
	Position.MID: NUM_MID
}

MAX_BUDGET = 1000
MAX_AMT_REMAINING = 2

@dataclass
class TreeData:
	numOfPosition: dict[Position, int]
	currentCost: float
	teamCounts: dict[str, int]
	positionIndexes: dict[Position, int]

class TeamTreeNode():
	def __init__(self, pData: TreeData, pPlayer: Player, pPlayerDict: dict[Position, list[Player]]):
		self.branches: list[TeamTreeNode] = []
		self.player: Player = pPlayer
		self.data = pData
		self.playerQ = pPlayerDict

	def shouldAddPlayer(self, pPlayer: Player) -> bool:
		playerCost = pPlayer.getCost()
		playerPosition = pPlayer.getPosition()
		playerTeam = pPlayer.getTeam()
		numOfTeam = self.data.teamCounts.get(playerTeam, 0)
		result = False

		if (self.data.numOfPosition[playerPosition] < MAX_POSITION[playerPosition]):
			if (self.data.currentCost + playerCost < MAX_BUDGET):
				if (numOfTeam < 3):
					result = True
		return result
	
	def addPlayer(self, pPlayer: Player, pNewIndex: int):
		# Copy so that future trees don't access the same tree
		dataCopy = deepcopy(self.data)
		playerPosition = pPlayer.getPosition()
		playerCost = pPlayer.getCost()
		playerTeam = pPlayer.getTeam()

		newNumOfPosition = self.data.numOfPosition.get(playerPosition, 0) + 1
		dataCopy.numOfPosition[playerPosition] = newNumOfPosition

		dataCopy.positionIndexes[playerPosition] = pNewIndex

		currentTeamCount = self.data.teamCounts.get(playerTeam, 0)		
		self.data.teamCounts[playerTeam] = currentTeamCount + 1

		dataCopy.currentCost += playerCost

		newBranch = TeamTreeNode(dataCopy, pPlayer, self.playerQ)
		newBranch.nextBranch()
		self.branches.append(newBranch)

	def nextBranch(self):
		for position, deque in self.playerQ.items():
			i: int = self.data.positionIndexes[position]
			if (i < len(deque)):
				i: int = self.data.positionIndexes[position]
				selectedPlayer = deque[i]
				while (not self.shouldAddPlayer(selectedPlayer) and i < len(deque)):
					i += 1
					if (i < len(deque)):
						selectedPlayer = deque[i]
				if (self.shouldAddPlayer(selectedPlayer)):
					self.addPlayer(selectedPlayer, i)

	def bestBranch(self) -> tuple[list[Player], float]:
		bestScore = -999.99
		bestPlayers: list[Player] = None
		for branch in self.branches:
			players, score = branch.bestBranch()
			if ((score > bestScore) and (not self.player in players)):
				bestScore = branch.getScore()
				bestPlayers = players

		if bestPlayers is None:
			#print(self.player.getScore())
			return ([self.player], self.player.getScore())
		else:
			scores = [x.getScore() for x in bestPlayers]
			score = sum(scores)
			if (self.player is not None):
				score += self.player.getScore()
			return ([self.player] + bestPlayers, score)

	def getPlayer(self) -> Player:
		return self.player
	def getBranches(self):
		return self.branches
	def getScore(self) -> float:
		return self.player.getScore()

class TeamTreeFactory():
	def __init__(self, pPlayerDf: pd.DataFrame):
		self.playerDf = pPlayerDf
		self.positions = Position.listValues()

	def splitPositions(self, pPlayerList: list[Player]) -> dict[Position, list[Player]]:
		result = {pos: [] for pos in self.positions}
		for player in pPlayerList:
			result[player.getPosition()].append(player)
		return result

	def create(self):
		positionCounts = {pos: 0 for pos in self.positions}
		positionIndexes = {pos: 0 for pos in self.positions}
		cost = 0
		playerList = self.playerDf.apply(lambda x: Player.fromRow(x), axis=1)
		treeData: TreeData = TreeData(positionCounts, cost, dict(), positionIndexes)
		return TeamTreeNode(treeData, None, self.splitPositions(playerList))