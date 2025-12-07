import pandas as pd
import numpy as np
from copy import deepcopy

from modules.player import Player, Position
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix

class Team():
    def __init__(self,
                 goalkeepers: list[Player],
                 defenders: list[Player],
                 forwarders: list[Player],
                 midfielders = list[Player]
                 ):
        self.goalkeepers = goalkeepers
        self.defenders = defenders
        self.forwarders = forwarders
        self.midfielders = midfielders
        pass
    
    @classmethod
    def fromPlayerList(cls, players: list[Player]):
        goalkeepers = []
        defenders = []
        forwarders = []
        midfielders = []
        for player in players:
            match player.getPosition():
                case Position.GKP:
                    goalkeepers.append(player)
                case Position.DEF:
                    defenders.append(player)
                case Position.FWD:
                    forwarders.append(player)
                case Position.MID:
                    midfielders.append(player)
        newTeam = cls(goalkeepers, defenders, forwarders, midfielders)
        newTeam.updatePlayers()
        return newTeam

    @classmethod
    def fromNameSet(cls, pSet: set[str], allPlayerData: pd.DataFrame):
        players = []
        for player in pSet:
            _player = Player.fromName(player,allPlayerData)
            players.append(_player)
        return cls.fromPlayerList(players)
    
    @classmethod
    def fromDataFrame(cls, pDataFrame: pd.DataFrame):
        players = []
        for index, player in pDataFrame.iterrows():
            isAvailable = player["status"] == "a"
            newPlayer = Player(player["id"],
                               player["name"],
                               player["cost"],
                               player["ict_index"],
                               player["total_points"],
                               player["points_per_game"],
                               player["form"],
                               Position.fromString(player["position"]),
                               player["team"], player["score"], isAvailable, player["play_percent"], player["opposing_team"], pScore=player["score"])
            players.append(newPlayer)
        return cls.fromPlayerList(players)

    def getGoalkeepers(self) -> list[Player]: return self.goalkeepers
    def getDefenders(self) -> list[Player]: return self.defenders
    def getForwarders(self) -> list[Player]: return self.forwarders
    def getMidfielders(self) -> list[Player]: return self.midfielders

    def getPlayers(self) -> list[Player]: return self.players
    
    def getTotalCost(self) -> float:
        return sum([player.getCost() for player in self.players])
    def getTotalScore(self) -> float:
        return sum([player.getScore() for player in self.players])
    def getPlayerNames(self) -> list[str]:
        return [player.getName() for player in self.players]
    def getPlayerScores(self) -> list[float]:
        return [player.getScore() for player in self.players]
    
    def getCaptain(self):
        return max(self.players)
    def getViceCaptain(self) -> Player:
        return self.sortedPlayers[1]
    
    def getPlayerByName(self, pName: str) -> Player:
        for player in self.players:
            if(player.getName() == pName):
                return player
            
    def getPlayersByPosition(self, pPosition: Position):
        match pPosition:
            case Position.GKP:
                return Team.fromPlayerList(self.goalkeepers)
            case Position.DEF:
                return Team.fromPlayerList(self.defenders)
            case Position.FWD:
                return Team.fromPlayerList(self.forwarders)
            case Position.MID:
                return Team.fromPlayerList(self.midfielders)
            
    def getPlayersListByPosition(self, pPosition: Position) -> list[Player]:
        # TODO: Replace this with strategy pattern
        match pPosition:
            case Position.GKP:
                return self.goalkeepers
            case Position.DEF:
                return self.defenders
            case Position.FWD:
                return self.forwarders
            case Position.MID:
                return self.midfielders
            
    def getWorstIndex(self) -> int:
        return np.argmin(self.getPlayerScores())
    def getWorstByPosition(self, pPosition: Position) -> int:
        playersOfPosition = self.getPlayersByPosition(pPosition)
        return playersOfPosition.getWorstIndex()
    
    def getBestIndex(self) -> int:
        return np.argmax(self.getPlayerScores())
    def getBestByPosition(self, pPosition: Position) -> int:
        playersOfPosition = self.getPlayersByPosition(pPosition)
        return playersOfPosition.getBestIndex()
    def getWorstScore(self) -> float:
        return min([score for score in self.getPlayerScores()])
    
    def _repr_html_(self):
        return self.toHtml()

    def getHtmlHeader(self):
        htmlTxt = "<table>"
        htmlTxt += ("<tr>"
                    "<th>ID</th>"
                    "<th>Name</th>"
                    "<th>Cost</th>"
                    "<th>ICT Index</th>"
                    "<th>Total Points</th>"
                    "<th>Form</th>"
                    "<th>Average Fixture Difficulty</th>"
                    "<th>Normalised Average Fixture Difficulty</th>"
                    "<th>Current Fixture Difficulty</th>"
                    "<th>Position</th>"
                    "<th>Availability</th>"
                    "<th>Team</th>"
                    "<th>Captain</th>"
                    "<th>Vice Captain</th>"
                    "<th>Score</th>\n")
        return htmlTxt

    def toHtml(self):
        htmlTxt = self.getHtmlHeader()
        for plr in self.getPlayers():
            htmlTxt += plr.toHtmlRow() + "\n"
        htmlTxt += "</table>"
        return htmlTxt

    def toBenchTeam(self):
        return BenchTeam.fromExistingTeam(self)

    def __str__(self):
        string = ""
        totalScore = self.getTotalScore()
        string += f"\nTotal Score: {totalScore}\n\n"
        string += "Goalkeepers:"
        for gkp in self.goalkeepers:
            string += "\n- " + str(gkp)
        string += "\nDefenders:"
        for _def in self.defenders:
            string += "\n- " + str(_def)
        string += "\nAttackers:"
        for fwd in self.forwarders:
            string += "\n- " + str(fwd)
        string += "\nMidfielders:"
        for mid in self.midfielders:
            string += "\n- " + str(mid)

        return string
    
    def __sub__(self, pOther):
        thisPlayerNameSet = set(self.getPlayerNames())
        otherPlayerNameSet = set(pOther.getPlayerNames())
        uniquePlayerNameSet = thisPlayerNameSet - otherPlayerNameSet
        players = []
        for player in self.getPlayers():
            if(player.getName() in uniquePlayerNameSet):
                players.append(player)
        return Team.fromPlayerList(players)
    
    def __len__(self):
        return len(self.players)
    
    def recalculateFixtureDifficulty(self, pMatrix: FixtureDifficultyMatrix):
        for player in self.players:
            player.recalculateFixtureDifficulty(pMatrix)
    
    def calculateScore(self, pHeuristicMethod: str):
        """
        Calculate score using the heuristic method specified.
        Note that this uses column names for the heuristic method.
        """
        for player in self.getPlayers():
            match pHeuristicMethod:
                case "points_per_game":
                    player.calculateScorePPG()
                case "total_points":
                    player.calculateScoreTotalPoints()
                case "combined":
                    player.calculateCombinedScore()
                case _:
                    raise NotImplementedError(f"Heuristic method {pHeuristicMethod} is not yet implemented.")
        minPlayerScore = self.getWorstScore()
        for player in self.getPlayers():
            #if(player.getName() == "Bukayo Saka"):
            #    print(f"[DEBUG]: {player}, availability: {player.isAvailable()}")
            if(not player.isAvailable()):
                player.setToMinScore(minPlayerScore)
                
    def removePlayerByIndex(self, pIndex: int, pPosition: Position):
        match pPosition:
            case Position.GKP:
                self.goalkeepers.pop(pIndex)
            case Position.DEF:
                self.defenders.pop(pIndex)
            case Position.FWD:
                self.forwarders.pop(pIndex)
            case Position.MID:
                self.midfielders.pop(pIndex)
        self.updatePlayers()
        pass

    def addPlayer(self, pPlayer: Player):
        match pPlayer.getPosition():
            case Position.GKP:
                self.goalkeepers.append(pPlayer)
            case Position.DEF:
                self.defenders.append(pPlayer)
            case Position.FWD:
                self.forwarders.append(pPlayer)
            case Position.MID:
                self.midfielders.append(pPlayer)
        self.updatePlayers()

    def updatePlayers(self):
        self.players: list[Player] = self.goalkeepers + self.defenders + self.forwarders + self.midfielders
        self.sortedPlayers = sorted(self.players, reverse=True)
        self.updateCaptains()

    def makeCaptains(self, pPlayer: Player, pBestPlayer: Player, pViceCaptain: Player):
        pPlayer.setCaptain(False)
        pPlayer.setViceCaptain(False)
        if(pPlayer.getId() == pBestPlayer.getId()):
            pPlayer.setCaptain(True)
        if(pPlayer.getId() == pViceCaptain.getId()):
            pPlayer.setViceCaptain(True)

    def updateCaptains(self):
        bestIndex = self.getBestIndex()
        bestPlayer = self.players[bestIndex]
        viceCaptain = self.getViceCaptain()

        for player in self.players:
            self.makeCaptains(player, bestPlayer, viceCaptain)

class BenchTeam(Team):

    def __init__(self,
                 pGoalkeepers: list[Player],
                 pDefenders: list[Player],
                 pForwarders: list[Player],
                 pMidfielders = list[Player]):
        super(BenchTeam, self).__init__(
            pGoalkeepers,
            pDefenders,
            pForwarders,
            pMidfielders
        )

        nonBenchPlayers = []
        benchPlayers = []
        for pos in Position.listValues():
            playersSorted = sorted(self.getPlayersListByPosition(pos), reverse=True)
            worstPlayerIndex = np.argmin(playersSorted)
            worstPlayer = playersSorted.pop(worstPlayerIndex)
            benchPlayers.append(worstPlayer)
            nonBenchPlayers += playersSorted

        self.nonBenchPlayersList: list[Player] = nonBenchPlayers
        self.benchPlayersList: list[Player] = benchPlayers

        self.nonBenchPlayers = Team.fromPlayerList(nonBenchPlayers)
        self.benchPlayers = Team.fromPlayerList(benchPlayers)
        
        for gkp in self.benchPlayers.getGoalkeepers():
            gkp.setBenched(True)
        for _def in self.benchPlayers.getDefenders():
            _def.setBenched(True)
        for fwd in self.benchPlayers.getForwarders():
            fwd.setBenched(True)
        for mid in self.benchPlayers.getMidfielders():
            mid.setBenched(True)

        self.updatePlayers()

    @classmethod
    def fromExistingTeam(cls, pExistingTeam: Team):
        return cls(
            deepcopy(pExistingTeam.getGoalkeepers()),
            deepcopy(pExistingTeam.getDefenders()),
            deepcopy(pExistingTeam.getForwarders()),
            deepcopy(pExistingTeam.getMidfielders())
            )
    
    def __str__(self):
        string = "# Starting 11:\n"
        string += str(self.nonBenchPlayers) + "\n"
        string += "\n# Bench:\n"
        string += str(self.benchPlayers)
        return string
    
    def _repr_html_(self):
        return self.toHtml()
    
    def toHtml(self):
        htmlTxt = "<h1>Starting 11</h1>\n"
        htmlTxt += self.getHtmlHeader()
        for player in self.nonBenchPlayersList:
            htmlTxt += player.toHtmlRow() + "\n"
        htmlTxt += "</table>\n"
        htmlTxt += "<h1>Bench</h1>\n"
        htmlTxt += self.getHtmlHeader()
        for player in self.benchPlayersList:
            htmlTxt += player.toHtmlRow() + "\n"
        htmlTxt += "</table>"
        return htmlTxt
    
    def updateCaptains(self):
        bestIndex = self.getBestIndex()
        bestPlayer = self.nonBenchPlayersList[bestIndex]
        viceCaptain = self.getViceCaptain()

        for player in self.players:
            self.makeCaptains(player, bestPlayer, viceCaptain)

    def getViceCaptain(self) -> Player:
        playersSorted = sorted(self.nonBenchPlayersList, reverse=True)
        return playersSorted[1]

    def getBestIndex(self) -> int:
        scores = [player.getScore() for player in self.nonBenchPlayersList]
        return np.argmax(scores)
    def getWorstScore(self) -> float:
        minNonBench = min([player.getScore() for player in self.nonBenchPlayersList])
        minBench = min([player.getScore() for player in self.benchPlayers])
        minMin = min(minNonBench, minBench)
        return minMin
    def getBenchPlayerList(self) -> list[Player]:
        return self.benchPlayersList
    def getAllPlayerList(self) -> list[Player]:
        return self.getBenchPlayerList() + self.getPlayers()