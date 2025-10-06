from enum import Enum
import pandas as pd
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix

class Position(Enum):
    GKP = 0,
    DEF = 1,
    FWD = 2,
    MID = 3
    def __str__(self):
        return self.name
    @staticmethod
    def fromString(pStr: str):
        positionNumMap = dict()
        positionNumMap = {"GKP": Position.GKP,
                          "DEF": Position.DEF,
                          "FWD": Position.FWD,
                          "MID": Position.MID}
        try:
            return positionNumMap[pStr.upper()]
        except KeyError:
            raise ValueError(f"Invalid position name: {pStr.upper()}")
    @classmethod
    def listValues(cls) -> list:
        return [cls.GKP, cls.DEF, cls.FWD, cls.MID]
        

class Player():
    def __init__(self,
                 id: int,
                 name: str,
                 cost: int,
                 ictIndex: float,
                 totalPoints: int,
                 pointsPerGame: float,
                 form: float,
                 position: Position,
                 teamName: str,
                 pCombinedScore: float,
                 pIsAvailable: bool,
                 pStartsPer90: float,
                 pScore: float = 0):
        self.id = id
        self.name = name
        self.cost = cost
        self.ictIndex = ictIndex
        self.totalPoints = totalPoints
        self.pointsPerGame = pointsPerGame
        self.form = form
        self.position = position
        self.teamName = teamName
        self.combinedScore = pCombinedScore
        self.score = pScore
        self.captain = False
        self.viceCaptain = False
        self.benchPlayer = False
        self.fixtureDifficulty = 0.0
        self.normalisedFixtureDifficulty = 0.0
        self.currentFixtureDifficulty = 0.0
        self.available = pIsAvailable
        self.startsPer90 = pStartsPer90
        pass

    @classmethod
    def fromName(cls, name: str, allPlayerData: pd.DataFrame):
        playerDf = allPlayerData.loc[allPlayerData["name"] == name]
        if(len(playerDf) == 0 ):
            raise LookupError(f"Unable to find player {name}")
        id = playerDf["id"].values[0]
        cost = playerDf["cost"].values[0]
        ictIndex = playerDf["ict_index"].values[0]
        totalPoints = playerDf["total_points"].values[0]
        pointsPerGame = playerDf["points_per_game"].values[0]
        form = playerDf["form"].values[0]
        combinedScore = playerDf["combined"].values[0]

        positionStr = playerDf["position"].values[0]
        position = Position.fromString(positionStr)
        teamName = playerDf["team"].values[0]
        isAvailable = playerDf["status"].values[0] == "a"
        print(name, playerDf["status"].values[0])
        startsPer90 = playerDf["play_percent"].values[0]
        return cls(id, name, cost, ictIndex, totalPoints, pointsPerGame, form, position, teamName, combinedScore, isAvailable, startsPer90)

    def getId(self): return self.id
    def getName(self): return self.name
    def getCost(self): return self.cost
    def getPosition(self): return self.position
    def getScore(self): return self.score
    def isCaptain(self): return self.captain
    def isViceCaptain(self): return self.viceCaptain
    def isBenched(self): return self.benchPlayer
    def isAvailable(self): return self.available
    def getCurrentDifficulty(self): return self.currentFixtureDifficulty
    
    def __str__(self):
        isCaptainStr = " (Captain) " if (self.isCaptain() and not self.isBenched()) else ""
        isViceCaptainStr = " (Vice Captain) " if (self.isViceCaptain() and not self.isBenched()) else ""
        string = self.name + isCaptainStr + isViceCaptainStr
        string += "\tScore: " + str(round(self.score,2)) + "\tCost: " + str(self.cost)
        string += "\tFixture Difficulty: " + str(self.normalisedFixtureDifficulty)
        return string
    
    def __gt__(self, pOther):
        return self.score > pOther.getScore()
    def __lt__(self, pOther):
        return self.score < pOther.getScore()
    
    def toHtmlRow(self) -> str:
        return ("<tr>"
                f"<td>{self.id}</td>"
                f"<td>{self.name}</td>"
                f"<td>{self.cost:.2f}</td>"
                f"<td>{self.ictIndex}</td>"
                f"<td>{self.totalPoints}</td>"
                f"<td>{self.form}</td>"
                f"<td>{self.fixtureDifficulty:.3f}</td>"
                f"<td>{self.normalisedFixtureDifficulty:.3f}</td>"
                f"<td>{self.currentFixtureDifficulty:.4f}</td>"
                f"<td>{self.position}</td>"
                f"<td>{self.available}</td>"
                f"<td>{self.teamName}</td>"
                f"<td>{self.captain}</td>"
                f"<td>{self.viceCaptain}"
                f"<td>{self.score:.2f}</td>"
                "<tr>")
    
    def setCaptain(self, pIsCaptain: bool): self.captain = pIsCaptain
    def setViceCaptain(self, pIsViceCaptain: bool): self.viceCaptain = pIsViceCaptain
    def setBenched(self, pIsBenchedPlayer: bool): self.benchPlayer = pIsBenchedPlayer
    def setCombinedScore(self, pNewScore: float): self.combinedScore = pNewScore
    def setToMinScore(self, pMinScore: float):
        self.score = min(pMinScore, 0)

    def recalculateFixtureDifficulty(self, pMatrix: FixtureDifficultyMatrix):
        self.fixtureDifficulty = pMatrix.getSimpleDifficulty(self.teamName)
        self.normalisedFixtureDifficulty = pMatrix.getNormalisedDifficulty(self.teamName)
        self.currentFixtureDifficulty = pMatrix.getCurrentDifficulty(self.teamName)

    def calculateScorePPG(self):
        """
        Calculate score using points per game as a heuristic
        """
        self.score = (self.form * self.pointsPerGame - self.normalisedFixtureDifficulty) * self.available * self.startsPer90

    def calculateScoreTotalPoints(self):
        """
        Calculate score using total points as a heuristic
        """
        self.score = (self.form * self.totalPoints - self.normalisedFixtureDifficulty) * self.available * self.startsPer90

    def calculateCombinedScore(self):
        """
        Calculate score using a combination of points per game, total points, and ICT Index
        """
        self.score = (self.form * self.combinedScore - self.normalisedFixtureDifficulty) * self.available * self.startsPer90
        pass