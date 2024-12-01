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
                 teamName: str):
        self.id = id
        self.name = name
        self.cost = cost
        self.ictIndex = ictIndex
        self.totalPoints = totalPoints
        self.pointsPerGame = pointsPerGame
        self.form = form
        self.position = position
        self.teamName = teamName
        self.score = 0
        self.captain = False
        self.viceCaptain = False
        self.benchPlayer = False
        matrix = FixtureDifficultyMatrix()
        self.normalisedFixtureDifficulty = matrix.getFixtureDifficulty(self.teamName)
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

        positionStr = playerDf["position"].values[0]
        position = Position.fromString(positionStr)
        teamName = playerDf["team"].values[0]
        return cls(id, name, cost, ictIndex, totalPoints, pointsPerGame, form, position, teamName)

    def getId(self): return self.id
    def getName(self): return self.name
    def getCost(self): return self.cost
    def getPosition(self): return self.position
    def getScore(self): return self.score
    def isCaptain(self): return self.captain
    def isViceCaptain(self): return self.viceCaptain
    def isBenched(self): return self.benchPlayer
    
    def __str__(self):
        isCaptainStr = " (Captain) " if (self.isCaptain() and not self.isBenched()) else ""
        isViceCaptainStr = " (Vice Captain) " if (self.isViceCaptain() and not self.isBenched()) else ""
        string = self.name + isCaptainStr + isViceCaptainStr
        string += "\tScore: " + str(round(self.score,2)) + "\tCost: " + str(self.cost)
        string += "\tFixture Difficulty: " + str(self.normalisedFixtureDifficulty)
        return string
    
    def __gt__(self, pOther):
        return self.score > pOther.score
    def __lt__(self, pOther):
        return self.score < pOther.score
    
    def setCaptain(self, pIsCaptain: bool): self.captain = pIsCaptain
    def setViceCaptain(self, pIsViceCaptain: bool): self.viceCaptain = pIsViceCaptain
    def setBenched(self, pIsBenchedPlayer: bool): self.benchPlayer = pIsBenchedPlayer
    
    def calculateScorePPG(self):
        """
        Calculate score using points per game as a heuristic
        """
        self.score = self.form * self.pointsPerGame + self.normalisedFixtureDifficulty

    def calculateScoreTotalPoints(self):
        """
        Calculate score using total points as a heuristic
        """
        self.score = self.form * self.totalPoints + self.normalisedFixtureDifficulty