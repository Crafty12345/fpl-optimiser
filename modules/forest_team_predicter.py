import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import json
import line_profiler
import time
import math

from modules.team_solver import TeamSolver, SolverMode
from modules.team_predicter import TeamPredicter
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix

NUM_TREES = 500

class RFTeamPredicter(TeamPredicter):
    '''
    Team Predicter using Random Forest Regression
    '''
    def __init__(self,
            pSolverMode: SolverMode, 
            verbose = False, 
            pLabel: str = None,
            pFreeHit = False):
        super().__init__("score", pSolverMode, verbose, pLabel, pFreeHit=pFreeHit)
        # TODO: Seperate this into `fit()` argument 
        # TODO: Implement way of saving model
        # TODO: Improve accuracy by seperating different positions into different models
        self.models: dict[str, RandomForestRegressor] = dict()

        # TODO: add calculations to get fixtures for next week

    def precalcScores(self, pData, pGameweek, pSeason):
        return
    
    def getFixtureDiff(self, pMatrix: FixtureDifficultyMatrix, pDatum: pd.Series, pAheadSteps: int):
        pMatrix.precomputeFixtureDifficulty(0, pDatum["gameweek"], pAheadSteps, pDatum["season"], 1.0)
        result = pMatrix.calcNormalisedDifficulty(pDatum["team"], pDatum["opposing_team"], -1.0, 1.0)
        return result

    def fit(self):
        tempDf = self.concatWeeks(self.setDummyCols)
        tempDf: pd.DataFrame = self.fixDataTypes(tempDf)
        tempDf = self.assignTime(tempDf)

        #print(tempDf["season"].is_monotonic_increasing)

        fixtureMatrix = FixtureDifficultyMatrix()

        y: pd.DataFrame = tempDf[self.yCols]
        self.x: pd.DataFrame = tempDf.drop(columns=self.yCols)
        self.x["fixture_dif"] = self.x.apply(lambda x: self.getFixtureDiff(fixtureMatrix, x, 0), axis=1)
        tempX = self.x[self.xCols]

        allR2s = []
        maxT = tempDf["t"].max()

        for position in tqdm(["GKP", "FWD", "MID", "DEF"], desc="Fitting models"):
            positionLoc = tempX["position"]==position
            # TODO: Maybe stop copying entire DF every time?
            weightDecay = 0.66
            #weights = np.exp(tempDf[positionLoc]["t"])
            weights = tempDf[positionLoc]["t"].apply(lambda x: self.decayTime(x, maxT-1, weightDecay))
            xCopy = tempX.copy()[positionLoc]
            xCopy = xCopy.drop(columns=["position"])
            #xCopy["weight"] = weights
            x = self.setDummies(xCopy)
            xTrain = x
            yOfPos = y.loc[positionLoc]
            #xTrain, xTest, yTrain, yTest = train_test_split(x, yOfPos, test_size=0.2, random_state=19)
        
            yTrain = yOfPos
            #yTrain: pd.Series = yTrain
            regressor = RandomForestRegressor(random_state=13, n_jobs=-1, n_estimators=NUM_TREES, verbose=1)
            #regressor = xgb.XGBRFRegressor(random_state=19)
            #xTrain = xTrain.drop(columns=["weight"])
            #xTest = xTest.drop(columns=["weight"])

            # Interestingly, accuracy seems to be MUCH higher when hyperparameters are NOT tuned!!!
            regressor = regressor.fit(xTrain, np.ravel(yTrain.values), sample_weight=weights)
            #yPredicted = regressor.predict(xTest)
            yPredicted = regressor.predict(xTrain)
            # TODO: Calculate adjusted r2
            r2 = r2_score(yTrain, yPredicted)
            allR2s.append(r2)
            self.models[position] = regressor

        meanR2 = sum(allR2s) / len(allR2s)
        self.accuracy = meanR2
        print(f"r2={meanR2}")

        self.setAccuracy(meanR2)


    def decayTime(self, t, pMaxT, pDecay: float = 0.02) -> float:
        # https://stats.stackexchange.com/questions/454415/how-to-account-for-the-recency-of-the-observations-in-a-regression-problem
        # https://jackbakerds.com/posts/upweight-recent-observations-regression-classification/
        #return math.exp(-pDecay * (pMaxT - t))
        return pDecay ** (pMaxT - t)

    def valueFromDummies(self, pDummies: pd.Series, pColumn: str) -> str:
        columnPrefix: str = f"{pColumn}_"
        for column in pDummies.index:
            if(column.startswith(columnPrefix)):
                if (pDummies[column]):
                    columnSplit = column.split(columnPrefix)
                    assert len(columnSplit) == 2
                    return columnSplit[1]
        raise KeyError(pColumn)

    # TODO: repredict() method
    def updateScores(self, pScore: pd.Series):
        name = pScore["name"]

        #_id = int(pScore["id"])
        #print(pScore["opposing_team"])
        #_oppTeam = self.valueFromDummies(pScore, "opposing_team")
        _oppTeam = pScore["opposing_team"]
        #name = pScore["name"]
        _score = pScore["temp"]
        #print(self.latestData["id"])

        # Fixes a bug where one player can have multiple IDs
        self.latestData.loc[self.latestData["name"]==name, "score"] = _score
        self.latestData.loc[self.latestData["name"]==name, "opposing_team"] = ",".join(_oppTeam)

    def sigmoid(self, pX: float) -> float:
        return 1.0 / (1 + math.exp(-pX))

    @line_profiler.profile
    def updatePredictionData(self, pSeason: int, pTargetSeason: int, pGameweek: int, pTargetWeek: int) -> None:
        """
        :param int pSeason: The season to default to if pTargetSeason has not happened yet
        :param int pTargetSeason: The season to predict values for
        :param int pGameweek: The gameweek to default to if pTargetWeek has not happened yet
        :param int pTargetWeek: The gameweek to predict values for
        """

        if (len(self.models) == 0):
            raise ValueError("Regressor has not been fitted yet. Remember to call fit().")

        xForPredict: pd.DataFrame = self.x.copy()[self.xCols]
        xWithNames: pd.DataFrame = self.x.copy()["name"]
        selectedGameweek: int = pGameweek
        predictionWeek: int = pTargetWeek

        assert predictionWeek > -1
        assert selectedGameweek > -1

        selectedSeason: int = pSeason
        predictionSeason: int = pTargetSeason
        
        assert selectedSeason > -1
        assert selectedGameweek > -1

        latestDataLoc = (self.x["gameweek"] == selectedGameweek) & (self.x["season"] == selectedSeason)
        xForPredict = xForPredict.loc[latestDataLoc]
        xWithNames = xWithNames.loc[latestDataLoc]

        if len(xForPredict) < 1:
            print(f"An error occured when trying to process week {selectedGameweek} of season {selectedSeason}")
            print(self.x.head(10))
            print(self.x.tail(10))
        assert len(xForPredict) > 0

        fixtureJsonRaw = dict()
        with open(f"./data/fixtures.json", "r") as f:
            fixtureJsonRaw = json.load(f)

        fixtureJsonRaw = fixtureJsonRaw[str(predictionSeason)][str(predictionWeek)]
        self.fixtureDf: pd.DataFrame = pd.DataFrame.from_records(fixtureJsonRaw)

        opposingTeams = xForPredict["team"].apply(lambda x: self.getOpposingTeam(x, self.fixtureDf))
        xForPredict["home_game"] = xForPredict["team"].apply(lambda x: self.isHomeGame(x, self.fixtureDf))
        xForPredict["opposing_team"] = opposingTeams
        xForPredict["gameweek"] = predictionWeek
        xForPredict["season"] = predictionSeason

        fixtureMatrix = FixtureDifficultyMatrix()

        xForPredict["fixture_dif"] = xForPredict.apply(lambda x: self.getFixtureDiff(fixtureMatrix, x, 5), axis=1)
        xForPredict["fixture_dif"] = xForPredict["fixture_dif"].astype(np.float32)

        tempDf = self.concatWeeks()
        #print(len(tempDf["name"].unique()))
        tempDf: pd.DataFrame = self.fixDataTypes(tempDf)
        tempDf = self.assignTime(tempDf)
        tempDf["improvement"] = 0.0
        #print(tempDf.loc[tempDf["name"]=="Erling Haaland"][["name", "t", "points_this_week"]])
        # Fix a bug where some players have multiple IDs
        for name in tempDf["name"].unique():
        #name = "Erling Haaland"
            playerData = tempDf.loc[(tempDf["name"]==name) & (tempDf["season"] == predictionSeason)]
            if len(playerData) > 0:
                x: pd.Series = playerData["t"]
                # Normalise x so that all x values start from 0
                x -= x.min()
                x = x.values.reshape(-1, 1)
                y = playerData["points_this_week"].values.reshape(-1, 1)
                tempModel = LinearRegression()
                tempModel = tempModel.fit(x, y)
                tempDf.loc[tempDf["name"]==name, "improvement"] = tempModel.coef_

        #tempDf["improvement"] += abs(tempDf["improvement"].min())
        #print(tempDf.loc[tempDf["name"]=="Erling Haaland", ["name", "improvement"]])
        #expVals = np.exp(tempDf["improvement"].values)
        #print(f"expVals={expVals}")
        #expSum = expVals.sum()
        #print(f"expSum={expSum}")
        #weights = expVals / expSum
        #tempDf["coef"] = weights
        #print(tempDf.loc[tempDf["name"]=="Erling Haaland", ["name", "coef", "improvement"]])
        #assert False
        #tempDf["coef"] = tempDf["improvement"] / tempDf["improvement"].max()
        #print(tempDf.sample(n=20))
        defaultCoef = 0.0
        #tempDf["coef"] = defaultCoef

        namesSet: set[str] = set(tempDf["name"].values)
        #possibleIDs: set[int] = set(dfCopy["id"].values)
        stdDev: float = np.std(tempDf["points_this_week"].values)
        # #dfCopy["temp"] *= dfCopy["coef"]
        
        for position in ["GKP", "DEF", "MID", "FWD"]:
            loc = xForPredict["position"] == position
            xOfPosition = xForPredict.loc[loc]
            namesOfPosition = xWithNames.loc[loc]
            assert len(xOfPosition) == len(namesOfPosition)
            opposingTeams = xOfPosition["opposing_team"]
            xOfPosition = xOfPosition.drop(columns=["opposing_team"])
            xOfPosition = xOfPosition.drop(columns=["position"])
            xOfPosition = self.setDummies(xOfPosition)
            futureScores = self.models[position].predict(xOfPosition)
            dfCopy: pd.DataFrame = xOfPosition.copy()
            #print(len(dfCopy), len(namesOfPosition))
            dfCopy = dfCopy.assign(name=namesOfPosition)
            #dfCopy["name"] = namesOfPosition
            #idsOfLoc = IDs[loc]
            for name in dfCopy["name"]:
                improvRate = 0.0
                if name in namesSet:
                    improvRate = tempDf.loc[tempDf["name"]==name, "improvement"].values[0]
                    #print(improvRate)
                    #assert False
                    #coef = tempDf.loc[tempDf["id"]==_id, "coef"].values[0]
                    #print(coef)
                dfCopy["temp"] = futureScores + (stdDev * improvRate)

            dfCopy["opposing_team"] = opposingTeams
            #print(dfCopy["opposing_team"])
        # TODO: Fix `PerformanceWarning`
        #dfCopy["id"] = IDs
            dfCopy = dfCopy.apply(self.updateScores, axis=1)

        self.latestData["gameweek"] = predictionWeek
        self.latestData["season"] = predictionSeason
        #self.latestData["score"] *= self.latestData["play_percent"]
        #self.latestData = self.latestData.set_index(self.x["id"])