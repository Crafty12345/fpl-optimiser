import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tqdm import tqdm
import numpy as np
import json

from modules.team_solver import TeamSolver, SolverMode
from modules.team_predicter import TeamPredicter
from modules.fixture_difficulty_matrix import FixtureDifficultyMatrix

NUM_TREES = 2000

class RFTeamPredicter(TeamPredicter):
    '''
    Team Predicter using Random Forest Regression
    '''
    def __init__(self,
            pSolverMode: SolverMode, 
            verbose = False, 
            pLabel: str = None):
        super().__init__("score", pSolverMode, verbose, pLabel)
        # TODO: Seperate this into `fit()` argument 
        # TODO: Implement way of saving model
        # TODO: Improve accuracy by seperating different positions into different models
        self.models: dict[str, RandomForestRegressor] = dict()

        # TODO: add calculations to get fixtures for next week

    def precalcScores(self, pData, pGameweek, pSeason):
        return
    
    def getFixtureDiff(self, pMatrix: FixtureDifficultyMatrix, pDatum: pd.Series):
        pMatrix.precomputeFixtureDifficulty(0, pDatum["gameweek"], 3, pDatum["season"], 1.0)
        result = pMatrix.calcNormalisedDifficulty(pDatum["team"], pDatum["opposing_team"], -1.0, 1.0)
        return result

    def fit(self):
        tempDf = self.concatWeeks(self.setDummyCols)
        tempDf = self.fixDataTypes(tempDf)
        
        self.fixtureMatrix = FixtureDifficultyMatrix()

        y: pd.DataFrame = tempDf[self.yCols]
        self.x: pd.DataFrame = tempDf.drop(columns=self.yCols)
        self.x["fixture_dif"] = self.x.apply(lambda x: self.getFixtureDiff(self.fixtureMatrix, x), axis=1)
        self.x["fixture_dif"] = self.x["fixture_dif"].astype(np.float32)
        tempX = self.x[self.xCols]

        # TODO: Make sure there is a column for ALL unique players (across all gameweeks, all seasons)
        # This would fix a bug caused by differences in the number of players
        allR2s = []
        for position in tqdm(["GKP", "FWD", "MID", "DEF"], desc="Fitting models"):
            positionLoc = tempX["position"]==position
            xCopy = tempX.copy()[positionLoc]
            xCopy = xCopy.drop(columns=["position"])
            x = self.setDummies(xCopy)
            yOfPos = y.loc[positionLoc]
            xTrain, xTest, yTrain, yTest = train_test_split(x, yOfPos, test_size=0.2, random_state=19)
        
            yTrain: pd.Series = yTrain
            regressor = RandomForestRegressor(random_state=13, n_jobs=-1, n_estimators=NUM_TREES, verbose=1)
            #regressor = xgb.XGBRFRegressor(random_state=19)

            # Interestingly, accuracy seems to be MUCH higher when hyperparameters are NOT tuned!!!
            regressor = regressor.fit(xTrain, np.ravel(yTrain.values))
            yPredicted = regressor.predict(xTest)
            r2 = r2_score(yTest, yPredicted)
            allR2s.append(r2)
            self.models[position] = regressor

        meanR2 = sum(allR2s) / len(allR2s)
        print(f"r2={meanR2}")

        self.setAccuracy(r2)

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
        _id = int(pScore["id"])
        #_id = int(self.valueFromDummies(pScore, "id"))
        _oppTeam = self.valueFromDummies(pScore, "opposing_team")
        #name = pScore["name"]
        _score = pScore["temp"]
        dataLoc = self.latestData["id"]==_id

        # TODO: Add way to get actual opposing team from `pScore`
        self.latestData.loc[dataLoc, "score"] = _score
        self.latestData.loc[dataLoc, "opposing_team"] = _oppTeam

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
        IDs = xForPredict["id"]
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
        xForPredict["opposing_team"] = opposingTeams
        xForPredict["gameweek"] = predictionWeek
        xForPredict["season"] = predictionSeason

        xForPredict["fixture_dif"] = xForPredict.apply(lambda x: self.getFixtureDiff(self.fixtureMatrix, x), axis=1)
        xForPredict["fixture_dif"] = xForPredict["fixture_dif"].astype(np.float32)
        
        for position in ["GKP", "DEF", "MID", "FWD"]:
            loc = xForPredict["position"] == position
            xOfPosition = xForPredict.loc[loc]
            xOfPosition = xOfPosition.drop(columns=["position"])
            xOfPosition = self.setDummies(xOfPosition)
            futureScores = self.models[position].predict(xOfPosition)
            dfCopy: pd.DataFrame = xOfPosition.copy()
            dfCopy["temp"] = futureScores
        # TODO: Fix `PerformanceWarning`
        #dfCopy["id"] = IDs
            dfCopy = dfCopy.apply(self.updateScores, axis=1)

        self.latestData["gameweek"] = predictionWeek
        self.latestData["season"] = predictionSeason
        #self.latestData["score"] *= self.latestData["play_percent"]
        #self.latestData = self.latestData.set_index(self.x["id"])