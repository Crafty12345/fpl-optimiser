import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import numpy as np
import json
import subprocess

from modules.team_solver import TeamSolver, SolverMode

def getOpposingTeam(pTeam: str, pFixtureDf: pd.DataFrame) -> str:
    result = pFixtureDf.loc[pFixtureDf["home_team"]==pTeam]["away_team"]
    if len(result) == 0:
        result = pFixtureDf.loc[pFixtureDf["away_team"]==pTeam]["home_team"]
    if len(result) == 0:
        return "UNK"
    else:
        return result.values[0]

class RFTeamPredicter(TeamSolver):
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
        self.allDummyColumns: set[str] = set()
        self.idNameDict: dict[int, str] = dict()
        self.regressor: RandomForestRegressor = None

        # TODO: add calculations to get fixtures for next week

    def precalcScores(self, pData, pGameweek, pSeason):
        return
    
    def fit(self):
        xCols = ["id","ict_index", "team", "gameweek", "season", "form", "position", "opposing_team", "play_percent", "clean_sheets", "expected_goals", "status"]
        yCols = ["points_this_week"]
        allCols = xCols + yCols
        tempDf = pd.DataFrame(columns=allCols)
        for datum in self.allData:
            opposingTeams = datum.copy()
            opposingTeams = opposingTeams[allCols]

            playerIds = zip(datum["id"].values, datum["name"])
            for key, val in playerIds:
                self.idNameDict[key] = val

            currentPlayers = set(datum["id"].apply(lambda x: "id_" + str(x)))
            self.allDummyColumns = self.allDummyColumns.union(currentPlayers)

            currentTeams = set(datum["team"].apply(lambda x: "team_" + x))
            self.allDummyColumns = self.allDummyColumns.union(currentTeams)

            currentOpposingTeams = set(datum["opposing_team"].apply(lambda x: "opposing_team_" + x))
            self.allDummyColumns = self.allDummyColumns.union(currentOpposingTeams)

            currentStatuses = set(datum["status"].apply(lambda x: "status_" + x))
            self.allDummyColumns = self.allDummyColumns.union(currentStatuses)

            tempDf = pd.concat([tempDf, opposingTeams])
        
        y: pd.DataFrame = tempDf[yCols]
        self.x: pd.DataFrame = tempDf.drop(columns=yCols)
        self.x["gameweek"] = self.x["gameweek"].astype(np.uint16)
        self.x["season"] = self.x["season"].astype(np.uint16)

        self.toDummyColumns = ["id", "position", "team", "opposing_team", "status"]
        x = self.setDummies(self.x)

        # TODO: Make sure there is a column for ALL unique players (across all gameweeks, all seasons)
        # This would fix a bug caused by differences in the number of players
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=19)
        
        yTrain: pd.Series = yTrain

        regressor = RandomForestRegressor(random_state=13, n_jobs=-1)
        #regressor = xgb.XGBRFRegressor(random_state=19)

        # Interestingly, accuracy seems to be MUCH higher when hyperparameters are NOT tuned!!!
        print("Fitting model...")
        self.regressor = regressor.fit(xTrain, np.ravel(yTrain.values))
        self.regressor = self.regressor.fit(xTrain, np.ravel(yTrain.values))
        
        featureImportances = self.regressor.feature_importances_
        featureNames = self.regressor.feature_names_in_
        tempDf = pd.DataFrame({"column": featureNames, "importance": featureImportances}).sort_values(by="importance", ascending=False)
        print("Best parameters:")
        print(tempDf.head(20))
        estimator = self.regressor
        print("Finished fitting model.")

        treeLimit = 5
        assert treeLimit < len(estimator.estimators_)

        print("Saving trees...")
        featureNames = []
        for column in x.columns:
            toAdd: str = column
            if (column.startswith("id_")):
                actualId = int(column.split("id_")[1])
                toAdd = self.idNameDict[actualId]
            featureNames.append(toAdd)

        for i in range(treeLimit):
            tree = estimator.estimators_[i]
            export_graphviz(tree, f"trees/tree{i}.dot", feature_names=featureNames, label="all", filled=True)
            subprocess.run(["dot", "-Tpng", f"trees/tree{i}.dot", "-o", f"trees/tree{i}.png"])
            subprocess.run(["rm", f"trees/tree{i}.dot"])
        print("Finished saving trees")

        yPredicted = self.regressor.predict(xTest)
        mse = mean_squared_error(yTest, yPredicted)
        r2 = r2_score(yTest, yPredicted)
        print(f"r2={r2}")
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
        _id = int(self.valueFromDummies(pScore, "id"))
        _oppTeam = self.valueFromDummies(pScore, "opposing_team")
        #name = pScore["name"]
        _score = pScore["temp"]
        dataLoc = self.latestData["id"]==_id

        # TODO: Add way to get actual opposing team from `pScore`
        self.latestData.loc[dataLoc, "score"] = _score
        self.latestData.loc[dataLoc, "opposing_team"] = _oppTeam

    def setDummies(self, pToDummy: pd.DataFrame) -> pd.DataFrame:
        result = pd.get_dummies(pToDummy, columns=self.toDummyColumns)
        colsToAdd = list()
        xCols: set[str] = set(result.columns)
        for col in self.allDummyColumns:
            if col not in xCols:
                colsToAdd.append(col)
        result = pd.concat([result, pd.DataFrame([{col:0 for col in colsToAdd}])], axis=1)
        result = result.sort_index(axis=1)
        return result

    def updatePredictionData(self, pSeason: int, pTargetSeason: int, pGameweek: int, pTargetWeek: int) -> None:
        """
        :param int pSeason: The season to default to if pTargetSeason has not happened yet
        :param int pTargetSeason: The season to predict values for
        :param int pGameweek: The gameweek to default to if pTargetWeek has not happened yet
        :param int pTargetWeek: The gameweek to predict values for
        """

        if (self.regressor is None):
            raise ValueError("Regressor has not been fitted yet. Remember to call fit().")

        xForPredict: pd.DataFrame = self.x.copy()
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
        with open("./data/team_translation_table.json", "r") as f:
            teamDataJson: dict = json.load(f)
        #teamDataDf: pd.DataFrame = pd.DataFrame.from_records(teamDataJson[str(latestSeason)])
        opposingTeams = xForPredict["team"].apply(lambda x: getOpposingTeam(x, self.fixtureDf))
        xForPredict["opposing_team"] = opposingTeams
        xForPredict = self.setDummies(xForPredict)
        #if (len(xForPredict.columns) != len(self.tempColumns)):
        #    print(f"xForPredict.columns={xForPredict.columns}")
        #    print(f"Column mismatch: xForPredict has {len(xForPredict.columns)}, and pd.get_dummies(self.x) has {len(pd.get_dummies(self.x).columns)}")
        
        futureScores = self.regressor.predict(xForPredict)
        dfCopy: pd.DataFrame = xForPredict.copy()
        dfCopy["temp"] = futureScores
        # TODO: Fix `PerformanceWarning`
        #dfCopy["id"] = IDs
        dfCopy = dfCopy.apply(self.updateScores, axis=1)

        self.latestData["gameweek"] = predictionWeek
        self.latestData["season"] = predictionSeason
        #self.latestData["score"] *= self.latestData["play_percent"]
        #self.latestData = self.latestData.set_index(self.x["id"])