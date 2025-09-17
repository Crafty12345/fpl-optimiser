import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
            pHeuristic: str, 
            pSolverMode: SolverMode, 
            verbose = False, 
            pLabel: str = None,
            pTuneHypers: bool = False):
        super().__init__(pHeuristic, pSolverMode, verbose, pLabel)
        # TODO: Seperate this into `fit()` argument 
        # TODO: Implement way of saving model
        self.allDummyColumns: set[str] = set()
        self.idNameDict: dict[int, str] = dict()

        xCols = ["id","ict_index", "gameweek", "season", "form", "starts_per_90", "position", "team", "opposing_team", "status"]
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

            currentTeams = set(datum["team"].apply(lambda x: "team_" + x))
            currentOpposingTeams = set(datum["opposing_team"].apply(lambda x: "opposing_team_" + x))
            currentStatuses = set(datum["status"].apply(lambda x: "status_" + x))
            self.allDummyColumns = self.allDummyColumns.union(currentPlayers)
            self.allDummyColumns = self.allDummyColumns.union(currentTeams)
            self.allDummyColumns = self.allDummyColumns.union(currentOpposingTeams)
            self.allDummyColumns = self.allDummyColumns.union(currentStatuses)

            tempDf = pd.concat([tempDf, opposingTeams])
        
        y: pd.DataFrame = tempDf[yCols]
        self.x: pd.DataFrame = tempDf.drop(columns=yCols)
        self.toDummyColumns = ["id", "position", "team", "opposing_team", "status"]
        x = self.setDummies(self.x)

        # TODO: Make sure there is a column for ALL unique players (across all gameweeks, all seasons)
        # This would fix a bug caused by differences in the number of players
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=19)
        
        yTrain: pd.Series = yTrain
        
        paramGrid = {
            "n_estimators": [500, 1000, 1500],
            #"criterion": ["entropy", "gini"],
            "min_samples_split": [5,10,15],
            "min_samples_leaf": [1,2,4],
            "max_depth": [10,20,30]
        }

        regressor = RandomForestRegressor(random_state=13)
        
        print("Fitting model...")
        regressor = regressor.fit(xTrain, np.ravel(yTrain.values))

        cvVerboseLevel = 0
        if(self.verbose):
            cvVerboseLevel = 3

        # WARNING: Hyperparameter tuning takes a long time
        if pTuneHypers:
            self.regressor = RandomizedSearchCV(regressor, paramGrid, scoring="r2", cv=3, n_jobs=-1, verbose=cvVerboseLevel, random_state=19)
            self.regressor = self.regressor.fit(xTrain, np.ravel(yTrain.values))
            print(f"best estimator = {self.regressor.best_estimator_}")
            print(f"best params = {self.regressor.best_params_}")
            estimator = self.regressor.best_estimator_
        else:
            bestParams = {
                "n_estimators": 500,
                "min_samples_split": 15,
                "min_samples_leaf": 2
            }
            self.regressor = RandomForestRegressor(random_state=13,
                                                   n_estimators=bestParams["n_estimators"],
                                                   min_samples_split=bestParams["min_samples_split"],
                                                   min_samples_leaf=bestParams["min_samples_leaf"])
            self.regressor = self.regressor.fit(xTrain, np.ravel(yTrain.values))
            estimator = self.regressor
        print("Finished fitting model.")

        treeLimit = 10
        assert treeLimit < len(estimator.estimators_)

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

        yPredicted = self.regressor.predict(xTest)
        mse = mean_squared_error(yTest, yPredicted)
        r2 = r2_score(yTest, yPredicted)
        print(f"r2={r2}")
        self.setAccuracy(r2)
        self.updatePredictionData()

        # TODO: add calculations to get fixtures for next week

    def precalcScores(self, pData, pGameweek, pSeason):
        return

    # TODO: repredict() method
    def updateScores(self, pScore: pd.DataFrame):
        _id = pScore["id"]
        #name = pScore["name"]
        _score = pScore["temp"]
        self.latestData.loc[self.latestData["id"]==_id, "score"] = _score

    def setDummies(self, pToDummy: pd.DataFrame) -> pd.DataFrame:
        result = pd.get_dummies(pToDummy, columns=self.toDummyColumns)
        xCols: set[str] = set(result.columns)
        for col in self.allDummyColumns:
            if col not in xCols:
                result[col] = 0
        result = result.reindex(sorted(result.columns), axis=1)
        return result

    def updatePredictionData(self, pSeason: int = None, pGameweek: int = None):
        latestSeason = self.x["season"].max()
        latestWeek = self.x.loc[self.x["season"]==latestSeason, "gameweek"].values[0]
        nextWeek = latestWeek + 1

        xForPredict: pd.DataFrame = self.x.copy()
        selectedGameweek: int = 0
        if pGameweek is None:
            selectedGameweek = latestWeek
        else:
            selectedGameweek = pGameweek
        xForPredict["gameweek"] = selectedGameweek

        selectedSeason: int = -1
        if pSeason is None:
            selectedSeason: int = latestSeason
        else:
            selectedSeason = pSeason
        xForPredict["season"] = selectedSeason

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
        if pGameweek is None:
            fixtureJsonRaw = fixtureJsonRaw[str(latestSeason)][str(nextWeek)]
        else:
            fixtureJsonRaw = fixtureJsonRaw[str(pSeason)][str(pGameweek)]
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
        dfCopy: pd.DataFrame = xForPredict
        dfCopy["temp"] = futureScores
        # TODO: Fix `PerformanceWarning`
        dfCopy["id"] = IDs
        dfCopy.apply(self.updateScores, axis=1)