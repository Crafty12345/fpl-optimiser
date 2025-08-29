import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_graphviz
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
            pToPredict: list[int] | None = None):
        super().__init__(pHeuristic, pSolverMode, verbose, pLabel)

        xCols = ["id", "ict_index", "gameweek", "season", "form", "starts_per_90", "position", "team", "opposing_team", "status"]
        yCols = ["total_points"]
        allCols = xCols + yCols
        tempDf = pd.DataFrame(columns=allCols)
        for datum in self.allData:
            opposingTeams = datum.copy()
            opposingTeams = opposingTeams[allCols]
            tempDf = pd.concat([tempDf, opposingTeams])
        
        y: pd.DataFrame = opposingTeams[yCols]
        x: pd.DataFrame = opposingTeams.drop(columns=yCols)
        xForPredict: pd.DataFrame = x.copy()
        x = pd.get_dummies(x)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=19)
        
        yTrain: pd.Series = yTrain
        regressor = RandomForestRegressor(random_state=13)
        
        regressor = regressor.fit(xTrain, np.ravel(yTrain.values))

        treeLimit = 10
        assert treeLimit < len(regressor.estimators_)
        for i in range(treeLimit):
            tree = regressor.estimators_[i]
            export_graphviz(tree, f"trees/tree{i}.dot", feature_names=x.columns, label="all", filled=True)
            subprocess.run(["rm", f"trees/tree{i}.dot"])
            subprocess.run(["dot", "-Tpng", f"trees/tree{i}.dot", "-o", f"trees/tree{i}.png"])
        yPredicted = regressor.predict(xTest)
        mse = mean_squared_error(yTest, yPredicted)
        r2 = r2_score(yTest, yPredicted)
        self.setAccuracy(r2)


        # TODO: add calculations to get fixtures for next week
        latestWeek = x["gameweek"].max()
        latestSeason = x["season"].max()
        latestDataLoc = x["gameweek"] == latestWeek
        xForPredict = xForPredict.loc[latestDataLoc]

        nextWeek = latestWeek + 1
        xForPredict["gameweek"] = nextWeek
        fixtureJsonRaw = dict()
        with open(f"./data/fixtures.json", "r") as f:
            fixtureJsonRaw = json.load(f)
        fixtureJsonRaw = fixtureJsonRaw[str(latestSeason)][str(nextWeek)]
        fixtureDf: pd.DataFrame = pd.DataFrame.from_records(fixtureJsonRaw)
        with open("./data/team_translation_table.json", "r") as f:
            teamDataJson: dict = json.load(f)
        teamDataDf: pd.DataFrame = pd.DataFrame.from_records(teamDataJson[str(latestSeason)])
        opposingTeams = xForPredict["team"].apply(lambda x: getOpposingTeam(x, fixtureDf))
        xForPredict["opposing_team"] = opposingTeams
        xForPredict = pd.get_dummies(xForPredict)
        
        futureScores = regressor.predict(xForPredict)
        dfCopy: pd.DataFrame = xForPredict
        dfCopy["temp"] = futureScores
        dfCopy.apply(self.updateScores, axis=1)


    def updateScores(self, pScore: pd.DataFrame):
        _id = pScore["id"]
        _score = pScore["temp"]
        self.latestData.loc[self.latestData["id"]==_id, "score"] = _score

    def precalcScores(self, pData, pGameweek, pSeason):
        pass