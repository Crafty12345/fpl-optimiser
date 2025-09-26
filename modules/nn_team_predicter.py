import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import json

from modules.team_predicter import TeamPredicter

class NNTeamPredicter(TeamPredicter):
    def __init__(self, pMode, verbose = False, pLabel = None):
        super().__init__("score", pMode, verbose, pLabel)
        self.toDummyColumns.append("id")
        self.toDummyColumns.append("position")

    def fit(self):
        allData = self.concatWeeks(self.setDummyCols)
        allData = self.fixDataTypes(allData)

        self.x = allData.drop(columns=self.yCols)

        x = self.x[self.xCols]
        x = self.setDummies(x)
        y = allData[self.yCols]
        self.device = torch.device("cuda")

        self.model = nn.Sequential(
            nn.Linear(len(x.columns), 324, device=self.device),
            nn.ReLU(),
            nn.Linear(324, 200, device=self.device),
            nn.ReLU(),
            nn.Linear(200, 1, device=self.device)
        ).to(self.device)
        lossFn = nn.MSELoss()
        optimiser = optim.Adam(self.model.parameters(), lr=0.0001)

        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=19, shuffle=True)
        xTrain = xTrain.to_numpy(dtype=np.float64)
        xTest = xTest.to_numpy(dtype=np.float64)
        yTrain = yTrain.to_numpy(dtype=np.float64)
        yTest = yTest.to_numpy(dtype=np.float64)
        
        xTrain = torch.tensor(xTrain, dtype=torch.float32, device=self.device)
        yTrain = torch.tensor(yTrain, dtype=torch.float32, device=self.device).reshape(-1, 1)
        xTest = torch.tensor(xTest, dtype=torch.float32, device=self.device)
        yTest = torch.tensor(yTest, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        numEpochs = 65
        batchSize = 10
        batchStart = torch.arange(0, len(xTrain), batchSize, device=self.device)

        bestMse = np.inf
        bestWeights = None
        history = []
        for epoch in range(numEpochs):
            self.model.train()
            with tqdm(batchStart, unit="batch", mininterval=0, disable=False) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    xBatch = xTrain[start:start+batchSize]
                    yBatch = yTrain[start:start+batchSize]

                    yPred = self.model(xBatch)
                    loss = lossFn(yPred, yBatch)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    bar.set_postfix(mse=float(loss))
            self.model.eval()
            yPred = self.model(xTest)
            mse = lossFn(yPred, yTest)
            mse = float(mse)
            history.append(mse)
            if mse < bestMse:
                bestMse = mse
                bestWeights = deepcopy(self.model.state_dict())
        self.model.load_state_dict(bestWeights)

    def loadModel(self, pFilename: str) -> None:
        with open(pFilename, "rb") as f:
            self.model = torch.load(f, weights_only=False)
        
        allData = self.concatWeeks(self.setDummyCols)
        allData = self.fixDataTypes(allData)

        self.x = allData[self.xCols]

    def saveModel(self, pFilename: str) -> None:
        with open(pFilename, "wb+") as f:
            torch.save(self.model, f)

    def precalcScores(self, pData, pGameweek, pSeason):
        return
    def updatePredictionData(self, pSeason: int, pTargetSeason: int, pGameweek: int, pTargetWeek: int) -> None:
        
        latestDataLoc = (self.x["gameweek"] == pGameweek) & (self.x["season"] == pTargetSeason)
        xForPredict = self.x.copy()
        xForPredict = xForPredict.loc[latestDataLoc]
        
        ids = xForPredict["id"]
        self.latestData = xForPredict.copy()

        xForPredict = xForPredict[self.xCols]

        fixtureJsonRaw = dict()
        with open(f"./data/fixtures.json", "r") as f:
            fixtureJsonRaw = json.load(f)

        fixtureJsonRaw = fixtureJsonRaw[str(pTargetSeason)][str(pTargetWeek)]
        self.fixtureDf: pd.DataFrame = pd.DataFrame.from_records(fixtureJsonRaw)

        opposingTeams = xForPredict["team"].apply(lambda x: self.getOpposingTeam(x, self.fixtureDf))
        xForPredict["opposing_team"] = opposingTeams
        xForPredict["gameweek"] = pTargetWeek
        xForPredict["season"] = pTargetSeason

        xForPredict = self.setDummies(xForPredict)

        xForPredict = torch.tensor(
            xForPredict.to_numpy(np.float32),
            dtype=torch.float32,
            device=self.device
        )
        predictedScores = self.model(xForPredict).cpu().detach().numpy()

        self.latestData["opposing_team"] = opposingTeams
        self.latestData["gameweek"] = pTargetWeek
        self.latestData["season"] = pTargetSeason
        self.latestData["score"] = -np.inf
        for (_id, score) in zip(ids, predictedScores):
            self.latestData.loc[self.latestData["id"]==_id, "score"] = score