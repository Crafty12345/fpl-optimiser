import pandas as pd
import os
import time
import json

from modules.team_solver import TeamSolver,SolverMode
from modules.lin_team_predicter import LinearTeamPredicter
from modules.forest_team_predicter import RFTeamPredicter
from modules.nn_team_predicter import NNTeamPredicter
from modules.team_evaluator import TeamEvaluator
from modules.new_team_solver import TeamTreeFactory,TeamTreeNode

import config
# TODO: Output into markdown document


CURRENT_DATE = config.CURRENT_DATE
startTime = time.perf_counter()
resultsDirectory = f"./results/{CURRENT_DATE}"

if(not os.path.exists(resultsDirectory)):
    os.makedirs(resultsDirectory)

calculationsFilename = f"{resultsDirectory}/scores_{CURRENT_DATE}.json"
summary_filename = f"{resultsDirectory}/summary_{CURRENT_DATE}.html"

json_filename = f"{resultsDirectory}/results_{CURRENT_DATE}.json"
nnFilename = f"{resultsDirectory}/model_{CURRENT_DATE}.pkl"

all_teams: list[TeamSolver] = []

team_solver = LinearTeamPredicter("combined",SolverMode.CHEAPEST_FIRST,verbose=True)
team_solver.fit()
print(team_solver.latestData.head())
all_teams.append(team_solver)

resultList: list[dict] = []

# TODO: Optimise team evaluating
for (i, team) in enumerate(all_teams):
    factory = TeamTreeFactory(team_solver.latestData)
    tree = factory.create()
    tree.nextBranch()
    bestPlayers = tree.bestBranch()
    for player in bestPlayers[0]:
        print(player)
    ...

endTime = time.perf_counter()
elapsedTime = endTime - startTime

print(f"Completed in {elapsedTime} seconds")