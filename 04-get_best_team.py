import pandas as pd
import os
import time
import json
import cProfile

from modules.team_solver import TeamSolver,SolverMode
from modules.lin_team_predicter import LinearTeamPredicter
from modules.forest_team_predicter import RFTeamPredicter
from modules.nn_team_predicter import NNTeamPredicter
from modules.team_evaluator import TeamEvaluator
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
all_teams.append(team_solver)

team_solver = LinearTeamPredicter("combined",SolverMode.HIGHEST_COST_FIRST,verbose=True)
team_solver.fit()
all_teams.append(team_solver)

rfTeam = RFTeamPredicter(SolverMode.CHEAPEST_FIRST,verbose=True)
rfTeam.fit()
rfTeam.updatePredictionData(config.CURRENT_SEASON, config.CURRENT_SEASON, config.CURRENT_GAMEWEEK, config.CURRENT_GAMEWEEK+1)
all_teams.append(rfTeam)

rfTeam = RFTeamPredicter(SolverMode.HIGHEST_COST_FIRST,verbose=True)
rfTeam.fit()
rfTeam.updatePredictionData(config.CURRENT_SEASON, config.CURRENT_SEASON, config.CURRENT_GAMEWEEK, config.CURRENT_GAMEWEEK+1)
all_teams.append(rfTeam)

# rfTeam = RFTeamPredicter(SolverMode.HIGHEST_COST_FIRST,verbose=True, pLabel="Random Forest with Free Hit", pFreeHit=True)
# rfTeam.fit()
# rfTeam.updatePredictionData(config.CURRENT_SEASON, config.CURRENT_SEASON, config.CURRENT_GAMEWEEK, config.CURRENT_GAMEWEEK+1)
# all_teams.append(rfTeam)

# nnTeam = NNTeamPredicter(SolverMode.CHEAPEST_FIRST,verbose=True)
# nnTeam.fit()
# nnTeam.saveModel(nnFilename)
# nnTeam.updatePredictionData(config.CURRENT_SEASON, config.CURRENT_SEASON, config.CURRENT_GAMEWEEK, config.CURRENT_GAMEWEEK+1)
# all_teams.append(nnTeam)

# nnTeam = NNTeamPredicter(SolverMode.CHEAPEST_FIRST,verbose=True)
# nnTeam.loadModel(nnFilename)
# nnTeam.updatePredictionData(config.CURRENT_SEASON, config.CURRENT_SEASON, config.CURRENT_GAMEWEEK, config.CURRENT_GAMEWEEK+1)
# all_teams.append(nnTeam)

resultList: list[dict] = []

# TODO: Optimise team evaluating
for (i, team) in enumerate(all_teams):
     # TODO: Somehow combine team.train() and team.find_team() into one method
     # TODO: Make HTML outputting more expandable
     #evaluator = TeamEvaluator()
     #accuracy: float = evaluator.evaluate(team)
     print(team.label)
     team.train()
     team.find_team()

     #team.setAccuracy(accuracy)
     team.save_summary(summary_filename,date=CURRENT_DATE, pIndex=i)
     resultList.append(team.toDict())
    

with open(json_filename, "w+") as f:
    json.dump(resultList, f, indent=4)

endTime = time.perf_counter()
elapsedTime = endTime - startTime

print(f"Completed in {elapsedTime} seconds")