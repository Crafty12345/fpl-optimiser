import pandas as pd
import os
import time
import json

from modules.team_solver import TeamSolver,SolverMode
from modules.team_predicter import LinearTeamPredicter
from modules.forest_team_predicter import RFTeamPredicter
from modules.team_evaluator import TeamEvaluator
import config
# TODO: Output into markdown document


CURRENT_DATE = config.CURRENT_DATE
startTime = time.perf_counter()
resultsDirectory = f"./results/{CURRENT_DATE}"
calculationsFilename = f"{resultsDirectory}/scores_{CURRENT_DATE}.json"
summary_filename = f"{resultsDirectory}/summary_{CURRENT_DATE}.html"

if(not os.path.exists(resultsDirectory)):
    os.makedirs(resultsDirectory)

json_filename = f"{resultsDirectory}/results_{CURRENT_DATE}.json"
with open(json_filename,"w+",encoding="utf-8") as f:
    f.write("{\"data\": []}")

all_teams: list[TeamSolver] = []

#team_solver = LinearTeamPredicter("combined",SolverMode.CHEAPEST_FIRST,verbose=True)
#all_teams.append(team_solver)

# team_solver = LinearTeamPredicter("combined",SolverMode.HIGHEST_COST_FIRST,verbose=True)
# all_teams.append(team_solver)

rfTeam = RFTeamPredicter("score",SolverMode.CHEAPEST_FIRST,verbose=True,pLabel="Random Forest")
all_teams.append(rfTeam)

# rfTeam = RFTeamPredicter("combined",SolverMode.HIGHEST_COST_FIRST,verbose=True)
# all_teams.append(rfTeam)

resultList: list[dict] = []

for (i, team) in enumerate(all_teams):
     # TODO: Somehow combine team.train() and team.find_team() into one method
     # TODO: Make HTML outputting more expandable
     #evaluator = TeamEvaluator()
     #accuracy: float = evaluator.evaluate(team)
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