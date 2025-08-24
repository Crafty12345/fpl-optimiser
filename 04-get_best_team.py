import pandas as pd
import os
import time
from modules.team_solver import TeamSolver,SolverMode
from modules.team_predicter import TeamPredicter
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

all_teams: list[TeamPredicter] = []

team_solver = TeamPredicter("combined",SolverMode.CHEAPEST_FIRST,verbose=True)
all_teams.append(team_solver)

team_solver = TeamPredicter("combined",SolverMode.HIGHEST_COST_FIRST,verbose=True)
all_teams.append(team_solver)

for team in all_teams:
     team.find_team()
     team.save_summary(summary_filename,date=CURRENT_DATE)
     team.to_json(json_filename)
    
all_teams[0].saveCalculations(calculationsFilename)

endTime = time.perf_counter()
elapsedTime = endTime - startTime

print(f"Completed in {elapsedTime} seconds")