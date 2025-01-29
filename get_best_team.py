import pandas as pd
import os
import time
from modules.team_solver import TeamSolver,SolverMode
from modules.team_predicter import TeamPredicter
import config
# TODO: Output into markdown document


CURRENT_DATE = config.CURRENT_DATE
startTime = time.perf_counter()
resultsDirectory = f"./data/results/{CURRENT_DATE}"
summary_filename = f"{resultsDirectory}/summary_{CURRENT_DATE}.html"

if(not os.path.exists(resultsDirectory)):
    os.makedirs(resultsDirectory)

json_filename = f"{resultsDirectory}/results_{CURRENT_DATE}.json"
with open(json_filename,"w+",encoding="utf-8") as f:
    f.write("{\"data\": []}")

all_teams: list[TeamPredicter] = []

# Legacy code:

#team_solver = TeamPredicter("ict_index",SolverMode.CHEAPEST_FIRST,log=False)
#all_teams.append(team_solver)
#
#team_solver = TeamPredicter("ict_index",SolverMode.HIGHEST_COST_FIRST,log=False)
#all_teams.append(team_solver)
#
#team_solver = TeamPredicter("total_points",SolverMode.CHEAPEST_FIRST,log=False)
#all_teams.append(team_solver)
#
#team_solver = TeamPredicter("total_points",SolverMode.HIGHEST_COST_FIRST,log=False)
#all_teams.append(team_solver)
#
#team_solver = TeamPredicter("points_per_game",SolverMode.CHEAPEST_FIRST,log=False)
#all_teams.append(team_solver)
#
#team_solver = TeamPredicter("points_per_game",SolverMode.HIGHEST_COST_FIRST,log=False)
#all_teams.append(team_solver)
#
team_solver = TeamPredicter("combined",SolverMode.CHEAPEST_FIRST,verbose=True)
all_teams.append(team_solver)

team_solver = TeamPredicter("combined",SolverMode.HIGHEST_COST_FIRST,verbose=True)
all_teams.append(team_solver)

for team in all_teams:
     team.find_team()
     team.save_summary(summary_filename,date=CURRENT_DATE)
     team.to_json(json_filename)
    
endTime = time.perf_counter()
elapsedTime = endTime - startTime

print(f"Completed in {elapsedTime} seconds")