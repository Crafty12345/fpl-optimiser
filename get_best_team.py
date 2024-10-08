import pandas as pd
import os
from team_solver import TeamSolver,SolverMode
# TODO: Output into markdown document

MAX_ITERS = 300
CURRENT_DATE = "05-09-2024"
summary_filename = f"./results/{CURRENT_DATE}/summary_{CURRENT_DATE}.html"

if(not os.path.exists(f"./results/{CURRENT_DATE}")):
    os.makedirs(f"./results/{CURRENT_DATE}")

json_filename = f"./results/{CURRENT_DATE}/results_{CURRENT_DATE}.json"
with open(json_filename,"w+",encoding="utf-8") as f:
    f.write("{\"data\": []}")

all_teams: list[TeamSolver] = []

team_solver = TeamSolver("ict_index",MAX_ITERS,SolverMode.CHEAPEST_FIRST,log=False)
all_teams.append(team_solver)

team_solver = TeamSolver("ict_index",MAX_ITERS,SolverMode.HIGHEST_COST_FIRST,log=False)
all_teams.append(team_solver)

team_solver = TeamSolver("total_points",MAX_ITERS,SolverMode.CHEAPEST_FIRST,log=False)
all_teams.append(team_solver)

team_solver = TeamSolver("total_points",MAX_ITERS,SolverMode.HIGHEST_COST_FIRST,log=False)
all_teams.append(team_solver)

team_solver = TeamSolver("points_per_game",MAX_ITERS,SolverMode.CHEAPEST_FIRST,log=False)
all_teams.append(team_solver)

team_solver = TeamSolver("points_per_game",MAX_ITERS,SolverMode.HIGHEST_COST_FIRST,log=False)
all_teams.append(team_solver)

for team in all_teams:
    team.find_team()
    team.save_summary(summary_filename,date=CURRENT_DATE)
    team.to_json(json_filename)