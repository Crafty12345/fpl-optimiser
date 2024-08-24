import pandas as pd
from team_solver import TeamSolver,SolverMode

MAX_ITERS = 300

team_solver_1 = TeamSolver("ict_index",MAX_ITERS,SolverMode.CHEAPEST_FIRST,log=False)
print("\nTeam Solver 1 - ICT Index with cheapest first:")
team_solver_1.find_team()

print("\nTeam Solver 2 - ICT Index with most expensive first:")
team_solver_2 = TeamSolver("ict_index",MAX_ITERS,SolverMode.HIGHEST_COST_FIRST,log=False)
team_solver_2.find_team()

team_solver_1 = TeamSolver("total_points",MAX_ITERS,SolverMode.CHEAPEST_FIRST,log=False)
print("\nTeam Solver 3 - Points with cheapest first:")
team_solver_1.find_team()

print("\nTeam Solver 4 - Points with most expensive first:")
team_solver_2 = TeamSolver("total_points",MAX_ITERS,SolverMode.HIGHEST_COST_FIRST,log=False)
team_solver_2.find_team()