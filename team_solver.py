import pandas as pd
from enum import Enum

NUM_GOALKEEPERS = 2
NUM_DEFENDERS = 5
NUM_FORWARD = 3
NUM_MID = 5

BUDGET = 1000

class SolverMode(Enum):
    CHEAPEST_FIRST = 1
    HIGHEST_COST_FIRST = 2
    pass
class TeamSolver():
    def __init__(self,heuristic: str, max_iters: int, mode: SolverMode,log: bool=True, use_form: bool=True):
        data = pd.read_csv("./data/data.csv")
        self.score_heuristic = heuristic

        if(use_form):
            data["score"] = data[self.score_heuristic] * data["form"]
        else:
            data["score"] = data[self.score_heuristic]

        goalkeepers = data.loc[data["position"]=="GKP"]
        defenders = data.loc[data["position"]=="DEF"]
        forward = data.loc[data["position"]=="FWD"]
        mid = data.loc[data["position"]=="MID"]

        self.max_iters = max_iters
        self.mode = mode
        self.log = log


        self.default_goalkeepers = goalkeepers.sort_values(by="score",ascending=False)
        self.default_defenders = defenders.sort_values(by="score",ascending=False)
        self.default_forwards = forward.sort_values(by="score",ascending=False)
        self.default_mid = mid.sort_values(by="score",ascending=False)

        self.goalkeepers = self.default_goalkeepers[0:NUM_GOALKEEPERS]
        self.defenders = self.default_defenders[0:NUM_DEFENDERS]
        self.forwards = self.default_forwards[0:NUM_FORWARD]
        self.mid = self.default_mid[0:NUM_MID]
        self.total_cost = self.get_cost()
        self.total_score = self.calculate_score()
        self.profit = BUDGET-self.total_cost
        self.iter = 0
    
    def get_cost(self):
        return self.goalkeepers["cost"].sum() + \
            self.defenders["cost"].sum() + \
            self.forwards["cost"].sum() + \
            self.mid["cost"].sum()
    
    def calculate_score(self):
        return self.goalkeepers["score"].sum() + \
            self.defenders["score"].sum() + \
            self.forwards["score"].sum() + \
            self.mid["score"].sum()
    
    def concat_team(self):
        return pd.concat([self.goalkeepers,self.defenders,self.forwards,self.mid])

    def get_captain_name(self,team: pd.DataFrame):
        team = team.sort_values(by="score",ascending=False)
        return team.iloc[0]["name"]
    def get_vice_captain_name(self,team: pd.DataFrame):
        team = team.sort_values(by="score",ascending=False)
        return team.iloc[1]["name"]

    def print_team(self):
        print("Final team:")
        print("Cost:",self.total_cost)
        print("Total score:",self.total_score)
        final_team = self.concat_team()
        print(final_team)
        captain = self.get_captain_name(final_team)
        print("Suggested captain:",captain)
        vice_captain = self.get_vice_captain_name(final_team)
        print("Suggested vice captain:",vice_captain)
        print("\n")

    def adjust_forward(self,cost,id) -> bool:
        old_forwards = self.forwards
        self.forwards = self.forwards[self.forwards["id"]!=id]
        new_forward = self.default_forwards[(self.default_forwards["cost"] < cost)]
        new_forward = new_forward[~(new_forward["id"].isin(self.forwards["id"]))].head(1)
        if(len(new_forward) == 0):
            self.forwards = old_forwards
            return False
        self.forwards = pd.concat([self.forwards,new_forward])
        assert len(self.forwards) == NUM_FORWARD
        return True

    def adjust_goalie(self,cost,id) -> bool:
        old_goalkeepers = self.goalkeepers
        self.goalkeepers = self.goalkeepers[self.goalkeepers["id"]!=id]
        new_goalkeeper = self.default_goalkeepers[(self.default_goalkeepers["cost"] < cost)]
        new_goalkeeper = new_goalkeeper[~(new_goalkeeper["id"].isin(self.goalkeepers["id"]))].head(1)
        if(len(new_goalkeeper) == 0):
            self.goalkeepers = old_goalkeepers
            return False
        self.goalkeepers = pd.concat([self.goalkeepers,new_goalkeeper])
        assert len(self.goalkeepers)==NUM_GOALKEEPERS
        return True

    def adjust_defender(self,cost,id) -> bool:
        old_defenders = self.defenders
        self.defenders = self.defenders[self.defenders["id"]!=id]
        new_defender = self.default_defenders[(self.default_defenders["cost"] < cost)]
        new_defender = new_defender[~(new_defender["id"].isin(self.defenders["id"]))].head(1)
        if(len(new_defender) == 0):
            self.defenders = old_defenders
            return False
        self.defenders = pd.concat([self.defenders,new_defender])
        assert len(self.defenders) == NUM_DEFENDERS
        return True

    def adjust_mid(self,cost,id) -> bool:
        old_mid = self.mid
        self.mid = self.mid[self.mid["id"]!=id]
        new_mid = self.default_mid[(self.default_mid["cost"] < cost)]
        new_mid = new_mid[~(new_mid["id"].isin(self.mid["id"]))].head(1)
        if(len(new_mid) == 0):
            self.mid = old_mid
            return False
        self.mid = pd.concat([self.mid,new_mid])
        assert len(self.mid) == NUM_MID
        return True

    def print_summary(self):
        print("Iteration:",str(self.iter)+"\tCost:",str(self.total_cost) + "\tTotal Score:",str(self.total_score) + "\tProfit:",str(self.profit))

    def get_nth_most_expensive(self,team: pd.DataFrame, n: int):
        assert (n < len(team)) and (n >= 0)
        team = team.sort_values(by="cost",ascending=False)
        nth_most_expensive = team.iloc[n]
        return nth_most_expensive
    
    def get_nth_cheapest(self,team: pd.DataFrame, n: int):
        assert (n < len(team)) and (n >= 0)
        team = team.sort_values(by="cost",ascending=True)
        nth_cheapest = team.iloc[n]
        return nth_cheapest

    def adjust_team(self,n: int, num_recursions: int = 0):
        team = self.concat_team()
        n = min(n,len(team)-1)
        n = max(0,n)
        if(self.mode == SolverMode.CHEAPEST_FIRST):
            selected = self.get_nth_cheapest(team,n)
        elif(self.mode==SolverMode.HIGHEST_COST_FIRST):
            selected = self.get_nth_most_expensive(team,n)
        selected_position = selected["position"]
        selected_cost = selected["cost"]
        selected_id = selected["id"]

        match selected_position:
            case "FWD":
                can_adjust = self.adjust_forward(selected_cost,selected_id)
            case "DEF":
                can_adjust = self.adjust_defender(selected_cost,selected_id)
            case "GKP":
                can_adjust = self.adjust_goalie(selected_cost,selected_id)
            case "MID":
                can_adjust = self.adjust_mid(selected_cost,selected_id)
        if not can_adjust:
            if(num_recursions > len(team)):
                return
            return self.adjust_team(n+1,num_recursions+1)

    def find_team(self):
        self.total_cost = self.get_cost()
        self.total_score = self.calculate_score()
        self.profit = self.total_cost - BUDGET

        if self.total_cost <= BUDGET:
            if(self.log):
                print("Successfully found team!")
            self.print_team()
            return
        
        if(self.log):
            self.print_summary()
        self.adjust_team(0)
        self.iter += 1

        if self.iter > self.max_iters:
            print("Failed to find optimal team.")
            self.print_team()
        else:
            return self.find_team()