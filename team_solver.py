import pandas as pd
from enum import Enum
import json

NUM_GOALKEEPERS = 1
NUM_DEFENDERS = 4
NUM_FORWARD = 2
NUM_MID = 4

MAX_BUDGET = 1000

instance_count = 0

class SolverMode(Enum):
    CHEAPEST_FIRST = 1
    HIGHEST_COST_FIRST = 2
    def __str__(self):
        if(self.value == 1):
            return "cheapest first"
        elif (self.value == 2):
            return "most expensive first"

class TeamSolver():
    def __init__(self,heuristic: str, max_iters: int, mode: SolverMode,log: bool=True, use_form: bool=True):
        data = pd.read_csv("./data/data.csv")
        
        # Use the pre-existing "first_name" column, rather than creating a new one, in order to preserve column order
        data["first_name"] = data["first_name"] + " " + data["last_name"]
        data.drop(columns=["last_name"])
        data = data.rename({"first_name": "name"})
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
        global instance_count
        instance_count += 1
        self.id = instance_count

    def get_bench(self):
        '''
        Get average players to put on bench, in order to allow more budget to be spent on players who are actually playing
        '''
        players = pd.DataFrame(columns=self.default_goalkeepers.columns) # Arbitrarily use the goalkeepers' columns as reference

        goalkeeper_index = len(self.default_goalkeepers) // 2
        goalkeeper = self.default_goalkeepers.iloc[goalkeeper_index]

        defender_index = len(self.default_defenders) // 2
        defenders = self.default_defenders.iloc[defender_index]

        forward_index = len(self.default_forwards) // 2
        forward = self.default_mid.iloc[forward_index]

        mid_index = len(self.default_mid) // 2
        mid = self.default_mid.iloc[mid_index]

        pd.concat([])

        pass
    
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

    def team_to_str(self) -> str:
        txt = "\n"
        txt += f"Cost: {self.total_cost}\n"
        txt += f"Score: {self.total_score}\n"
        final_team = self.concat_team()
        txt += final_team.to_string() + "\n"
        captain = self.get_captain_name(final_team)
        txt += f"Suggested captain: {captain}\n"
        vice_captain = self.get_vice_captain_name(final_team)
        txt += f"Suggested vice captain: {vice_captain}"
        return txt
    
    def prettyify_str(self,txt: str):
        if txt.strip() == "": return ""
        new_str = txt.replace("_"," ")
        new_str = txt.title().replace("_"," ")
        return new_str

    def __str__(self) -> str:
        txt = f"Team Solver {self.id} - {self.prettyify_str(self.score_heuristic)} with mode {self.mode}"
        return txt

    def save_summary(self,filename, mode: str = "a+"):
        with open(filename,mode,encoding="utf-8") as f:
            f.writelines([str(self),self.team_to_str(),"\n\n"])

    def to_json(self,filename: str) -> None:
        with open(filename,"r") as f:
            json_data = json.load(f)
        team = self.concat_team()
 
        team_json = team.to_dict(orient="records")
        json_data["data"].append(team_json)
        json_str = json.dumps(json_data,indent=4)
        with open(filename,"w+") as f:
            f.write(json_str)

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
            return
        
        if(self.log):
            self.print_summary()
        self.adjust_team(0)
        self.iter += 1

        if self.iter > self.max_iters:
            print("Failed to find optimal team.")
        else:
            return self.find_team()