import pandas as pd
from enum import Enum
import numpy as np
import json

import config

pd.options.mode.copy_on_write = True

NUM_GOALKEEPERS = 1
NUM_DEFENDERS = 4
NUM_FORWARD = 2
NUM_MID = 4

MAX_BUDGET = 1000
MAX_AMT_REMAINING = 5

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
	def __init__(self, pHeuristic: str, pMode: SolverMode, pDataFileName: str, verbose: bool=False):
		self.data = pd.read_csv(pDataFileName)

		if(pHeuristic == "combined"):
			self.data["combined"] = self.calculateCombinedScore(self.data)
		self.score_heuristic = pHeuristic

		self.data["score"] = self.data[self.score_heuristic] * self.data["form"] * self.data["starts_per_90"]

		self.max_iters = config.MAX_ITERS
		self.mode = pMode
		self.log = verbose

		self.registerInstance()
		self.start()

	def registerInstance(self):
		global instance_count
		instance_count += 1
		self.id = instance_count
		pass

	def calcPScores(self, pSeries: pd.Series) -> pd.Series:
		stdDev = np.std(pSeries)
		avg = pSeries.mean()
		return (pSeries - avg) / stdDev
	
	def start(self):
		self.default_players = dict()

		goalkeepers = self.data.loc[self.data["position"]=="GKP"]
		defenders = self.data.loc[self.data["position"]=="DEF"]
		forward = self.data.loc[self.data["position"]=="FWD"]
		mid = self.data.loc[self.data["position"]=="MID"]

		self.default_players["GKP"] = goalkeepers.sort_values(by="score",ascending=False)
		self.default_players["DEF"] = defenders.sort_values(by="score",ascending=False)
		self.default_players["FWD"] = forward.sort_values(by="score",ascending=False)
		self.default_players["MID"] = mid.sort_values(by="score",ascending=False)

		self.budget = MAX_BUDGET
		self.get_bench()

		self.players: dict[str,pd.DataFrame] = dict()
		default_not_bench = self.default_players.copy()
		for position in default_not_bench.keys():
			temp_players = default_not_bench[position]
			default_not_bench[position] = temp_players.loc[~temp_players["id"].isin(self.bench["id"])]
		
		self.players["GKP"] = default_not_bench["GKP"][0:NUM_GOALKEEPERS]
		self.players["DEF"] = default_not_bench["DEF"][0:NUM_DEFENDERS]
		self.players["FWD"] = default_not_bench["FWD"][0:NUM_FORWARD]
		self.players["MID"] = default_not_bench["MID"][0:NUM_MID]
		self.validate_team()

		self.total_cost = self.sum_stat("cost")
		self.total_score = self.sum_stat("score")
		self.profit = self.budget-self.total_cost
		self.iter = 0

		self.update_stats()

	def get_bench_player(self, position: str) -> pd.Series:
		'''
		Get the best bench player, given a position
		'''
		valid_options = {"GKP", "DEF", "FWD", "MID"}
		assert (position in valid_options), f"Position {position} is an invalid position. Valid options: {valid_options}"

		players_value: pd.DataFrame = self.default_players[position]
		players_value.loc[:,"value"] = players_value["score"] / players_value["cost"]
		players_value = players_value.sort_values(by=["value"],ascending=False)
		threshold = 0.5
		temp_players = players_value.loc[players_value["value"] >= threshold]
		while len(temp_players) == 0:
			threshold -= 0.05
			temp_players = players_value.loc[players_value["value"] >= threshold]
		#print(temp_players)
		#print(temp_players["cost"].sort_values(ascending=True))
		minCost = temp_players["cost"].min()
		minCostPlayers = temp_players.loc[temp_players["cost"]==minCost]
		maxScorePlayerIndex = minCostPlayers["score"].argmax()
		maxScorePlayer = minCostPlayers.iloc[[maxScorePlayerIndex]]
		return maxScorePlayer

	def get_bench(self):
		'''
		Get average players to put on bench, in order to allow more budget to be spent on players who are actually playing
		'''
		forward = self.get_bench_player("FWD")
		mid = self.get_bench_player("MID")
		defender = self.get_bench_player("DEF")
		goalkeeper = self.get_bench_player("GKP")

		players = pd.concat([forward,mid,defender,goalkeeper])
		self.bench_cost = players["cost"].sum()
		self.budget = MAX_BUDGET - self.bench_cost
		self.bench: pd.DataFrame = players
		pass
	
	def sum_stat(self, column: str):
		'''
		This method sums a specified column in the `self.players` dictionary
		'''
		_sum = 0
		for position in self.players.values():
			_sum += position[column].sum()
		return _sum
	
	def calculateCombinedScore(self, pData: pd.DataFrame) -> pd.Series:
		"""
		Calculates combined score using the provided data
		"""
		ictIndexPScores = self.calcPScores(pData["ict_index"])
		totalPointsPScores = self.calcPScores(pData["total_points"])
		pointsPerGamePScores = self.calcPScores(pData["points_per_game"])
		return ictIndexPScores + totalPointsPScores + pointsPerGamePScores
	
	def sum_bench_column(self,column: str) -> float:
		return self.bench[column].sum()
	
	def concat_team(self):
		return pd.concat(position for position in self.players.values())

	def get_captain_name(self,team: pd.DataFrame):
		team = team.sort_values(by="score",ascending=False)
		return team.iloc[0]["name"]
	def get_vice_captain_name(self,team: pd.DataFrame):
		team = team.sort_values(by="score",ascending=False)
		return team.iloc[1]["name"]

	def team_to_html(self) -> str:
		final_team = self.concat_team()
		final_team_html = final_team.to_html()
		bench_score = self.sum_bench_column("score")
		total_total_cost = self.total_cost + self.bench_cost
		total_total_score = self.total_score + bench_score

		txt = f"""
		<p>Cost:{self.total_cost}</p>
		<p>Score:{self.total_score}</p>
		{final_team_html}
		<p>Suggested captain: {self.get_captain_name(final_team)}</p>
		<p>Suggested vice captain: {self.get_vice_captain_name(final_team)}</p>
		<h2>Bench</h2>
		{self.bench.to_html()}
		<p>Bench cost: {self.bench_cost}</p>
		<p>Bench score: {bench_score}</p>
		<h2>Summary</h2>
		Total cost: {total_total_cost}
		Total score: {total_total_score}
		"""
		return txt

	def team_to_str(self) -> str:
		txt = "\n"
		txt += f"Cost: {self.total_cost}\n"
		txt += f"Score: {self.total_score}\n"
		final_team = self.concat_team()
		txt += final_team.to_string() + "\n"
		captain = self.get_captain_name(final_team)
		txt += f"Suggested captain: {captain}\n"
		vice_captain = self.get_vice_captain_name(final_team)
		txt += f"Suggested vice captain: {vice_captain}\n\n"
		txt += "Bench:\n"
		txt += self.bench.to_string()
		bench_cost = self.bench_cost
		txt += f"\nBench Cost: {bench_cost}\n"
		bench_score = self.sum_bench_column("score")
		txt += f"Bench Score: {bench_score}\n"

		total_total_cost = self.total_cost + bench_cost
		total_total_score = self.total_score + bench_score
		txt += f"\nTotal cost: {total_total_cost}"
		txt += f"\nTotal score: {total_total_score}\n"
		return txt
	
	def prettyify_str(self,txt: str):
		if txt.strip() == "": return ""
		new_str = txt.replace("_"," ")
		new_str = txt.title().replace("_"," ")
		return new_str

	def __str__(self) -> str:
		txt = f"Team Solver {self.id} - {self.prettyify_str(self.score_heuristic)} with mode {self.mode}"
		return txt
	
	def validate_current_html(self,contents):
		'''
		This method validates HTML, by checking if certain strings are contained within the HTML
		'''
		txts = ["<html>","<head>","<meta charset='UTF-8'>", "Team Solver", "</head>", "<body>"]
		for txt in txts:
			if txt not in contents:
				return False
		return True
	
	def save_html(self, filename: str, date: str, mode: str = "a+"):
		if date == "":
			raise ValueError("Date is not provided")
		try:
			with open(filename,"r",encoding="utf-8") as f:
				current_contents = f.read()
				should_append = self.validate_current_html(current_contents)
				pass
		# Write instead of appending, if the file does not yet exist
		except FileNotFoundError as err:
			should_append = False
			
		added_content = \
f"""
		<h1> {str(self)} </h1>
		{self.team_to_html()}
		<hr>
"""

		# If starting the file from scratch
		if not should_append:
			content = \
f"""
<html>
	<head>
		<meta charset='UTF-8'>
		<title>{date} Team Solver</title>
	</head>
	<body>
"""
			content += added_content
			with open(filename,"w+",encoding="utf-8") as f:
				f.write(content)
		else:
			with open(filename,mode,encoding="utf-8") as f:
				f.write(added_content)
		pass

	def save_txt(self, filename, mode: str = "a+"):
		with open(filename,mode,encoding="utf-8") as f:
			f.writelines([str(self),self.team_to_str(),"\n\n"])
		pass

	def save_summary(self,filename, mode: str = "a+", date: str = ""):
		if("." not in filename):
			raise f"Error: file extension not found in '{filename}'"
		filename_split = filename.split(".")
		file_extension = filename_split[-1]
		match file_extension:
			case "html":
				self.save_html(filename,date,mode)
			case "txt":
				self.save_txt(filename,mode)
			case _:
				raise NotImplementedError(f"File extension '{file_extension}' has not yet been implemented")

	def to_json(self,filename: str) -> None:
		with open(filename,"r") as f:
			json_data = json.load(f)
		
		team = self.concat_team()
		bench_temp = self.bench
		bench_temp["is_benched"] = True
		team["is_benched"] = False
		team = pd.concat([team,bench_temp])
		#print(team["status"])
		team = team.sort_values(by=["position","is_benched","score"])
		team = team.drop(columns=["value"])
 
		team_json = team.to_dict(orient="records")
		json_data["data"].append(team_json)
		json_str = json.dumps(json_data,indent=4)
		with open(filename,"w+") as f:
			f.write(json_str)

	def saveCalculations(self, filename: str) -> None:
		self.data.to_json(filename, orient="records")

	def check_players(self,position):
		match position:
			case "forward":
				assert len(self.players[position]) == NUM_FORWARD
			case "goalkeeper":
				assert len(self.players[position]) == NUM_GOALKEEPERS
			case "defender":
				assert len(self.players[position]) == NUM_DEFENDERS
			case "mid":
				assert len(self.players[position]) == NUM_MID

	def adjust_players(self,cost,id, position: str, current_score: float, score_threshold: float = 1.0) -> bool:
		PLAY_PERCENT_THRESHOLD = 0.99
		old_players = self.players[position]
		self.players[position] = self.players[position][self.players[position]["id"]!=id]
		new_players = self.default_players[position][(self.default_players[position]["cost"] < cost)]
		new_players = new_players.loc[(new_players["form"] >= 1.0) & (new_players["starts_per_90"] >= PLAY_PERCENT_THRESHOLD)]
		new_players = new_players[~(new_players["id"].isin(self.players[position]["id"])) & ~(new_players["id"].isin(self.bench["id"]))]
		new_players = new_players.sort_values(by="score",ascending=False)
		#print(new_players)
		#print("test")
		new_players = new_players.head(1)
		#assert new_players["id"].values[0] != 315.0
		if(len(new_players) == 0):
			self.players[position] = old_players
			return False
		self.players[position] = pd.concat([self.players[position],new_players])
		self.check_players(position)
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
		##print("nth_cheapest",nth_cheapest)
		##print(team)
		# If a player is cheaper, but has a lower score remove them
		return nth_cheapest

	def adjust_team(self,n: int, num_recursions: int = 0):
		team = self.concat_team()
		#print(team)
		n = min(n,len(team)-1)
		n = max(0,n)
		if(self.mode == SolverMode.CHEAPEST_FIRST):
			selected = self.get_nth_cheapest(team,n)
		elif(self.mode==SolverMode.HIGHEST_COST_FIRST):
			selected = self.get_nth_most_expensive(team,n)
		selected_position = selected["position"]
		selected_cost = selected["cost"]
		selected_id = selected["id"]

		can_adjust = self.adjust_players(selected_cost,selected_id,selected_position,selected["score"])
		if not can_adjust:
			if(num_recursions > len(team)):
				return
			return self.adjust_team(n+1,num_recursions+1)
		
	def get_worst_player(self, team_concat: pd.DataFrame):
		worst_player_index = team_concat["score"].argmin()
		worst_player = team_concat.iloc[worst_player_index]
		return worst_player

	def replace_player(self, df: pd.DataFrame, old_player_id: int, new_player: pd.DataFrame | pd.Series):
		
		df = df.loc[df["id"] != old_player_id]
		df = pd.concat([df,new_player])
		return df

	def try_replace(self,team: pd.DataFrame):
		worst_player = self.get_worst_player(team)
		worst_player_position = worst_player["position"]
		options = self.default_players[worst_player_position]
		options = options.loc[~(options["id"].isin(team["id"])) & ~(options["id"].isin(self.bench["id"]))]
		actual_options = options.loc[options["score"] > worst_player["score"]]
		amount_remaining = MAX_BUDGET - (self.bench_cost + self.total_cost)
		if len(actual_options) > 0:
			max_price = worst_player["cost"] + amount_remaining
			actual_options = actual_options.loc[actual_options["cost"] <= max_price]
			if(len(actual_options) > 0):
				best_option_index = actual_options["score"].argmax()
				best_option = actual_options.iloc[[best_option_index]]
				new_team_df = self.replace_player(team,worst_player["id"],best_option)
				return new_team_df
		return None
		#print(self.is_best_player(worst_player,options))

	def backward_adjust(self):
		positions = ["GKP", "MID", "FWD", "DEF"]
		for position in positions:
			new_team = self.try_replace(self.players[position])
			if(new_team is not None):
				self.players[position] = new_team
				self.update_stats()
		amount_remaining = MAX_BUDGET - (self.bench_cost + self.total_cost)
		# if amount_remaining > 0:
		#     new_bench = self.try_replace(self.bench)
		#     if(new_bench is not None):
		#         self.bench = new_bench
		#         self.validate_team()
		#         self.bench_cost = self.sum_bench_column("cost")
		
	def swap_bench(self, old_bench_player: pd.DataFrame, new_bench_player: pd.DataFrame, position: str):
		'''
		This method puts `old_bench_player` in the main team and `new_bench_player` in the bench team
		'''
		# Remove old bench player from bench
		old_bench_player_id = old_bench_player["id"].values[0]
		new_bench_player_id = new_bench_player["id"].values[0]

		self.bench = self.bench.loc[self.bench["id"]!=old_bench_player_id]
		# Add new bench player to bench
		self.bench = pd.concat([self.bench,new_bench_player])
		
		players = self.players[position]
		# Remove player who is currently unbenched from unbenched team
		players = players.loc[players["id"]!=new_bench_player_id]
		# Add player who used to be benched, to new team
		players = pd.concat([players,old_bench_player])
		self.players[position] = players
		pass

	def manage_bench(self,position: str):
		'''
		This method ensures that only worst players are placed on the bench
		'''
		current_bench_player = self.bench.loc[self.bench["position"]==position]
		bench_score = current_bench_player["score"].values[0]
		
		players = self.players[position]
		worst_score_index = players["score"].argmin()
		if(self.players[position]["score"].iloc[worst_score_index] < bench_score):
			new_bench_player = self.players[position].iloc[[worst_score_index]]
			self.swap_bench(current_bench_player,new_bench_player,position)
		self.bench_cost = self.sum_bench_column("cost")

	def update_stats(self):
		self.total_cost = self.sum_stat("cost")
		self.total_score = self.sum_stat("score")
		self.profit = self.budget - self.total_cost

	def validate_team(self):
		ids = list(self.players.values())
		for player in ids:
			_id = player["id"].values[0]
			for id2 in self.bench["id"]:
				assert _id != id2, f"Detected duplicate player with id: {_id}"
		pass

	def removeOutliers(self):
		positions = {"DEF", "FWD", "MID", "GKP"}
		pScoreCutoff = 0
		for position in positions:
			scoresOfPosition = self.data.loc[self.data["position"]==position]["score"]
			scorePScores = self.calcPScores(scoresOfPosition)
			mask = scorePScores > pScoreCutoff
			# If outlier, set score to something exceedingly low, so that the player will not be chosen
			# Due to the score being required by other files (such as 05-get_best_transfer.ipynb), entirely removing outliers does not work.
			self.data.loc[self.data["position"] == position].loc[mask] = -99
		self.data = self.data.dropna()
		pass

	def queryPlayerScore(self, pPlayerName: str):
		foundPlayer = self.data.loc[self.data["name"] == pPlayerName]
		return foundPlayer["score"].values[0]
	def queryPlayerScores(self, pPlayers: pd.DataFrame):
		players = self.data.loc[self.data["id"] == pPlayers["id"]]
		return players["score"]

	def find_team(self):
		self.update_stats()
		amount_remaining = MAX_BUDGET - (self.total_cost + self.bench_cost)
		MAX_ITERS = 50
		if (self.total_cost <= self.budget):
			iter = 0
			while (amount_remaining > MAX_AMT_REMAINING) and (amount_remaining >= 0) and (iter < MAX_ITERS):
				if(self.log):
					print(f"Iter: {iter}")
				self.backward_adjust()
				self.update_stats()
				amount_remaining = MAX_BUDGET - (self.total_cost + self.bench_cost)
				iter += 1
			self.manage_bench("GKP")
			self.manage_bench("FWD")
			self.manage_bench("MID")
			self.manage_bench("DEF")

			self.update_stats()
			self.bench_cost = self.sum_bench_column("cost")

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