import pandas as pd
from enum import Enum
import numpy as np
import json
from abc import ABC, abstractmethod
from copy import deepcopy

import config

pd.options.mode.copy_on_write = True

NUM_GOALKEEPERS = 1
# TODO: Implement auto-tuning for number of defenders/forwards/mid
NUM_DEFENDERS = 4
NUM_FORWARD = 2
NUM_MID = 4
assert NUM_GOALKEEPERS + NUM_DEFENDERS + NUM_FORWARD + NUM_MID == 11

MAX_BUDGET = 995
MAX_AMT_REMAINING = 2

instance_count = 0

class SolverMode(Enum):
	CHEAPEST_FIRST = 1
	HIGHEST_COST_FIRST = 2
	def __str__(self):
		if(self.value == 1):
			return "cheapest first"
		elif (self.value == 2):
			return "most expensive first"

# TODO : Continue refactoring; maybe change self.id to self.label: str?
class TeamSolver(ABC):
	@abstractmethod
	def precalcScores(self, pData: pd.DataFrame, pGameweek: int, pSeason: int): raise NotImplementedError()
	@abstractmethod
	def updatePredictionData(self, pRefSeason: int, pTargetSeason: int, pRefWeek: int, pTargetWeek: int): raise NotImplementedError()
	@abstractmethod
	def fit(self): raise NotImplementedError()

	def __init__(self, pHeuristic: str, pMode: SolverMode, verbose: bool=False, pLabel: str = None, pFreeHit = False):
		self.accuracy = None
		if (pLabel is None):
			self.label = type(self).__name__
		else:
			self.label = pLabel

		self.max_iters = config.MAX_ITERS
		self.mode = pMode
		self.verbose = verbose
		self.freeHit = pFreeHit

		if(self.verbose):
			print("[DEBUG]: Reading from data file...")

		with open("./data/player_stats.json", "r") as f:
			self.dataJson: dict = json.load(f)

		self.score_heuristic = pHeuristic

		self.allData: list[pd.DataFrame] = []
		# Number of gameweeks which have been sampled
		self.sampleSize = 0
		for (season, tempDict) in self.dataJson.items():
			season = int(season)
			for (currentGameweek, playerData) in tempDict.items():
				currentGameweek = int(currentGameweek)
				currentData = pd.DataFrame.from_records(playerData)
				currentData = currentData.set_index("id", drop=False)
				currentData["gameweek"] = currentData["gameweek"].astype(np.uint16)
				currentData["season"] = currentData["season"].astype(np.uint16)
				self.precalcScores(currentData, currentGameweek, season)
				
				self.allData.append(currentData)
				self.sampleSize += 1
		self.allData = sorted(self.allData,key=lambda x: (x["season"].values[0], x["gameweek"].values[0]))
		self.latestData: pd.DataFrame = self.allData[-1].copy()

	def calcPScores(self, pSeries: pd.Series) -> pd.Series:
		stdDev = np.std(pSeries)
		avg = pSeries.mean()
		return (pSeries - avg) / stdDev
	
	def getDfByWeekAndSeason(self, pGameweek: int, pSeason: int) -> pd.DataFrame:
		result: pd.DataFrame = None
		for datum in self.allData:
			if result is None:
				if (datum["season"].values[0] == pSeason and datum["gameweek"].values[0] == pGameweek):
					result = datum
		return result
	
	# TODO: replace .loc with .at
	def getOpposingTeam(self, pTeam: str, pFixtureDf: pd.DataFrame) -> str:
		result = pFixtureDf.loc[pFixtureDf["home_team"]==pTeam]["away_team"]
		if len(result) == 0:
			result = pFixtureDf.loc[pFixtureDf["away_team"]==pTeam]["home_team"]
		if len(result) == 0:
			return "UNK"
		else:
			return result.item()
	
	def train(self):
		self.default_players = dict()
		if "id" in self.latestData.columns:
			self.latestData = self.latestData.set_index(self.latestData["id"])
			assert (self.latestData["id"].values == self.latestData.index.values).all()
		
		self.latestData.loc[((self.latestData["status"] == "i") | (self.latestData["status"] == "s")), "score"] = 0.0

		goalkeepers = self.latestData.loc[self.latestData["position"]=="GKP"]
		defenders = self.latestData.loc[self.latestData["position"]=="DEF"]
		forward = self.latestData.loc[self.latestData["position"]=="FWD"]
		mid = self.latestData.loc[self.latestData["position"]=="MID"]

		self.default_players["GKP"] = goalkeepers.sort_values(by="score",ascending=False)
		self.default_players["DEF"] = defenders.sort_values(by="score",ascending=False)
		self.default_players["FWD"] = forward.sort_values(by="score",ascending=False)
		self.default_players["MID"] = mid.sort_values(by="score",ascending=False)

		self.budget = MAX_BUDGET
		# TODO: Combine bench players into regular team DF

		self.players: dict[str,pd.DataFrame] = dict()
		

		# TODO: Refactor this to reduce code duplication
		# Add 1 for bench
		if (self.freeHit):
			self.players["GKP"] = self.default_players["GKP"].head(NUM_GOALKEEPERS)
			cheapestGkp = self.default_players["GKP"].loc[self.default_players["GKP"]["cost"] == self.default_players["GKP"]["cost"].min()].head(1)
			self.players["GKP"] = pd.concat([self.players["GKP"], cheapestGkp])
		else:
			self.players["GKP"] = self.default_players["GKP"].head(NUM_GOALKEEPERS+1)

		teamCounts = self.countTeams()
		boolArr: pd.Series[bool] = self.default_players["DEF"]["team"].apply(lambda x: teamCounts.get(x, 0) < 3)
		possibleDefs = self.default_players["DEF"].loc[boolArr]
		if (self.freeHit):
			self.players["DEF"] = possibleDefs.head(NUM_DEFENDERS)
			cheapestDef = possibleDefs.loc[possibleDefs["cost"] == possibleDefs["cost"].min()].head(1)
			self.players["DEF"] = pd.concat([self.players["DEF"], cheapestDef])
		else:
			self.players["DEF"] = possibleDefs.head(NUM_DEFENDERS+1)

		teamCounts = self.countTeams()
		boolArr: pd.Series[bool] = self.default_players["FWD"]["team"].apply(lambda x: teamCounts.get(x, 0) < 3)
		possibleFwds = self.default_players["FWD"].loc[boolArr]
		if (self.freeHit):
			self.players["FWD"] = possibleFwds.head(NUM_FORWARD)
			cheapestFwd = possibleFwds.loc[possibleFwds["cost"] == possibleFwds["cost"].min()].head(1)
			self.players["FWD"] = pd.concat([self.players["FWD"], cheapestFwd])
		else:
			self.players["FWD"] = possibleFwds.head(NUM_FORWARD+1)

		teamCounts = self.countTeams()
		boolArr: pd.Series[bool] = self.default_players["MID"]["team"].apply(lambda x: teamCounts.get(x, 0) < 3)
		possibleMids = self.default_players["MID"].loc[boolArr]
		if (self.freeHit):
			self.players["MID"] = possibleMids.head(NUM_MID)
			cheapestMid = possibleMids.loc[possibleMids["cost"] == possibleMids["cost"].min()].head(1)
			self.players["MID"] = pd.concat([self.players["MID"], cheapestMid])
		else:
			self.players["MID"] = possibleMids.head(NUM_MID+1)


		#self.validate_team()

		teamCounts = self.countTeams()
		self.total_cost = self.sum_stat("cost")
		self.total_score = self.sum_stat("score")
		self.profit = self.budget-self.total_cost
		self.iter = 0

		self.update_stats()

	def countTeams(self) -> pd.DataFrame:
		flatPlayers: pd.DataFrame = self.flattenTeam()
		teamCounts: pd.Series = flatPlayers["team"].value_counts()
		return teamCounts

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
	
	def flattenTeam(self):
		return pd.concat(position for position in self.players.values())
	
	def getTeam(self) -> pd.DataFrame:
		final_team = self.flattenTeam()
		return final_team
	def worstOfPosition(self, pPosition: str) -> pd.Series:
		return self.players[pPosition][self.score_heuristic].idxmin()

	def splitStartBench(self) -> tuple[pd.DataFrame, pd.DataFrame]:
		dfColumns: list[str] = self.players["FWD"].columns
		starting: pd.DataFrame = pd.DataFrame(columns=dfColumns)
		bench: pd.DataFrame = pd.DataFrame(columns=dfColumns)

		for position in self.players.keys():
			temp = self.players[position]
			worst = self.worstOfPosition(position)
			for (idx, player) in temp.iterrows():
				if (idx == worst):
					bench.loc[len(bench)] = player
				else:
					starting.loc[len(starting)] = player

		return (starting, bench)

	def get_captain_name(self,team: pd.DataFrame):
		team = team.sort_values(by="score",ascending=False)
		return team.iloc[0]["name"]
	def get_vice_captain_name(self,team: pd.DataFrame):
		team = team.sort_values(by="score",ascending=False)
		return team.iloc[1]["name"]

	def team_to_html(self, pIndex: int | None) -> str:
		team, bench = self.splitStartBench()
		final_team_html = team.to_html()
		bench_score = bench["score"].sum()
		benchCost = bench["cost"].sum()
		startingScore = team["score"].sum()
		startingCost = team["cost"].sum()
		total_total_cost = self.total_cost
		total_total_score = self.total_score

		txt = ""
		if (pIndex is not None):
			txt += f"<p>Index: {pIndex}</p>\n"
		txt += f"""
		<p>Cost: {self.total_cost}</p>
		<p>Score: {self.total_score}</p>
		{final_team_html}
		<p>Suggested captain: {self.get_captain_name(team)}</p>
		<p>Suggested vice captain: {self.get_vice_captain_name(team)}</p>
		<h2>Bench</h2>
		{bench.to_html()}
		<p>Bench cost: {benchCost}</p>
		<p>Bench score: {bench_score}</p>
		<h2>Summary</h2>
		Total cost: {total_total_cost}
		<br>
		Total score: {total_total_score}
		"""
		if self.accuracy is not None:
			txt += f"\n<br>Accuracy: {(self.accuracy*100):.2f}%"
		return txt

	def team_to_str(self) -> str:
		txt = "\n"
		txt += f"Cost: {self.total_cost}\n"
		txt += f"Score: {self.total_score}\n"
		team, bench = self.splitStartBench()
		txt += team.to_string() + "\n"
		captain = self.get_captain_name(team)
		txt += f"Suggested captain: {captain}\n"
		vice_captain = self.get_vice_captain_name(team)
		txt += f"Suggested vice captain: {vice_captain}\n\n"
		txt += "Bench:\n"
		txt += bench.to_string()
		bench_cost = bench["cost"].sum()
		txt += f"\nBench Cost: {bench_cost}\n"
		bench_score = bench["cost"].sum()
		txt += f"Bench Score: {bench_score}\n"

		txt += f"\nTotal cost: {self.total_cost}"
		txt += f"\nTotal score: {self.total_score}\n"
		return txt
	
	def prettyify_str(self,txt: str):
		if txt.strip() == "": return ""
		new_str = txt.replace("_"," ")
		new_str = txt.title().replace("_"," ")
		return new_str

	def __str__(self) -> str:
		txt = f"Team Solver '{self.label}' - {self.prettyify_str(self.score_heuristic)} with mode {self.mode}"
		return txt
	def getAccuracy(self):
		return self.accuracy
	def setAccuracy(self, pAccuracy: float):
		self.accuracy = pAccuracy
	def setVerbose(self, pVerbose: bool):
		self.verbose = pVerbose
	
	def validate_current_html(self,contents):
		'''
		This method validates HTML, by checking if certain strings are contained within the HTML
		'''
		txts = ["<html>","<head>","<meta charset='UTF-8'>", "Team Solver", "</head>", "<body>"]
		for txt in txts:
			if txt not in contents:
				return False
		return True
	
	def save_html(self, filename: str, date: str, mode: str = "a+", pIndex: int | None = None):
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
		{self.team_to_html(pIndex)}
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

	def save_summary(self,filename, mode: str = "a+", date: str = "", pIndex: int | None = None):
		if("." not in filename):
			raise f"Error: file extension not found in '{filename}'"
		filename_split = filename.split(".")
		file_extension = filename_split[-1]
		match file_extension:
			case "html":
				self.save_html(filename,date,mode, pIndex=pIndex)
			case "txt":
				self.save_txt(filename,mode)
			case _:
				raise NotImplementedError(f"File extension '{file_extension}' has not yet been implemented")

	def toDict(self) -> dict:
		start, bench = self.splitStartBench()
		bench["is_benched"] = True
		start["is_benched"] = False
		team = pd.concat([start,bench])
		#print(team["status"])
		team = team.sort_values(by=["position","is_benched","score"])
		if "value" in team.columns:
			team = team.drop(columns=["value"])
 
		teamDict: dict = team.to_dict(orient="records")
		otherData: dict = self.latestData.to_dict(orient="records")
		return {
			"team": teamDict,
			"players": otherData
		}

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

	def adjust_players(self,cost,id, position: str, current_score: float) -> bool:
		MINS_THREHSHOLD = 0.5
		old_players = self.players[position]
		self.players[position] = self.players[position][self.players[position]["id"]!=id]
		new_players: pd.DataFrame = self.default_players[position][(self.default_players[position]["cost"] < cost)]
		# TODO: Fix bug where players with 0 play chance are being selected
		new_players = new_players.loc[(new_players["form"] > 0.0) & (new_players["play_percent"] >= MINS_THREHSHOLD)]
		new_players = new_players[~(new_players["id"].isin(self.players[position]["id"]))]
		new_players = new_players.sort_values(by="score",ascending=False)
		flatPlayers: pd.DataFrame = self.flattenTeam()
		teamCounts: pd.Series = flatPlayers["team"].value_counts()
		#print("new_players=")
		#print(new_players)
		boolArr: pd.Series[bool] = new_players["team"].apply(lambda x: teamCounts.get(x, 0) < 3)
		new_players = new_players.loc[boolArr]

		#if (position == "DEF" and self.label =="Random Forest"):
		#	print(new_players)
		#print("test")
		new_players = new_players.head(1)
		#assert new_players["id"].values[0] != 315.0
		if(len(new_players) == 0):
			#if position == "DEF":
			#	print("WARNING: Unable to find players who fit all criteria")
			#	...
			self.players[position] = old_players
			return False
		self.players[position] = pd.concat([self.players[position],new_players])
		self.check_players(position)
		return True

	def print_summary(self):
		actualTotalCost = self.total_cost
		print("Iteration:",str(self.iter)+"\tCost:",str(actualTotalCost) + "\tTotal Score:",str(self.total_score) + "\tProfit:",str(self.profit))

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
		# TODO: Refactor program so that starting team and bench team are stored in the same DF
		# This would make it so that when doing `get_nth_cheapest()`/`get_nth_most_expensive()`, benches will be taken into account too
		team = self.flattenTeam()
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
		options = options.loc[~(options["id"].isin(team["id"]))]
		actual_options = options.loc[options["score"] > worst_player["score"]]
		flatPlayers: pd.DataFrame = self.flattenTeam()
		teamCounts: pd.Series = flatPlayers["team"].value_counts()
		boolArr: pd.Series[bool] = actual_options["team"].apply(lambda x: teamCounts.get(x, 0) < 3)
		actual_options = actual_options.loc[boolArr]

		amount_remaining = MAX_BUDGET - self.total_cost
		if len(actual_options) > 0:
			max_price = worst_player["cost"] + amount_remaining
			actual_options = actual_options.loc[actual_options["cost"] <= max_price]
			if(len(actual_options) > 0):
				best_option_index = actual_options["score"].argmax()
				best_option = actual_options.iloc[[best_option_index]]
				new_team_df = self.replace_player(team,worst_player["id"],best_option)
				assert (self.latestData["id"].values == self.latestData.index.values).all()
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
		amount_remaining = MAX_BUDGET - self.total_cost

	def update_stats(self):
		self.total_cost = self.sum_stat("cost")
		self.total_score = self.sum_stat("score")
		self.profit = self.budget - self.total_cost

	def removeOutliers(self):
		positions = {"DEF", "FWD", "MID", "GKP"}
		pScoreCutoff = 0
		for position in positions:
			scoresOfPosition = self.latestData.loc[self.latestData["position"]==position]["score"]
			scorePScores = self.calcPScores(scoresOfPosition)
			mask = scorePScores > pScoreCutoff
			# If outlier, set score to something exceedingly low, so that the player will not be chosen
			# Due to the score being required by other files (such as 05-get_best_transfer.ipynb), entirely removing outliers does not work.
			self.latestData.loc[self.latestData["position"] == position].loc[mask] = -99
		self.latestData = self.latestData.dropna()
		pass

	def queryPlayerScore(self, pPlayerName: str):
		foundPlayer = self.latestData.loc[self.latestData["name"] == pPlayerName]
		return foundPlayer["score"].values[0]
	def queryPlayerScores(self, pPlayers: pd.DataFrame):
		players = self.latestData.loc[self.latestData["id"] == pPlayers["id"]]
		return players["score"]

	def find_team(self):
		self.update_stats()
		amount_remaining = MAX_BUDGET - (self.total_cost)
		if (self.total_cost <= self.budget):
			iter = 0
			while (amount_remaining >= MAX_AMT_REMAINING) and (iter < config.MAX_ITERS):
				#if(self.verbose):
				#	print(f"Iter: {iter}")
				self.backward_adjust()
				self.update_stats()
				amount_remaining = MAX_BUDGET - (self.total_cost)
				iter += 1

			self.update_stats()

			if(self.verbose):
				print("Successfully found team!")
			return
		
		if(self.verbose):
			self.print_summary()
		self.adjust_team(0)
		assert (self.latestData["id"].values == self.latestData.index.values).all()
		self.iter += 1

		if self.iter > self.max_iters:
			print("Failed to find optimal team.")
		else:
			return self.find_team()