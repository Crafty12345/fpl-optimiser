#! /usr/bin/env python3

import pandas as pd
import json
from glob import glob
import re
import datetime
import numpy as np
from pathlib import Path
import json

import config
from modules.data_file import DataFile, RawDataFile

def get_position_name(position: int):
    match position:
        case 1:
            return "GKP"
        case 2:
            return "DEF"
        case 3:
            return "MID"
        case 4:
            return "FWD"
        
def sortFiles(pFiles: list[str]) -> list[DataFile]:
    # Pattern that makes sure files are named correctly
    pattern = r"^\.\/data\/raw\/player_stats\/\d{2}-\d{2}\/fpl_stats_\d{2}-\d{2}-\d{4}\.json$"
    regex = re.compile(pattern)

    datePattern = r"\d{2}-\d{2}-\d{4}"
    dateRegex = re.compile(datePattern)

    dates: list[DataFile] = []
    for file in pFiles:
        if not (regex.fullmatch(file)):
            raise LookupError(f"Invalid file: {file}")
        else:
            newFile = RawDataFile.parse(file)
            dates.append(newFile)
    dates = sorted(dates)
    return dates

def team_from_code(team_code, pTeamData: pd.DataFrame):
    return pTeamData.loc[pTeamData["code"]==team_code]["short_name"].values[0]

def processFile(pDate: RawDataFile) -> pd.DataFrame:
    '''
    Function to clean data
    '''

    with open(pDate.filename) as f:
        all_data_raw = json.load(f)

    # Need eventData in order to get correct gameweek
    eventData = pd.DataFrame(all_data_raw["events"])
    eventData = eventData.dropna(subset=["most_captained"])

    if len(eventData) < 1:
        currentGameweek = 0
    else:
        currentGameweek = eventData["id"].values[-1]

    df = pd.DataFrame(all_data_raw["elements"])
    
    allowed_cols = ["id", "first_name","second_name","now_cost","ict_index","total_points","points_per_game","element_type","team_code","form", "status", "starts_per_90"]
    player_data = df[allowed_cols]
    player_data["position"] = player_data["element_type"].apply(lambda x: get_position_name(x))

    player_data["ict_index"] = player_data["ict_index"].astype(float)
    player_data["points_per_game"] = player_data["points_per_game"].astype(float)
    player_data["id"] = player_data["id"].astype(int)

    player_data = player_data.drop(columns=["element_type"])
    player_data = player_data.rename(columns={"now_cost":"cost"})

    team_data = pd.DataFrame(all_data_raw["teams"])
    team_data = team_data[["code","short_name"]]

    player_data["team"] = player_data["team_code"].apply(lambda x: team_from_code(x, team_data))
    player_data = player_data.drop(columns=["team_code"])

    player_data["first_name"] = player_data["first_name"] + " " + player_data["second_name"]
    player_data = player_data.drop(columns=["second_name"])
    player_data = player_data.rename(columns={"first_name": "name"})

    player_data["form"] = pd.to_numeric(player_data["form"])

    # if (pDate.season.endYear == 26):
    #     print(player_data.loc[player_data["name"]=="Alexander Isak"])
    #     assert False

    return player_data, currentGameweek

allFiles = glob("./data/raw/player_stats/**-**/*.json")


# Source: https://fantasy.premierleague.com/api/bootstrap-static/

filesSorted: list[RawDataFile] = sortFiles(allFiles)
filesProcessed = []
for i, fileName in enumerate(filesSorted):
    data, gameweek = processFile(fileName)
    filesProcessed.append({
        "gameweek": gameweek,
        "season": fileName.season.endYear,
        "data": data
    })

filteredFiles = []
seenGameweeks: dict[str, set] = dict()
seenSeasons: set[int] = set()

# Make sure each gameweek only has 1 file
for file in filesProcessed:
    season = file["season"]
    if(season not in seenSeasons):
        seenGameweeks[season] = set()
        seenSeasons.add(season)

    if(file["gameweek"] not in seenGameweeks[season]):
        tempFiles = []
        for file2 in filesProcessed:
            # Check same gameweek
            if((file2["gameweek"] == file["gameweek"]) and (file2["season"] == season)):
                tempFiles.append(file2)
        # Since files are already sorted, we can just add the last file in the list
        # This ensures that only the latest data for each gameweek is used
        filteredFiles.append(tempFiles[-1])
        seenGameweeks[season].add(tempFiles[-1]["gameweek"])

previousForm: pd.DataFrame = None
allJsonData = dict()

for file in filteredFiles:

    # Cast to Python `int` to fix compatibility with 
    gameweek = int(file["gameweek"])
    season = int(file["season"])

    if season not in allJsonData.keys():
        allJsonData[season] = dict()
    allJsonData[season][gameweek] = dict()

    # If all players' form == 0.0, then use form of previous gameweek
    if (file["data"]["form"] == 0.0).all():
        assert previousForm is not None
        averageForm = previousForm["form"].mean()
        file["data"] = file["data"].merge(previousForm, on="name")
        file["data"]["form_x"] = file["data"]["form_y"]
        file["data"] = file["data"].drop(["form_y"], axis="columns")
        file["data"] = file["data"].rename({"form_x": "form"}, axis="columns")

    else:
        previousForm = file["data"][["name", "form"]]

    file["data"] = file["data"].sort_values(by="id")

    # Fixes an issue where Python's `json` library expects a Python int (not int64 which is used by Pandas)
    tempDict: dict = file["data"].to_dict(orient="records")
    allJsonData[season][gameweek] = tempDict

dataDir = f"./data/player_stats.json"
with open(dataDir, "w+") as f:
    json.dump(allJsonData, f,indent=4)