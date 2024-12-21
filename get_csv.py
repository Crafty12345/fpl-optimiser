import pandas as pd
import json
import config
from glob import glob
import re
import datetime
import numpy as np

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
        
def sortFiles(pFiles: list[str]) -> list[str]:
    # Pattern that makes sure files are named correctly
    pattern = r"^\.\/data\/raw\/player_stats\/fpl_stats_\d{2}-\d{2}-\d{4}\.json"
    regex = re.compile(pattern)

    datePattern = r"\d{2}-\d{2}-\d{4}"
    dateRegex = re.compile(datePattern)

    dates: list[datetime.date] = []
    for file in pFiles:
        if not (regex.fullmatch(file)):
            raise LookupError(f"Error: Invalid file: {file}")
        else:
            date = dateRegex.findall(file)[0]
            year = int(date[-4::])
            month = int(date[3:5])
            day = int(date[0:2])
            dates.append(datetime.date(year, month, day))
    dates = sorted(dates)
    dateStrings = [date.strftime("%d-%m-%Y") for date in dates]
    return dateStrings

def team_from_code(team_code, pTeamData: pd.DataFrame):
    return pTeamData.loc[pTeamData["code"]==team_code]["short_name"].values[0]

def processFile(pDate: str) -> pd.DataFrame:
    '''
    Function to clean data
    '''
    currentGameweek = -1
    with open(f"./data/raw/player_stats/fpl_stats_{pDate}.json") as f:
        all_data_raw = json.load(f)
    df = pd.DataFrame(all_data_raw["elements"])
    
    allowed_cols = ["first_name","second_name","now_cost","ict_index","total_points","points_per_game","element_type","team_code","form", "status", "starts_per_90"]
    player_data = df[allowed_cols]
    player_data["position"] = player_data["element_type"].apply(lambda x: get_position_name(x))
    player_data = player_data.drop(columns=["element_type"])
    player_data = player_data.rename(columns={"now_cost":"cost"})

    team_data = pd.DataFrame(all_data_raw["teams"])
    team_data = team_data[["code","short_name"]]

    player_data["team"] = player_data["team_code"].apply(lambda x: team_from_code(x, team_data))
    player_data = player_data.drop(columns=["team_code"])

    player_data["first_name"] = player_data["first_name"] + " " + player_data["second_name"]
    player_data = player_data.drop(columns=["second_name"])
    player_data = player_data.rename(columns={"first_name": "name"})

    eventData = pd.DataFrame(all_data_raw["events"])
    eventData = eventData.dropna(subset=["most_captained"])
    currentGameweek = eventData["id"].values[-1]

    return player_data, currentGameweek

allFiles = glob("./data/raw/player_stats/*.json")

# Source: https://fantasy.premierleague.com/api/bootstrap-static/

filesSorted = sortFiles(allFiles)
print(filesSorted)
filesProcessed = []
for i, fileName in enumerate(filesSorted):
    data, gameweek = processFile(fileName)
    filesProcessed.append({
        "gameweek": gameweek,
        "data": data
    })

filesFiltered = []
seenGameweeks = set()
# Make sure each gameweek only has 1 file
for file in filesProcessed:
    if(file["gameweek"] not in seenGameweeks):
        tempFiles = []
        for file2 in filesProcessed:
            if(file2["gameweek"] == file["gameweek"]):
                tempFiles.append(file2)
        filesFiltered.append(tempFiles[-1])
        seenGameweeks.add(tempFiles[-1]["gameweek"])

for file in filesFiltered:
    gameweek = file["gameweek"]
    file["data"].to_csv(f"./data/player_stats/data_{gameweek}.csv",index_label="id")