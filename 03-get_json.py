#! /usr/bin/env python3

import pandas as pd
import json
from glob import glob
import re
import datetime
import numpy as np
from pathlib import Path
import json
import os
import lzma

import config
from modules.data_file import DataFile, RawFixtureDataFile, RawPlayerDataFile, Season

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
        
def parseFileNames(pFiles: list[str]) -> list[DataFile]:
    # Pattern that makes sure files are named correctly
    pattern = r"^\.\/data\/raw\/player_stats\/\d{2}-\d{2}\/fpl_stats_\d{2}-\d{2}-\d{4}\.json$"
    regex = re.compile(pattern)

    dates: list[DataFile] = []
    for file in pFiles:
        if not (regex.fullmatch(file)):
            raise LookupError(f"Invalid file: {file}")
        else:
            newFile = RawPlayerDataFile.parse(file)
            dates.append(newFile)
    return dates

def parseOldFilenames(pFiles: list[str]) -> list[DataFile]:
    # Pattern that makes sure files are named correctly
    pattern = r"^data\/raw\/old_data\/players\/(\d{2})-(\d{2})\/fpl_stats_(\d{2})-(\d{2})-(\d{4})\.json$"
    regex = re.compile(pattern)

    dates: list[DataFile] = []
    for file in pFiles:
        if not (regex.fullmatch(file)):
            raise LookupError(f"Invalid file: {file}")
        else:
            newFile = RawPlayerDataFile.parseOld(file)
            dates.append(newFile)
    return dates

def team_from_code(team_code: int, pTeamData: pd.DataFrame) -> str:
    return pTeamData.loc[pTeamData["code"]==team_code]["short_name"].values[0]

def teamFromId(teamId: int, pTeamData: pd.DataFrame) -> str:
    return pTeamData.loc[pTeamData["id"]==teamId]["short_name"].values[0]

def getOpposingTeam(pPlayingTeamCode: int, pTeamData: pd.DataFrame, pFixtureData: pd.DataFrame) -> str:
    #print(pFixtureData[["team_h","team_a"]])

    playingTeamId: int = pTeamData.loc[pTeamData["code"]==pPlayingTeamCode]["id"].values[0]
    homeTeamResult = pFixtureData.loc[pFixtureData["team_h"]==playingTeamId]

    resultInt: int = -1
    # If team is playing home that week
    if(len(homeTeamResult) >= 1):
        resultInt = homeTeamResult["team_a"].values[0]
    else:
        # If team is playing away that week
        awayTeamResult = pFixtureData.loc[pFixtureData["team_a"]==playingTeamId]
        if (len(awayTeamResult) > 0):
            resultInt = awayTeamResult["team_h"].values[0]
    if resultInt > -1:
        return teamFromId(resultInt, pTeamData)
    else:
        # There may be weeks in which teams don't have a game
        return "UNK"

def getOldOpposingTeam(pPlayingTeam: str, pFixtures: pd.DataFrame):
    homeTeamResult = pFixtures.loc[pFixtures["home_team"]==pPlayingTeam]
    if (len(homeTeamResult) > 0):
        return homeTeamResult["away_team"].values[0]
    else:
        awayTeamResult = pFixtures.loc[pFixtures["away_team"] == pPlayingTeam]
        if len(awayTeamResult) > 0:
            return awayTeamResult["home_team"].values[0]
    return "UNK"

def updateDict(pId: int, pName: str, pPlayerNameDict: dict[int, str]):
    pPlayerNameDict[pId] = pName


def processFile(pRawData: dict, pDate: RawPlayerDataFile, pOldData: dict[str,list[dict]], pPlayerNameDict: dict[int, str]) -> pd.DataFrame:
    '''
    Function to clean data
    '''

    # Need eventData in order to get correct gameweek
    eventData = pd.DataFrame(pRawData["events"])
    nextWeek = eventData.loc[eventData["is_next"]==True]

    if len(nextWeek) < 1:
        currentGameweek = 0
    else:
        nextGameweek: int = nextWeek["id"].values[-1]
        currentGameweek = nextGameweek - 1

    df = pd.DataFrame(pRawData["elements"])
    
    allowed_cols = ["id", "first_name","second_name","now_cost","ict_index","total_points","points_per_game","element_type", "form", "status", "clean_sheets", "minutes"]
    if "team_name" in df.columns:
        allowed_cols.append("team_name")
    elif "team_code" in df.columns:
        allowed_cols.append("team_code")
    else:
        raise KeyError(f"Unable to find team in gameweek {currentGameweek} of season {currentSeason}")
    
    player_data = df[allowed_cols]

    player_data["clean_sheets"] = player_data["clean_sheets"].astype(np.float64)
    #player_data["expected_goals"] = player_data["expected_goals"].astype(np.float64)
    playPercent = player_data["minutes"] / (currentGameweek + 1) / 90.00
    player_data = player_data.drop(columns=["minutes"])
    player_data["play_percent"] = playPercent

    player_data["position"] = player_data["element_type"].apply(lambda x: get_position_name(x))

    player_data["ict_index"] = player_data["ict_index"].astype(float)
    player_data["points_per_game"] = player_data["points_per_game"].astype(float)
    player_data["id"] = player_data["id"].astype(int)

    player_data = player_data.drop(columns=["element_type"])
    player_data = player_data.rename(columns={"now_cost":"cost"})

    team_data = pd.DataFrame(pRawData["teams"])
    team_data = team_data[["code", "id","short_name"]]

    print(f"Processing file '{pDate.filename}'")
    print(f"currentGameweek={currentGameweek}")
    print(f"currentSeason={pDate.season.endYear}")
    if "team_code" in player_data.columns:
        player_data["team"] = player_data["team_code"].apply(lambda x: team_from_code(x, team_data))
    elif "team_name" in player_data.columns:
        player_data["team"] = player_data["team_name"]
    else:
        raise ValueError(f"Team not found for player {player_data['id']}")
    
    player_data["gameweek"] = currentGameweek
    player_data["season"] = pDate.season.endYear

    # UNK = "UNKOWN"
    if "opposing_team" not in player_data.columns:
        player_data["opposing_team"] = "UNK"
        fixtureFilename = f"./data/raw/fixture_data/{pDate.season.endYear}/fixture_data_{currentGameweek}.json"
        altFixtureFilename = f"data/raw/old_data/fixtures/"
        if(os.path.isfile(fixtureFilename)):
            with open(fixtureFilename, "r") as f:
                dataJson = json.load(f)
            fixtures = pd.DataFrame.from_records(dataJson)
            if ("team_h" in fixtures) and ("team_a" in fixtures):
                player_data["opposing_team"] = player_data["team_code"].apply(lambda x: getOpposingTeam(x, team_data, fixtures))
            else:
                player_data["opposing_team"] = player_data["team"].apply(lambda x: getOldOpposingTeam(x, fixtures))

    if "team_code" in player_data.columns:
        player_data = player_data.drop(columns=["team_code"])
    
    if ("first_name" in player_data.columns and "second_name" in player_data.columns):
        player_data["first_name"] = player_data["first_name"] + " " + player_data["second_name"]
        player_data = player_data.drop(columns=["second_name"])
        player_data = player_data.rename(columns={"first_name": "name"})
        player_data.apply(lambda x: updateDict(x["id"], x["name"], pPlayerNameDict), axis=1)
    else:
        player_data["name"] = player_data["id"].apply(lambda x: pPlayerNameDict[x])


    player_data["form"] = pd.to_numeric(player_data["form"])
    
    player_data["points_this_week"] = 0

    actualPointsFilename: str = f"./data/raw/weekly_points/{pDate.season.endYear}/{currentGameweek}.json"
    if(os.path.isfile(actualPointsFilename)):
        with open(actualPointsFilename, "r") as f:
            actualPointsAllTemp = json.load(f)
        actualPointsAll: list[dict] = actualPointsAllTemp["elements"]
        for tempDict in actualPointsAll:
            _id = tempDict["id"]
            receivedPoints: float = tempDict["stats"]["total_points"]
            player_data.loc[player_data["id"]==_id, "points_this_week"] = receivedPoints
        player_data["points_this_week"] = player_data["points_this_week"].fillna(0.0)
    else:
        oldSeasonPoints: list[dict] = pOldData.get(str(pDate.season.endYear), list())
        for temp in oldSeasonPoints:
            playerId = int(temp["id"])
            pointsAtGameweek = temp["gw_points"].get(str(currentGameweek), 0)
            # Fix an edge case where some players may not play anymore
            if (playerId in player_data["id"]):
                player_data.loc[player_data["id"]==playerId, "points_this_week"] = pointsAtGameweek

    return player_data, currentGameweek, pRawData["teams"]

def filterDuplicates(pToFilter: list[DataFile]):
    seenGameweeks: dict[str, set] = dict()
    seenSeasons: set[int] = set()
    filteredFiles: list[dict] = []
    for file in pToFilter:
        season = file["season"]
        if(season not in seenSeasons):
            seenGameweeks[season] = set()
            seenSeasons.add(season)

        if(file["gameweek"] not in seenGameweeks[season]):
            tempFiles = []
            for file2 in pToFilter:
                # Check same gameweek
                if((file2["gameweek"] == file["gameweek"]) and (file2["season"] == season)):
                    tempFiles.append(file2)
            # Since files are already sorted, we can just add the last file in the list
            # This ensures that only the latest data for each gameweek is used
            filteredFiles.append(tempFiles[-1])
            seenGameweeks[season].add(tempFiles[-1]["gameweek"])
    return filteredFiles

allFiles = glob("./data/raw/player_stats/**-**/*.json")\

with open("./data/raw/old_data/points/old_points.json", "r") as f:
    oldPointData = json.load(f)

# Source: https://fantasy.premierleague.com/api/bootstrap-static/
filesSorted: list[RawPlayerDataFile] = parseFileNames(allFiles)

allOldFiles = glob("data/raw/old_data/players/**/**.json")
filesSorted += parseOldFilenames(allOldFiles)
filesSorted = sorted(filesSorted)

filesProcessed: list[dict] = []
playerNames: dict[int, str] = dict()

for i, fileName in enumerate(filesSorted):
    with open(fileName.filename, "r") as f:
        jsonData = json.load(f)
    data, gameweek, teams = processFile(jsonData, fileName, oldPointData, playerNames)
    filesProcessed.append({
        "gameweek": gameweek,
        "season": fileName.season.endYear,
        "data": data,
        "teams": teams
    })

#print("Finished processing new data. Processing old data now.")

filteredFiles: list[dict] = filterDuplicates(filesProcessed)
temp = []
for file in filteredFiles:
    temp.append((file["gameweek"], file["season"]))


# Make sure each gameweek only has 1 file

previousForm: pd.DataFrame = None
allJsonData = dict()
teamDict: dict[int, dict] = dict()

for (i, file) in enumerate(filteredFiles):

    ignoreFile = False

    # Cast to Python `int` to fix compatibility with 
    gameweek = int(file["gameweek"])
    season = int(file["season"])
    teamDf = pd.DataFrame.from_records(file["teams"])

    # Ignore files where opposing team is unknown
    if((file["data"]["opposing_team"]=="UNK").all()):
        print(f"Warning: Unable to find opposing team of week {gameweek} of season {season}")
        ignoreFile = True

    if not ignoreFile:
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
    

    # TODO: Fix this bit
    # If before last element
    if (i < len(filteredFiles)-1):
        nextFile = filteredFiles[i+1]
        nextSeason = int(nextFile["season"])
        # If last element of currently-selected season
        if (nextSeason > season):
            toAdd: pd.DataFrame = pd.DataFrame(columns=["name", "id", "code"])
            toAdd["name"] = teamDf["short_name"]
            toAdd["id"] = teamDf["id"]
            toAdd["code"] = teamDf["code"]
            teamDict[season] = toAdd.to_dict(orient="records")

    # If is last element
    elif (i == len(filteredFiles) - 1):
        # TODO: Fix
        toAdd: pd.DataFrame = pd.DataFrame(columns=["name", "id", "code"])
        toAdd["name"] = teamDf["short_name"]
        toAdd["id"] = teamDf["id"]
        toAdd["code"] = teamDf["code"]
        teamDict[season] = toAdd.to_dict(orient="records")

dataDir = f"./data/player_stats.json"
with open(dataDir, "w+") as f:
    json.dump(allJsonData, f,indent=4, allow_nan=True, sort_keys=True)

with open(f"./data/team_translation_table.json", "w+") as f:
    json.dump(teamDict, f, indent=4, sort_keys=True)


allFixtureFileNames = glob("./data/raw/fixture_data/**/**.json")
fixtureDataMerged = dict()
tempFixtureFiles: list[RawFixtureDataFile] = []

for filename in allFixtureFileNames:
   currentFile: RawFixtureDataFile = RawFixtureDataFile.parse(filename)
   tempFixtureFiles.append(currentFile)

tempFixtureFiles = sorted(tempFixtureFiles)

#fixtureDataFiltered: list[RawFixtureDataFile] = filterDuplicates(tempFixtureFiles)
results: list[dict] = dict()

for file in tempFixtureFiles:
    currentSeason = file.season.endYear
    teamDataDf = pd.DataFrame.from_records(teamDict[currentSeason])
    with open(file.filename, "r") as f:
        fixtureDatum: list[dict] = json.load(f)
    for tempDict in fixtureDatum:
        toAdd = dict()

        actualGameweek = tempDict["event"]
        toAdd["finished"] = tempDict["finished"]
        if "team_h" in tempDict.keys():
            homeTeamData = teamDataDf.loc[teamDataDf["id"]==tempDict["team_h"]]
            toAdd["home_team"] = homeTeamData["name"].values[0]
        else:
            toAdd["home_team"] = tempDict["home_team"]

        if "team_a" in tempDict.keys():
            awayTeamData = teamDataDf.loc[teamDataDf["id"]==tempDict["team_a"]]
            toAdd["away_team"] = awayTeamData["name"].values[0]
        else:
            toAdd["away_team"] = tempDict["away_team"]

        if(currentSeason not in results.keys()):
            results[currentSeason] = dict()
        if (actualGameweek not in results[currentSeason].keys()):
            results[currentSeason][actualGameweek] = []
        results[currentSeason][actualGameweek].append(toAdd)

print(results)

with open("./data/fixtures.json", "w+") as f:
    json.dump(results, f, indent=4, sort_keys=True)