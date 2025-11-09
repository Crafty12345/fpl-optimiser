import requests
import json
from tqdm import tqdm
from time import sleep
import random
from glob import glob
import lzma
import os
import re

from modules.data_file import Season

def parseFilename(pFilename: str) -> dict[str, int]:
    pattern = r"^external\/fplcache\/cache\/(\d+)\/(\d+)$"
    regex = re.compile(pattern)
    year, month = regex.findall(pFilename)[0]
    return {
        "year": int(year),
        "month": int(month)
    }

def getFileTime(pFilename: str) -> int:
    pattern = r"^external\/fplcache\/cache\/\d+\/\d+\/\d+\/(\d+)\.json\.xz$"
    regex = re.compile(pattern)
    return int(regex.findall(pFilename)[0])

def parseDay(pFilename: str) -> int:
    pattern = r"^external\/fplcache\/cache\/\d+\/\d+\/(\d+)$"
    regex = re.compile(pattern)
    return int(regex.findall(pFilename)[0])

def writeFile(pFilename: str, pDate: dict, pSeason: Season) -> None:
    # TODO: Write contents of pFilename to output file
    seasonStr = f"{pSeason.startYear}-{pSeason.endYear}"
    outputDirectory = f"data/raw/old_data/players/{seasonStr}"
    os.makedirs(outputDirectory, exist_ok=True)
    outputFilename = f"{outputDirectory}/fpl_stats_{pDate['day']:02}-{pDate['month']:02}-{pDate['year']:04}.json"

    with lzma.open(pFilename, "rt", encoding="utf-8") as f:
        jsonData = json.loads(f.read())

    with open(outputFilename, "w+") as f:
        json.dump(jsonData, f, indent=4)

def writeTeamTable(pFilename: str, pSeason: Season):
    currentData = dict()
    teamTableFilename = "data/team_translation_table.json"
    if (os.path.exists(teamTableFilename)):
        with open("data/team_translation_table.json", "r") as f:
            currentData = json.load(f)

    if (str(pSeason.endYear)) in currentData.keys():
        # Nothing to do (team translation table has already been created)
        return

    with lzma.open(pFilename, "rt", encoding="utf-8") as f:
        jsonData: list[dict] = json.loads(f.read())
    outputList: list[dict] = []
    requiredKeys = ["code", "id", "short_name"]
    for tempDict in jsonData:
        outDict = dict()
        for key in requiredKeys:
            outDict[key] = tempDict[key]
        outputList.append(outDict)
    currentData[pSeason.startYear] = outputList
    with open(teamTableFilename, "w+") as f:
        json.dump(currentData, f, indent=4)
    

def loadCachedFiles(pSeason: Season) -> list[dict]:
    START_MONTH: int = 8
    END_MONTH: int = 5
    startYearFull = "20" + str(pSeason.startYear)
    startYearGlob = glob(f"external/fplcache/cache/{startYearFull}/*")
    toUse = []

    for file in tqdm(startYearGlob):
        date = parseFilename(file)
        if date["month"] >= START_MONTH:
            dayGlob = glob(file + "/*")
            for dayDir in dayGlob:
                day = parseDay(dayDir)
                timeGlob = glob(dayDir + "/*")
                bestTime = max(timeGlob, key=lambda x: getFileTime(x))
                date = {
                    "year": date["year"],
                    "month": date["month"],
                    "day": day
                    }
                writeTeamTable(bestTime, pSeason)
                writeFile(bestTime, date, pSeason)

def retrieveFixtures(pApiUrl: str, pPlayerId: int) -> dict:
    # Note to self: opponent_team_name
    playerUrl = pApiUrl + "?player=" + str(pPlayerId)
    response = requests.get(playerUrl)
    if not (response.ok):
        return None
    history: list[dict] = json.loads(response.text)["history"]
    result: dict[int, dict] = dict()
    for data in history:
        week = data["round"]
        result[week] = dict()
        team: str = data["team_name"]
        opposingTeam: str = data["opponent_team_name"]
        if data["was_home"]:
            result[week]["home_team"] = team
            result[week]["away_team"] = opposingTeam
        else:
            result[week]["away_team"] = team
            result[week]["home_team"] = opposingTeam
    return result

def findTeam(pSeenTeams: dict[int, set[str]], pTeam: str):
    if (len(pSeenTeams) > 0):
        for (week, tempSet) in pSeenTeams.items():
            if pTeam in tempSet:
                return True
            else:
                return False
    return False

def getFixtureHistory(pSeason: Season):
    baseUrl = "https://www.fantasynutmeg.com/api/history/season/"
    seasonFormatted = f"20{pSeason.startYear}-{pSeason.endYear}"
    apiUrl: str = baseUrl + seasonFormatted
    response = requests.get(apiUrl)
    if (response.status_code != 200):
        print(f"Unable to retrieve data. HTTP status code: {response.status_code}")
        return
    history: list[dict] = json.loads(response.text)["history"]
    fixtures: dict[int, list[dict]] = dict()
    playerIds: list[int] = []
    seenTeams: dict[int, set[str]] = dict()
    for val in tqdm(history):
        team = val["team_name"]
        if not findTeam(seenTeams, team):
            toMerge = retrieveFixtures(apiUrl, val["id"])
            if (toMerge is not None):
                for week, datum in toMerge.items():
                    if week not in fixtures.keys():
                        fixtures[week] = list()
                    if week not in seenTeams.keys():
                        seenTeams[week] = set()
                    seenTeams[week].add(datum["home_team"])
                    seenTeams[week].add(datum["away_team"])
                    if datum not in fixtures[week]:
                        fixtures[week].append(datum)


    outputDir = f"data/raw/fixture_data/{pSeason.endYear}"
    for week, fixture in fixtures.items():
        outputFilename = f"fixture_data_{week}.json"
        outputPath = outputDir + "/" + outputFilename
        print(fixture)
        print("\n\n")
        for val in fixture:
            val["finished"] = True
            val["event"] = week
        with open(outputPath, "w+") as f:
            json.dump(fixture, f, indent=4)




season: Season = Season(24,25)
#loadCachedFiles(season)
getFixtureHistory(season)