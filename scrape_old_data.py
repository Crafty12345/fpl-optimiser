import requests
import json
from tqdm import tqdm
from time import sleep
import random

def loadPlayer(pId: int, pBaseUrl: str) -> dict:
    url: str = pBaseUrl + f"?player={pId}"
    response = json.loads(requests.get(url).text)
    sleep(random.random())
    return response

def renameColumns(pPlayerData: dict) -> None:
    pPlayerData["cost"] = pPlayerData["value"]
    del pPlayerData["value"]


def scrapeSeason(pSeason: str):
    pointFilename = f"data/raw/old_data/points/{pSeason[2::]}.json"

    url = f"https://www.fantasynutmeg.com/api/history/season/{pSeason}"
    seasonData = json.loads(requests.get(url).text)
    with open(pointFilename, "w+") as f:
        json.dump(seasonData, f, indent=4)

    allData: dict = dict()

    for datum in tqdm(seasonData["matrix"]):
        playerData = loadPlayer(datum["id"], url)
        for week, val in enumerate(playerData["history"]):
            if (week not in allData.keys()):
                allData[week] = dict()
                allData[week]["elements"] = []
            val["web_name"] = datum["web_name"]
            team = val["team_name"]
            
            val["opposing_team"] = val["opponent_team_name"]
            del val["opponent_team_name"]

            val["now_cost"] = val["value"]
            del val["value"]

            val["id"] = datum["id"]

            allData[week]["elements"].append(val)
            allData[week]["events"] = []
            allData[week]["events"].append({"id": week+1, "is_next": True})

    outDataFilename = f"data/raw/old_data/players/{pSeason[2::]}.json"

    with open(outDataFilename, "w+") as f:
        json.dump(allData, f, indent=4)

season = "2024-25"
scrapeSeason(season)