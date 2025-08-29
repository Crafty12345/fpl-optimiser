import platform
from glob import glob
import re
import pandas as pd

def lerp(pMin: float, pMax: float, pPercent: float) -> float:
    return pMin + (pMax - pMin) * pPercent

def getDataFilesSorted() -> list[dict]:
    allDataFiles = sorted(glob(r"./data/player_stats/**/data_*.csv"))
    filesSorted = []
    userPlatform = platform.system()
    if (userPlatform == "Windows"):
        # TODO: Test Windows regex
        regex = r"^(\.\/data\/player_stats)\/(\d{2})\/data_(\d+)(\.csv$)"
        #regex = r"^(\.\/data\/player_stats\\data_)(\d+)(\.csv)"
    elif (userPlatform == "Linux"):
        regex = r"(^\.\/data\/player_stats)\/(\d{2})\/data_(\d+)(\.csv$)"
        pass
    else:
        raise NotImplementedError(f"Support for {userPlatform} is not yet implemented.")
    pattern = re.compile(regex)
    assert len(allDataFiles) >= 1
    for file in allDataFiles:
        fileNameSplit = re.split(pattern,file)
        if (len(fileNameSplit) == 6):
            season = fileNameSplit[2]
            gameweek = fileNameSplit[3]
            # TODO: Keep working on this bit
            dictObj = {
                "name": file,
                "season": int(season),
                "gameweek": int(gameweek)
            }
            filesSorted.append(dictObj)
        else:
            raise ValueError(f"Invalid filename {file}")
    filesSorted.sort(key=lambda x: (x["season"], x["gameweek"]))
    return filesSorted

def team_from_code(team_code: int, pTeamData: pd.DataFrame):
    return pTeamData.loc[pTeamData["code"]==team_code]["short_name"].values[0]

def teamFromId(teamId: int, pTeamData: pd.DataFrame):
    return pTeamData.loc[pTeamData["id"]==teamId]["short_name"].values[0]