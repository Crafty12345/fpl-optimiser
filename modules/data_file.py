from dataclasses import dataclass
from datetime import datetime
import re
from abc import ABC as AbstractClass
from abc import abstractmethod

@dataclass
class Season(object):
    startYear: int
    endYear: int
    def __gt__(self, pOther: 'Season'):
        return self.endYear > pOther.endYear
    def __lt__(self, pOther: 'Season'):
        return self.endYear < pOther.endYear
    def __eq__(self, pOther: 'Season'):
        return self.endYear == pOther.endYear

@dataclass
class DataFile(AbstractClass):
    season: Season
    filename: str
    comparable: datetime | int

    @abstractmethod
    def parse(pFilename:str) -> 'DataFile': ...

    def __lt__(self, pOther: 'DataFile'):
        isLess: bool = False

        if (self.season < pOther.season):
            isLess = True
        elif (self.season == pOther.season):
            if (self.comparable < pOther.comparable):
                isLess = True
            elif (self.comparable >= pOther.comparable):
                isLess = False
        else:
            isLess = False
        return isLess
    
    def __eq__(self, pOther: 'DataFile'):
        return self.season == pOther.season and self.comparable == pOther.comparable

class RawPlayerDataFile(DataFile):

    @staticmethod
    def parse(pFilename: str) -> "DataFile":
        pattern = r"^\.\/data\/raw\/player_stats\/\d{2}-\d{2}\/fpl_stats_\d{2}-\d{2}-\d{4}\.json$"
        regex = re.compile(pattern)
        if not regex.fullmatch(pFilename):
            raise ValueError(f"Invalid filename: '{pFilename}'")
        
        seasonPattern = r"\d{2}-\d{2}\/fpl_stats"
        seasonRegex = re.compile(seasonPattern)
        temp = seasonRegex.findall(pFilename)
        seasonStr = temp[0][0:5]
        seasonStart = int(seasonStr[0:2])
        seasonEnd = int(seasonStr[3:])
        season = Season(seasonStart, seasonEnd)

        datePattern = r"\d{2}-\d{2}-\d{4}"
        dateRegex = re.compile(datePattern)
        dateStr = dateRegex.findall(pFilename)[0]

        date = datetime.strptime(dateStr, r"%d-%m-%Y")
        return RawPlayerDataFile(season, pFilename, date)
    
    @staticmethod
    def parseOld(pFilename: str) -> DataFile:
        pattern = r"^data\/raw\/old_data\/players\/(\d{2})-(\d{2})\/fpl_stats_(\d{2})-(\d{2})-(\d{4})\.json$"
        regex = re.compile(pattern)
        if not regex.fullmatch(pFilename):
            raise ValueError(f"Invalid filename: '{pFilename}'")
        args = regex.findall(pFilename)[0]
        startYear = int(args[0])
        endYear = int(args[1])
        day = args[2]
        month = args[3]
        year = args[4]
        season = Season(startYear, endYear)
        dateStr = f"{day}-{month}-{year}"
        date = datetime.strptime(dateStr, r"%d-%m-%Y")

        return RawPlayerDataFile(season, pFilename, date)
    
class RawFixtureDataFile(DataFile):

    def __init__(self, pSeason: Season, pFilename: str, pGameweek: int):
        self.season = pSeason
        self.filename = pFilename
        self.gameweek = pGameweek
        self.comparable = self.gameweek
        ...

    @staticmethod
    def parse(pFilename: str) -> "DataFile":
        pattern = r"^\.\/data\/raw\/fixture_data\/\d{2}\/fixture_data_\d+\.json$"
        regex = re.compile(pattern)
        if not regex.fullmatch(pFilename):
            raise ValueError(f"Invalid filename: '{pFilename}'")
        
        seasonPattern = r"fixture_data\/\d{2}"
        seasonRegex = re.compile(seasonPattern)
        temp = seasonRegex.findall(pFilename)
        seasonStr = temp[0][-2:]
        seasonEnd = int(seasonStr)
        seasonStart = seasonEnd - 1
        season = Season(seasonStart, seasonEnd)

        gameweekPattern = r"\/fixture_data_\d+\.json$"
        gameweekRegex = re.compile(gameweekPattern)
        temp = (gameweekRegex.findall(pFilename))[0]
        # Get contents between "/" and before file extension
        fixtureDataPrefix = "/fixture_data_"
        gameweekStr = temp[len(fixtureDataPrefix):len(temp)-len(".json")]
        gameweek = int(gameweekStr)

        return RawFixtureDataFile(season, pFilename, gameweek)