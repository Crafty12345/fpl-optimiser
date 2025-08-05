from dataclasses import dataclass
from datetime import datetime
import re
from abc import ABC as AbstractClass

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
    def parse(pFilename:str) -> 'DataFile': ...

class RawDataFile(DataFile):

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
        return RawDataFile(season, pFilename, date)

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

    def __gt__(self, pOther: 'DataFile'):
        isGreater: bool = False
        if (self.season < pOther.season):
            isGreater = False
        elif (self.season == pOther.season):
            if (self.date <= pOther.comparable):
                isGreater = False
            elif (self.date > pOther.comparable):
                isGreater = True
        elif (self.season > pOther.season):
            isGreater = True
        return