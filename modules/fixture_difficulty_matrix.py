import json
import pandas as pd
from copy import copy, deepcopy
from modules.utils import lerp
import math
from warnings import warn
import os

import config
# TODO: Have FixtureDifficultyMatrix take into account how well a team is doing NOW (sliding window??)
class FixtureDifficultyMatrix():
    class _Memoise():
        def __init__(self, pFunc):
            self.func = pFunc
            self.memo = dict()

        def __call__(self,
                  pBehindSteps: int,
                  pCurrentGameweek: int,
                  pAheadSteps: int,
                  pSeason: int,
                  pScale: float):
            
            if (pSeason in self.memo.keys()):
                if (pCurrentGameweek in self.memo[pSeason].keys()):
                    dictList = self.memo[pSeason][pCurrentGameweek]
                    for tempDict in dictList:
                        otherBehindSteps = tempDict["behind_steps"]
                        otherAheadSteps = tempDict["ahead_steps"]
                        otherScale = tempDict["scale"]
                        if (pBehindSteps == otherBehindSteps) and \
                            (pAheadSteps == otherAheadSteps) and \
                            (pScale == otherScale):
                            return tempDict["data"]
            result = self.func()

    def __init__(self):
        
        with open("./data/current_table.txt", "r") as f:
            self.txtTable = f.readlines()
        with open(f"./data/team_points.json", "r") as f:
            teamPoints = json.load(f)

        self.txtTable: list[str] = [line.strip() for line in self.txtTable]

        seasonPoints: dict[str, int] = teamPoints.get(str(config.CURRENT_SEASON), dict())
        start = max(config.CURRENT_GAMEWEEK-config.TEAM_POINT_SAMPLE_SIZE, 1)
        totals: dict[str, int] = dict()
        if (len(seasonPoints) > 0):
            for i in range(start, config.CURRENT_GAMEWEEK):
                for k, v in seasonPoints[str(i)].items():
                    teamTotal = totals.get(k, 0)
                    teamTotal += v
                    totals[k] = teamTotal
            totals["UNK"] = 9999
            # Add defaults for teams that have been relegated
            for team in self.txtTable:
                if team not in totals.keys():
                    totals[team] = -9999
            self.table = sorted(totals.items(), key=lambda x: x[1])[::-1]
            self.table = [x[0] for x in self.table]
        else:
            self.table = [team.strip() for team in self.table]
        #print(max(teamPoints.keys()))
        # Add 1 because unknown team
        self.numTeams = config.NUM_TEAMS + 1
        self.allTeams = sorted(self.table)
        self.indexes = dict()
        for i, team in enumerate(self.table):
            self.indexes[team] = min(i+1, self.numTeams) / self.numTeams
        print("Team rankings:")
        print(self.indexes)

        self.thisGameweekDiffs = dict()
        self.fixtureDataExists = True

        self.simpleDifficulties = dict()
        self.normalisedDifficulties = dict()

        with open("./data/team_translation_table.json") as f:
            teamNamesJson = json.load(f)
        self.teamTranslationDict: dict[int, pd.DataFrame] = {int(k): pd.DataFrame.from_records(v) for k, v in teamNamesJson.items()}
        fixtureDataPath = "./data/fixtures.json"
        # TODO: Implement data/team_points.json
        with open(fixtureDataPath, "r") as f:
            self.fixturesJson: dict[str, dict[str, list[dict]]] = json.load(f)

        self.cache = dict()

    def _findCache(self,
                  pBehindSteps: int,
                  pCurrentGameweek: int,
                  pAheadSteps: int,
                  pSeason: int,
                  pScale: float) -> dict | None:
        
        if (pSeason in self.cache.keys()):
            if (pCurrentGameweek in self.cache[pSeason].keys()):
                dictList = self.cache[pSeason][pCurrentGameweek]
                for tempDict in dictList:
                    otherBehindSteps = tempDict["behind_steps"]
                    otherAheadSteps = tempDict["ahead_steps"]
                    otherScale = tempDict["scale"]
                    if (pBehindSteps == otherBehindSteps) and \
                        (pAheadSteps == otherAheadSteps) and \
                        (pScale == otherScale):
                        return tempDict
        return None

    def precomputeFixtureDifficulty(self,
                  pBehindSteps: int,
                  pCurrentGameweek: int,
                  pAheadSteps: int,
                  pSeason: int,
                  pScale: float):
        
        foundCache = self._findCache(pBehindSteps, pCurrentGameweek, pAheadSteps, pSeason, pScale)
        if (foundCache is not None):
            self.thisGameweekDiffs = foundCache["this_gw_diffs"]
            self.simpleDifficulties = foundCache["simple_diffs"]
            self.normalisedDifficulties = foundCache["normalised_diffs"]
            return

        MIN_SCORE_OFFSET = -pScale
        MAX_SCORE_OFFSET = pScale
        if pCurrentGameweek < 0 or pCurrentGameweek > config.MAX_GAMEWEEKS:
            raise ValueError("Invalid current gameweek")

        startGameweek = max(pCurrentGameweek - pBehindSteps, 0)
        endGameweek = min(pCurrentGameweek + pAheadSteps, config.MAX_GAMEWEEKS)

        allFixtureDataRaw = []

        numFixtures = len(allFixtureDataRaw)
        sums = dict()
        for _season in self.fixturesJson.keys():
            season = int(_season)

            for _gameweek, weekData in self.fixturesJson[_season].items():
                alreadyPlayedTeams = set()
                currentGameweekDict = dict()
                gameweek = int(_gameweek)
                if (season == pSeason and
                    (gameweek >= startGameweek and gameweek <= endGameweek)):
                    for fixture in weekData:
                        homeTeam = fixture["home_team"]
                        awayTeam = fixture["away_team"]

                        homeTeamDifficulty = self.calcSimpleDifficulty(homeTeam, awayTeam)
                        awayTeamDifficulty = self.calcSimpleDifficulty(awayTeam, homeTeam)

                        # If the home team has already played this week (i.e. if it is a double-gameweek for the home team),
                        # then decrease the sum. We do this because it is statistically likely for players to earn more points during
                        # a double-gameweek due to bonuses.

                        if (homeTeam in alreadyPlayedTeams):
                            homeTeamDifficulty = -self.decayDifficulty(homeTeamDifficulty)
                        if (awayTeam in alreadyPlayedTeams):
                            awayTeamDifficulty = -self.decayDifficulty(awayTeamDifficulty)

                        if(homeTeam not in sums.keys()):
                            sums[homeTeam] = 0
                        if(awayTeam not in sums.keys()):
                            sums[awayTeam] = 0

                        sums[homeTeam] += homeTeamDifficulty
                        sums[awayTeam] += awayTeamDifficulty

                        if(homeTeam not in currentGameweekDict.keys()):
                            currentGameweekDict[homeTeam] = 0
                        if (awayTeam not in currentGameweekDict.keys()):
                            currentGameweekDict[awayTeam] = 0
                        currentGameweekDict[homeTeam] += homeTeamDifficulty
                        currentGameweekDict[awayTeam] += awayTeamDifficulty

                        alreadyPlayedTeams.add(homeTeam)
                        alreadyPlayedTeams.add(awayTeam)

                    if (gameweek == pCurrentGameweek):
                        self.thisGameweekDiffs = copy(currentGameweekDict)
                        avg = self.averageDifficulty(self.thisGameweekDiffs)

                        for team in self.allTeams:
                            if team not in self.thisGameweekDiffs.keys():
                                if (team in self.teamTranslationDict[season]["name"]):
                                    print(f"Warning: {team} does not have any games in gameweek {gameweek} of season {season}")
                                    self.thisGameweekDiffs[team] = avg

        for team, sum in sums.items():
            # simpleDifficulties = average of how badly a team will do in relation to another team
            if (numFixtures == 0):
                # Set an arbitrary "average" value of 0.5
                self.simpleDifficulties[team] = 0.5
            else:
                self.simpleDifficulties[team] = sum / numFixtures
            self.normalisedDifficulties[team] = lerp(MIN_SCORE_OFFSET, MAX_SCORE_OFFSET, self.simpleDifficulties[team])

        if pSeason not in self.cache.keys():
            self.cache[pSeason] = dict()
        
        if (pCurrentGameweek not in self.cache[pSeason]):
            self.cache[pSeason][pCurrentGameweek] = list()
        
        toAdd = dict()
        toAdd["behind_steps"] = pBehindSteps
        toAdd["ahead_steps"] = pAheadSteps
        toAdd["scale"] = pScale
        toAdd["this_gw_diffs"] = deepcopy(self.thisGameweekDiffs)
        toAdd["simple_diffs"] = deepcopy(self.simpleDifficulties)
        toAdd["normalised_diffs"] = deepcopy(self.normalisedDifficulties)

        self.cache[pSeason][pCurrentGameweek].append(toAdd)

    def averageDifficulty(self, pDict: dict) -> float:
        total = sum(pDict.values())
        return total / len(pDict)

    def decayDifficulty(self, pNewVal):
        multiplier = 3.5
        return math.exp(-(multiplier*pNewVal))

    def calcNormalisedDifficulty(self, pTeamA: str, pTeamB: str, pMin: float, pMax: float):
        if ("," in pTeamB):
            simpleDifficulty = self.calcSimpleDifficulty(pTeamA, pTeamB.split(","))
        else: 
            simpleDifficulty = self.calcSimpleDifficulty(pTeamA, pTeamB)
        return lerp(pMin, pMax,simpleDifficulty)

    def calcSimpleDifficulty(self, pTeamA: str, pTeamB: str | list[str]) -> float:
        teamAPosition = self.indexes[pTeamA]
        # Factor in double-gameweeks
        if (type(pTeamB) is type(list())):
            total = 0
            numTeams = len(pTeamB)
            # Artificially lower the difficulty in the case of double-gameweeks
            for (i, team) in enumerate(pTeamB):
                weight = (numTeams - i) / numTeams
                total += self.indexes[team] * weight
            teamBPosition = total / numTeams
        elif (type(pTeamB) is type(str())):
            teamBPosition = self.indexes[pTeamB]
        else:
            raise AssertionError()
        simpleDifficulty = (teamAPosition - teamBPosition + 1) / 2
        return simpleDifficulty
        
    def getSimpleDifficulty(self, pTeam: str) -> float:
        if(pTeam not in self.simpleDifficulties.keys()):
            # Avoid warning twice for the same issue
            if (self.fixtureDataExists):
                warn(f"Team '{pTeam}' not found in season {self.season}")
            return 0.5

        return self.simpleDifficulties[pTeam]
    def getCurrentDifficulty(self, pTeam: str) -> float:
        return self.thisGameweekDiffs[pTeam]