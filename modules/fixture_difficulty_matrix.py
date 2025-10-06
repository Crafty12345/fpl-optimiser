import json
import pandas as pd
from copy import copy
from modules.utils import lerp
import math
from warnings import warn
import os

import config

class FixtureDifficultyMatrix():
    def __init__(self):
        
        with open("./data/current_table.txt") as f:
            self.table = f.readlines()
        self.table = [team.strip() for team in self.table]
        self.numTeams = len(self.table)
        self.allTeams = sorted(self.table)
        self.indexes = dict()
        for i, team in enumerate(self.table):
            self.indexes[team] = (i+1) / self.numTeams

        self.thisGameweekDiffs = dict()
        self.fixtureDataExists = True

        self.simpleDifficulties = dict()
        self.normalisedDifficulties = dict()

    def precomputeFixtureDifficulty(self,
                  pBehindSteps: int,
                  pCurrentGameweek: int,
                  pAheadSteps: int,
                  pSeason: int,
                  pScale: float):
        
        MIN_SCORE_OFFSET = -pScale
        MAX_SCORE_OFFSET = pScale
        if pCurrentGameweek < 0 or pCurrentGameweek > config.MAX_GAMEWEEKS:
            raise ValueError("Invalid current gameweek")

        startGameweek = max(pCurrentGameweek - pBehindSteps, 0)
        endGameweek = min(pCurrentGameweek + pAheadSteps, config.MAX_GAMEWEEKS)

        allFixtureDataRaw = []
        with open("./data/team_translation_table.json") as f:
            teamNamesJson = json.load(f)[str(pSeason)]
        teamNameDf: pd.DataFrame = pd.DataFrame.from_records(teamNamesJson)


        fixtureDataPath = "./data/fixtures.json"
        with open(fixtureDataPath, "r") as f:
            fixturesJson: dict[str, dict[str, list[dict]]] = json.load(f)

        numFixtures = len(allFixtureDataRaw)
        sums = dict()
        for _season in fixturesJson.keys():
            season = int(_season)

            for _gameweek, weekData in fixturesJson[_season].items():
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
        #print(jsonStr)

    def averageDifficulty(self, pDict: dict) -> float:
        total = sum(pDict.values())
        return total / len(pDict)

    def decayDifficulty(self, pNewVal):
        multiplier = 3.5
        return math.exp(-(multiplier*pNewVal))

    def calcNormalisedDifficulty(self, pTeamA: str, pTeamB: str, pMin: float, pMax: float):
        simpleDifficulty = self.calcSimpleDifficulty(pTeamA, pTeamB)
        return lerp(pMin, pMax,simpleDifficulty)

    def calcSimpleDifficulty(self, pTeamA: str, pTeamB: str) -> float:
        teamAPosition = self.indexes[pTeamA]
        teamBPosition = self.indexes[pTeamB]
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