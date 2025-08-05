import json
import pandas as pd
from copy import copy
from modules.utils import lerp
import math
from warnings import warn

import config

class FixtureDifficultyMatrix():
    def __init__(self,
                  pScale: float, 
                  pStartGameweek: int, 
                  pEndGameweek: int):
        
        with open("./data/current_table.txt") as f:
            self.table = f.readlines()
        self.table = [team.strip() for team in self.table]
        self.numTeams = len(self.table)
        self.allTeams = sorted(self.table)
        self.indexes = dict()
        for i, team in enumerate(self.table):
            self.indexes[team] = (i+1) / self.numTeams
        
        self.startGameweek = pStartGameweek
        self.endGameweek = pEndGameweek + 1

        self.thisGameweek = config.CURRENT_GAMEWEEK
        self.thisGameweekDiffs = dict()

        self.simpleDifficulties = dict()
        self.normalisedDifficulties = dict()
        self.precomputeFixtureDifficulty(pScale)

    def precomputeFixtureDifficulty(self, pScale: float):
        MIN_SCORE_OFFSET = -pScale
        MAX_SCORE_OFFSET = pScale

        # The range of gameweeks to get fixture data from
        fixtureRange = range(self.startGameweek, self.endGameweek+1)
        allFixtureDataRaw = []
        for gameweek in fixtureRange:
            with open(f"./data/fixture_data/fixture_data_{gameweek}.json") as f:
                allFixtureDataRaw.append(json.load(f))
        with open("./data/team_translation_table.json") as f:
            teamNames = json.load(f)

        numFixtures = len(allFixtureDataRaw)
        sums = dict()
        for gameweek in allFixtureDataRaw:

            alreadyPlayedTeams = set()
            currentGameweekDict = dict()
            currentGameweek = -1

            for val in gameweek:
                currentGameweek = val["event"]

                homeTeam = teamNames[str(val["team_h"])]
                awayTeam = teamNames[str(val["team_a"])]

                homeTeamDifficulty = self.calcSimpleDifficulty(homeTeam, awayTeam)
                awayTeamDifficulty = self.calcSimpleDifficulty(awayTeam, homeTeam)

                # If the home team has already played this week (i.e. if it is a double-gameweek for the home team),
                # then decrease the sum. We do this because it is statistically likely for players to earn more points during
                # a double-gameweek due to bonuses.

                if (homeTeam in alreadyPlayedTeams):
                    homeTeamDifficulty = -self.decayDifficulty(homeTeamDifficulty)

                if (awayTeam in alreadyPlayedTeams):
                    #print()
                    #print(f"{awayTeam} has double-gameweek in {val}.")
                    #print(f"Difficulty: {awayTeamDifficulty}")
                    awayTeamDifficulty = -self.decayDifficulty(awayTeamDifficulty)
                    #print(f"Sum: {sums[awayTeam]}")
                    #print(f"New average: {sums[awayTeam]/numFixtures}")

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

            if (currentGameweek == self.thisGameweek):
                self.thisGameweekDiffs = copy(currentGameweekDict)
                avg = self.averageDifficulty(self.thisGameweekDiffs)
                
                # Account for teams not having any games in a given gameweek
                for team in self.allTeams:
                    if team not in self.thisGameweekDiffs.keys():
                        print(f"Warning: {team} does not have any games in gameweek {gameweek}")
                        self.thisGameweekDiffs[team] = avg

        for team, sum in sums.items():
            self.simpleDifficulties[team] = sum / numFixtures
            self.normalisedDifficulties[team] = lerp(MIN_SCORE_OFFSET, MAX_SCORE_OFFSET, self.simpleDifficulties[team])
        jsonStr = json.dumps(self.simpleDifficulties,indent=4)
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
            warn(f"Team '{pTeam}' not found")
            return 0.5
            ...
        return self.simpleDifficulties[pTeam]
    def getNormalisedDifficulty(self, pTeam: str) -> float:
        return self.normalisedDifficulties[pTeam]
    def getCurrentDifficulty(self, pTeam: str) -> float:
        return self.thisGameweekDiffs[pTeam]