import json
import pandas as pd
from modules.utils import lerp
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
            for val in gameweek:
                homeTeam = teamNames[str(val["team_h"])]
                awayTeam = teamNames[str(val["team_a"])]
                homeTeamDifficulty = self.calcSimpleDifficulty(homeTeam, awayTeam)
                awayTeamDifficulty = self.calcSimpleDifficulty(awayTeam, homeTeam)
                if(homeTeam in sums.keys()):
                    sums[homeTeam] += homeTeamDifficulty
                else:
                    sums[homeTeam] = homeTeamDifficulty

                if(awayTeam in sums.keys()):
                    sums[awayTeam] += awayTeamDifficulty
                else:
                    sums[awayTeam] = awayTeamDifficulty

        for team, sum in sums.items():
            self.simpleDifficulties[team] = sum / numFixtures
            self.normalisedDifficulties[team] = lerp(MIN_SCORE_OFFSET, MAX_SCORE_OFFSET, self.simpleDifficulties[team])

    def calcNormalisedDifficulty(self, pTeamA: str, pTeamB: str, pMin: float, pMax: float):
        simpleDifficulty = self.calcSimpleDifficulty(pTeamA, pTeamB)
        return lerp(pMin, pMax,simpleDifficulty)

    def calcSimpleDifficulty(self, pTeamA: str, pTeamB: str) -> float:
        teamAPosition = self.indexes[pTeamA]
        teamBPosition = self.indexes[pTeamB]
        simpleDifficulty = (teamAPosition - teamBPosition + 1) / 2
        return simpleDifficulty
        
    
    def getSimpleDifficulty(self, pTeam: str) -> float:
        return self.simpleDifficulties[pTeam]
    def getNormalisedDifficulty(self, pTeam: str) -> float:
        return self.normalisedDifficulties[pTeam]