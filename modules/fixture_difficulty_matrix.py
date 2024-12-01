import json
from modules.utils import lerp

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
        
        self.currentFixtureDifficulties = dict()
        self.precomputeFixtureDifficulty()
    def precomputeFixtureDifficulty(self):

        MIN_SCORE_OFFSET = -10.0
        MAX_SCORE_OFFSET = 10.0

        # The range of gameweeks to get fixture data from
        fixtureRange = range(13, 15+1)
        allFixtureDataRaw = []
        for gameweek in fixtureRange:
            with open(f"./data/fixture_data/fixture_data_{gameweek}.json") as f:
                print(gameweek)
                allFixtureDataRaw.append(json.load(f))
        with open("./data/team_translation_table.json") as f:
            teamNames = json.load(f)

        numFixtures = len(allFixtureDataRaw)
        sums = dict()
        for gameweek in allFixtureDataRaw:
            for val in gameweek:
                homeTeam = teamNames[str(val["team_h"])]
                awayTeam = teamNames[str(val["team_a"])]
                homeTeamDifficulty = self.getDifficulty(homeTeam, awayTeam, MIN_SCORE_OFFSET, MAX_SCORE_OFFSET)
                awayTeamDifficulty = self.getDifficulty(awayTeam, homeTeam, MIN_SCORE_OFFSET, MAX_SCORE_OFFSET)
                if(homeTeam in sums.keys()):
                    sums[homeTeam] += homeTeamDifficulty
                else:
                    sums[homeTeam] = homeTeamDifficulty

                if(awayTeam in sums.keys()):
                    sums[awayTeam] += awayTeamDifficulty
                else:
                    sums[awayTeam] = awayTeamDifficulty

        for team, sum in sums.items():
            self.currentFixtureDifficulties[team] = sum / numFixtures

    def getDifficulty(self, pTeamA: str, pTeamB: str, pMin: float, pMax: float) -> float:
        teamAPosition = self.indexes[pTeamA]
        teamBPosition = self.indexes[pTeamB]
        normalisedDifficulty = (teamAPosition - teamBPosition + 1) / 2

        return lerp(pMin, pMax,normalisedDifficulty)
    
    def getFixtureDifficulty(self, pTeam: str) -> float:
        return self.currentFixtureDifficulties[pTeam]