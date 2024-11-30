import json
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
        with open("./data/current_fixture_data.json") as f:
            fixtureDataRaw = json.load(f)
        with open("./data/team_translation_table.json") as f:
            teamNames = json.load(f)
        for val in fixtureDataRaw:
            homeTeam = teamNames[str(val["team_h"])]
            awayTeam = teamNames[str(val["team_a"])]
            homeTeamDifficulty = self.getDifficulty(homeTeam, awayTeam)
            awayTeamDifficulty = self.getDifficulty(awayTeam, homeTeam)
            self.currentFixtureDifficulties[homeTeam] = homeTeamDifficulty
            self.currentFixtureDifficulties[awayTeam] = awayTeamDifficulty

    def getDifficulty(self, pTeamA: str, pTeamB: str, pDifficultyScale: float = 1.0) -> float:
        teamAPosition = self.indexes[pTeamA]
        teamBPosition = self.indexes[pTeamB]
        normalisedDifficulty = (teamAPosition - teamBPosition + 1) / 2
        return normalisedDifficulty * pDifficultyScale
    
    def getFixtureDifficulty(self, pTeam: str) -> float:
        return self.currentFixtureDifficulties[pTeam]

FixtureDifficultyMatrix()