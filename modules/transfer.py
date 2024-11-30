from modules.player import Player

class Transfer():
    def __init__(self, pOldPlayer: Player, pNewPlayer: Player):
        self.oldPlayer = pOldPlayer
        self.newPlayer = pNewPlayer
        self.scoreDif = self.newPlayer.getScore() - self.oldPlayer.getScore()
        self.costDif = self.newPlayer.getCost() - self.oldPlayer.getCost()
        self.position = self.oldPlayer.getPosition()
        pass

    def getOldPlayer(self): return self.oldPlayer
    def getNewPlayer(self): return self.newPlayer
    def getScoreDif(self): return self.scoreDif
    def getCostDif(self): return self.costDif
    def getPosition(self): return self.position
    
    def __gt__(self, pOther):
        return self.scoreDif > pOther.getScoreDif()
    
    def __str__(self):
        string = f"Transfer from {self.oldPlayer.getName()} -> {self.newPlayer.getName()}:\n"
        string += f"Old player: {self.oldPlayer}\n"
        string += f"New player: {self.newPlayer}\n"
        string += f"- Cost change: {self.costDif}\n"
        string += f"- Score change: {round(self.scoreDif,2)}"
        return string
