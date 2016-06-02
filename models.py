from StatsModel import StatsModel

class LM(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'lm')

class GLM(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'glm')

class LME(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'lme')

class GLME(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'glme')