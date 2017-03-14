from pyVS.Util.StatsUtil import StringModel

s = StringModel('A ~ C(D_F) + s(P_Q)', ['A', 'D_F'])
print(s)

