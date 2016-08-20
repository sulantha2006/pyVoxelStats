import rpy2.robjects as r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas
pandas2ri.activate()
d = pandas.read_csv('~/Downloads/All_baseline_fdg_av45_data.csv', delimiter=',')
#d = r.DataFrame.from_csvfile('~/Downloads/All_baseline_fdg_av45_data.csv', header=True)

mgcv = importr('mgcv')
family=r.r('gaussian()')

gam1 = mgcv.gam(r.Formula("BETA12~s(AV45_MCI_t4_cent)+s(FDG_MCI_t4_cent)+PTGENDER+Age+APOE+PTEDUCAT"), family=family, data=d, method="REML")
sum = r.r.summary(gam1)
print(sum)
print(sum.names)

print(sum.rx2('p.table').names)
print(sum.rx2('s.table'))
print(sum.rx2('s.table'))
print(sum.rx2('s.table').rx(1, True)[2])
print('HI')
