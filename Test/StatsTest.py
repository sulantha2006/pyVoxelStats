import statsmodels.api as sm
import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset('epil', package='MASS').data

fam = sm.families.Poisson()
ind = sm.cov_struct.Exchangeable()
mod = smf.gee("y ~ age + trt + base", "subject", data,
              cov_struct=ind, family=fam)
res = mod.fit()
print(res.summary())
print(res)