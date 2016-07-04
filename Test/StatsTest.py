import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas
import seaborn.apionly as sns
#iris = sns.load_dataset('iris')
#print(iris.head())
# Fit model and print summary
#rlm_model = smf.rlm(formula='sepal_length ~ sepal_width + petal_length + petal_width', data=iris, M=None)

data = sm.datasets.get_rdataset('epil', package='MASS').data

fam = sm.families.Poisson()
ind = sm.cov_struct.Exchangeable()
mod = smf.rlm("y ~ age + trt + base", data=data)
r = mod.fit()
print(r.summary())