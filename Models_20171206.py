
# coding: utf-8

from pandas import DataFrame, Series,read_csv
import statsmodels.formula.api as smf
from warnings import filterwarnings
filterwarnings("ignore")

analysis_home = '/Users/catcamacho/Box/LNCD_rewards_connectivity'
data = read_csv(analysis_home + '/doc/fullsample_means.csv', index_col=None)

independent_variables = ['age','ageInv','ageSq','Male']
dependent_variables = data.columns.values.tolist()[7:]
#remove hm from dv list
dependent_variables = [x for x in dependent_variables if 'hm_' not in x]

print('First the linear Age Models')
for a in dependent_variables:
    model = smf.mixedlm('%s ~ age + Male + age*Male' % (a), data, groups=data['timepoint'])
    fitmodel = model.fit()
    print(fitmodel.summary())
    aic = 
    print('AIC: ' + str(aic))

print('Next are the Inverse Age Models')

for a in dependent_variables:
    model = smf.mixedlm('%s ~ ageInv + Male + ageInv*Male' % (a), data, groups=data['timepoint'])
    fitmodel = model.fit()
    print(fitmodel.summary())
    print('AIC: ' + str(aic))

    
