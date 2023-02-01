import numpy as np
import pandas as pd
import joblib
from artificial_bee_colony import ABC
target = "Score"
predictor = joblib.load(f'GBR_{target}.sav')
df = pd.read_excel('ODPC Initial Data.xlsx')
nFeats = 18
X = np.array(df.iloc[:, :nFeats])
lbs = np.hstack((np.zeros([15]), [600], [600], [3]))
ubs = np.hstack((np.ones([15]), [600], [600], [3]))
nLimit = 6
nComp = 15
def pnt_func(x):
    pnt = 0
    for i in range(0, nComp):
        if x[0, i] < 0.0:
            pnt -= 1e+3
        if x[0, i] > 1.0:
            pnt -= 1e+3
    summation = 0
    for i in range(nComp):
        summation += x[0, i]
    if np.abs(summation - 1.0) > 0.0001:
        pnt -= 1e+3
    non_neg = 0
    for i in range(0, nComp):
        if np.abs(x[0, i]) > 0.0001:
            non_neg += 1
    if non_neg > (nLimit):
        pnt -= 1e+3
    return pnt
opt = ABC(nFeats, predictor.predict, lbs, ubs, opt_type='max', lim_trial=10, pnt_func=pnt_func)
filename = f"sol_GBR_{target}.csv"
for i in range(100):
    sol, val = opt.run(1000)
    import os
    if not os.path.isfile(filename):
        np.savetxt(filename, np.hstack((sol.reshape(1, -1), val.reshape(1,-1))), delimiter=',', fmt='%.3f', header=f'Al2O3, SiO2, ZrO2, Co, Cr, Fe, Ga, In, Mg, Mo, Ni, Pt, Sn, V, Zn, sint_temp, rxn_temp,rxn_flow(co2),{target}')
    else:
        with open(filename, 'a') as f:
            np.savetxt(f, np.hstack((sol.reshape(1, -1), val.reshape(1,-1))), delimiter=',', fmt='%.3f')
