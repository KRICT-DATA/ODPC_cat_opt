import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from matplotlib import pyplot as plt
import joblib
fig, axes = plt.subplots(1,2)
fig.set_size_inches((12, 4))
thickness = 2
fontdict = {'font': 'Times New Roman', 'size': 18, 'fontweight': 'bold'}
df = pd.read_excel('ODPC Initial Data.xlsx')
X = np.array(df.iloc[:, :18])
targets = ["Yield", "Score"]
t = 0
for target in targets:
    Y = np.array(df[target])
    if target == "Yield":
        model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features='log2', min_samples_leaf=2,
                                          min_samples_split=5, n_estimators=340, subsample=0.95)  ## Yield
    else:
        model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_features='log2', min_samples_leaf=2, min_samples_split=3, n_estimators=350, subsample=0.75)  ## Score
    cv = LeaveOneOut()
    pred = cross_val_predict(estimator=model, X=X, y=Y, cv=cv)
    r2 = r2_score(Y, pred)
    mae = np.mean(np.abs(pred - Y))
    ax1 = plt.subplot(1,2,t+1)
    ax1.spines['left'].set_linewidth(thickness)
    ax1.spines['right'].set_linewidth(thickness)
    ax1.spines['top'].set_linewidth(thickness)
    ax1.spines['bottom'].set_linewidth(thickness)
    ax1.tick_params(length=5, width=thickness)
    plt.scatter(x=Y, y=pred, c='blue', alpha=0.3)
    plt.plot([-0.03,0.3], [-0.03, 0.3], color='red', linestyle='--', linewidth=2)
    if target == "Yield":
        plt.xlim(-0.03, 0.20)
        plt.ylim(-0.03, 0.20)
        plt.text(0.10, 0.015, r'$\mathregular{R^2} = $', font='Times New Roman',
                 fontsize=18, fontweight='bold', style='italic')
        plt.text(0.135, 0.015, f'{r2: .2f}', font='Times New Roman', fontsize=18,
                 fontweight='bold')
        plt.text(0.10, -0.015, f'MAE = {mae: .3f}', **fontdict)
    else:
        plt.xlim(-0.03, 0.25)
        plt.ylim(-0.03, 0.25)
        plt.text(0.125, 0.017, r'$\mathregular{R^2} = $', font='Times New Roman',
                 fontsize=18, fontweight='bold', style='italic')
        plt.text(0.16, 0.017, f' {r2: .2f}', font='Times New Roman', fontsize=18,
                 fontweight='bold')
        plt.text(0.125, -0.015, f'MAE = {mae: .3f}', **fontdict)
    plt.xlabel(f"{target} (Exp.)", **fontdict)
    plt.ylabel(f"{target} (Predicted)", **fontdict)
    plt.xticks(**fontdict)
    plt.yticks(**fontdict)
    plt.tick_params(width=2)
    t += 1
    model.fit(X, Y)
    joblib.dump(model, f"GBR_{target}.sav")

plt.subplots_adjust(wspace=0.4)
plt.gcf().subplots_adjust(left=0.20, bottom=0.20)
plt.show()