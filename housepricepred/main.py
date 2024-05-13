"""
US CPI (inflation measure) -> CPIAUCSL.csv
rental vacancy rate, quarterly -> RRVRUSQ156N.csv
mortgage interest rates, weekly -> MORTGAGE30US.csv
median sale price for US houses -> Metro_median_sale_price_uc_sfrcondo_week.csv
Zillow home value index -> Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

fed_files = ["data/MORTGAGE30US.csv", "data/RRVRUSQ156N.csv", "data/CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]

fed_data = pd.concat(dfs, axis=1)

#data for every file is in different time periods, so we can use forward filling
fed_data = fed_data.ffill()
fed_data = fed_data.dropna()

zillow_files = ["data/Metro_median_sale_price_uc_sfrcondo_week.csv", "data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]
dfs = [pd.read_csv(f) for f in zillow_files]
#we choose the US region and we remove first 5 cols
dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs]
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")

price_data = dfs[0].merge(dfs[1], on="month")
price_data.index = dfs[0].index
price_data = price_data.drop(columns=["month"])
price_data.columns = ["price", "value"]

#aligning the zillow and the fed data to be concistent
#shifts federal reserve data forward 2 days to align with zillow data
fed_data.index = fed_data.index + timedelta(days=2)
price_data = fed_data.merge(price_data, left_index=True, right_index=True)
price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]
price_data = price_data.sort_index()
#we adjust the value and price because we dont want the model to also predict cpi
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100
price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data.dropna(inplace=True)
price_data["change"] = (price_data["next_quarter"]>price_data["adj_price"]).astype(int)
predictors = ["interest", "vacancy", "adj_price", "adj_value"]
target = "change"

#we make with scicitlearn
def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train[target])
    preds = rf.predict(test[predictors])
    return preds

#we have time state data, we cant do that because it wont work well.
#we will do backtesting
START = 260 #5 years worth of data
STEP = 52 #52 weeks a year

def backtest(data, predictors, target):
    all_preds=[]
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, predictors, target))

    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

yearly = price_data.rolling(52, min_periods=1).mean()
yearly_ratios = [p + "_year" for p in predictors]
price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]

preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)
print(accuracy)
#DIAGNOSTICS
#seeing where our model is making mistakes
pred_match = (preds == price_data[target].iloc[START:])
pred_match[pred_match==True] = "green"
pred_match[pred_match==False] = "red"
plot_data = price_data.iloc[START:].copy()
#plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
#plt.show()

from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(price_data[predictors], price_data[target])

result = permutation_importance(rf, price_data[predictors], price_data[target], n_repeats=10, random_state=1)
print(predictors)
print(result["importances_mean"])
#we see that the most important predictors are closer to 1
