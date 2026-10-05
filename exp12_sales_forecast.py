"""Experiment 12 - ML for predicting future business trends
Dataset: Kaggle Store Sales - Time Series Forecasting (Favorita, Ecuador)
Target : total daily sales (all stores, all families); forecast horizon 16 days
Models : Linear Regression, Decision Tree, Random Forest, XGBoost (tuned)
Metrics: MAE, RMSE, R2 (+ MAPE)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
H=16                      # forecast horizon (days)  = Kaggle test window
TEST_DAYS=56              # hold-out for evaluation
plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"font.size":10})
C=["#2a78d6","#eb6834","#1baf7a","#8a63d2"]

# ---------- 1. load + preprocess ----------
tr=pd.read_csv("train.csv",parse_dates=["date"]); te=pd.read_csv("test.csv",parse_dates=["date"])
oil=pd.read_csv("oil.csv",parse_dates=["date"]); hol=pd.read_csv("holidays_events.csv",parse_dates=["date"])
stores=pd.read_csv("stores.csv")
print("train",tr.shape,"| test",te.shape,"| missing values:",int(tr.isna().sum().sum()))
daily=tr.groupby("date").agg(sales=("sales","sum"),promo=("onpromotion","sum"))
fut=te.groupby("date").agg(promo=("onpromotion","sum")); fut["sales"]=np.nan
alld=pd.concat([daily,fut])
full=pd.date_range(alld.index.min(),alld.index.max()); alld=alld.reindex(full)
alld["closed"]=alld.sales.isna()&(alld.index<=daily.index.max())      # 25-Dec gaps
alld.loc[alld.closed,["sales","promo"]]=0
alld=alld[alld.index>="2013-01-02"]                                   # 1-Jan: stores closed
oil=oil.set_index("date").reindex(alld.index).interpolate(limit_direction="both")
alld["oil"]=oil["dcoilwtico"]
nat=hol[(hol.locale=="National")&(~hol.transferred)&(hol.type.isin(["Holiday","Transfer","Additional","Bridge"]))]
alld["holiday"]=alld.index.isin(nat.date).astype(int)
print("daily rows",len(alld),alld.index.min().date(),"->",alld.index.max().date())

# ---------- 2. EDA ----------
d=alld[alld.index<=daily.index.max()]
fig,ax=plt.subplots(2,2,figsize=(13,8))
ax[0,0].plot(d.sales/1e3,color=C[0],lw=.7); ax[0,0].plot((d.sales.rolling(30).mean())/1e3,color=C[1],lw=1.6,label="30-day mean")
ax[0,0].set_title("Daily total sales (thousand $)"); ax[0,0].legend(frameon=False)
m=d.groupby(d.index.month).sales.mean()/1e3; ax[0,1].bar(m.index,m.values,color=C[0]); ax[0,1].set_title("Avg daily sales by month (seasonality)"); ax[0,1].set_xticks(range(1,13))
w=d.groupby(d.index.dayofweek).sales.mean()/1e3; ax[1,0].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],w.values,color=C[2]); ax[1,0].set_title("Avg daily sales by weekday")
fam=tr.groupby("family").sales.sum().sort_values().tail(8)/1e6; ax[1,1].barh(fam.index,fam.values,color=C[3]); ax[1,1].set_title("Top 8 product families (total sales, million)")
plt.tight_layout(); plt.savefig("eda_overview.png"); plt.close()
yr=d.groupby(d.index.year).sales.sum()/1e6
st=tr.merge(stores,on="store_nbr").groupby("type").sales.sum()/1e6
corr=d[["sales","promo","oil","holiday"]].corr()["sales"].round(3)
print("yearly (M):",yr.round(1).to_dict()); print("corr w/ sales:",corr.to_dict())
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].scatter(d.promo/1e3,d.sales/1e3,s=4,color=C[0],alpha=.5); ax[0].set_xlabel("items on promotion (thousand)"); ax[0].set_ylabel("sales (thousand $)"); ax[0].set_title("Promotion vs sales")
ax[1].scatter(d.oil,d.sales/1e3,s=4,color=C[1],alpha=.5); ax[1].set_xlabel("oil price ($)"); ax[1].set_title("Oil price vs sales")
plt.tight_layout(); plt.savefig("eda_drivers.png"); plt.close()

# ---------- 3. feature engineering ----------
f=alld.copy(); s=f.sales
f["dow"]=f.index.dayofweek; f["dom"]=f.index.day; f["month"]=f.index.month; f["year"]=f.index.year; f["doy"]=f.index.dayofyear
f["payday"]=((f.dom==15)|f.index.is_month_end).astype(int)
f["weekend"]=(f.dow>=5).astype(int)
f["xmas_wk"]=((f.month==12)&(f.dom>=20)).astype(int)
f["sin_doy"]=np.sin(2*np.pi*f.doy/365.25); f["cos_doy"]=np.cos(2*np.pi*f.doy/365.25)
for L in (16,21,28,35,364): f[f"lag_{L}"]=s.shift(L)           # lags >= horizon -> no leakage, direct 16-day forecast
f["roll7_l16"]=s.shift(H).rolling(7).mean(); f["roll28_l16"]=s.shift(H).rolling(28).mean()
f["lag_dow4"]=(s.shift(H).groupby(f.dow).transform(lambda x:x.rolling(4,min_periods=1).mean()))  # same-weekday avg
f=f[f.index>="2014-01-02"]
FEAT=[c for c in f.columns if c not in ("sales","closed")]
lab=f[f.index<=daily.index.max()]; fut=f[f.index>daily.index.max()]
cut=lab.index.max()-pd.Timedelta(days=TEST_DAYS-1)
Xtr,ytr=lab[lab.index<cut][FEAT],lab[lab.index<cut].sales; Xte,yte=lab[lab.index>=cut][FEAT],lab[lab.index>=cut].sales
print("features",len(FEAT),"| train",Xtr.shape,"| test",Xte.shape,cut.date())

# ---------- 4. models ----------
def met(y,p): return dict(MAE=mean_absolute_error(y,p),RMSE=mean_squared_error(y,p)**.5,R2=r2_score(y,p),MAPE=float(np.mean(np.abs((y-p)/y))*100))
tscv=TimeSeriesSplit(n_splits=4)
base={"Naive (lag-364)":None,
 "Linear Regression":make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
 "Decision Tree":DecisionTreeRegressor(max_depth=6,random_state=0),
 "Random Forest":RandomForestRegressor(n_estimators=300,max_depth=10,random_state=0,n_jobs=-1),
 "XGBoost":XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,random_state=0)}
rows=[];preds={}
for n,m in base.items():
    if m is None: p=Xte["lag_364"].values
    else: m.fit(Xtr,ytr); p=m.predict(Xte)
    preds[n]=p; rows.append({"Model":n,"Stage":"baseline",**met(yte,p)})
# tuning
grids={"Random Forest":(RandomForestRegressor(random_state=0,n_jobs=-1),{"n_estimators":[200,400],"max_depth":[6,10,14],"min_samples_leaf":[1,3]}),
 "XGBoost":(XGBRegressor(random_state=0),{"n_estimators":[200,400],"max_depth":[3,4,6],"learning_rate":[.03,.07],"subsample":[.8,1.0]}),
 "Decision Tree":(DecisionTreeRegressor(random_state=0),{"max_depth":[3,5,7,10],"min_samples_leaf":[1,5,10]}),
 "Linear Regression":(make_pipeline(StandardScaler(),Ridge()),{"ridge__alpha":[.01,.1,1,10,100]})}
best={};bp=[]
for n,(est,g) in grids.items():
    gs=GridSearchCV(est,g,cv=tscv,scoring="neg_mean_absolute_error",n_jobs=-1).fit(Xtr,ytr)
    best[n]=gs.best_estimator_; p=gs.best_estimator_.predict(Xte); preds[n+" (tuned)"]=p
    rows.append({"Model":n+" (tuned)","Stage":"tuned",**met(yte,p)}); bp.append({"Model":n,"Best params":str(gs.best_params_),"CV MAE":-gs.best_score_})
    print(n,gs.best_params_,round(-gs.best_score_))
res=pd.DataFrame(rows); print(res.round(3).to_string())
tuned=res[res.Stage=="tuned"].sort_values("RMSE"); champ=tuned.iloc[0].Model.replace(" (tuned)",""); print("champion:",champ)

# ---------- 5. plots ----------
fig,ax=plt.subplots(1,3,figsize=(14,4))
for i,mt in enumerate(["MAE","RMSE","R2"]):
    r=res.set_index("Model")[mt]; ax[i].barh(r.index,r.values,color=[C[0] if "tuned" in k else "#9aa0a6" for k in r.index]); ax[i].set_title(mt); ax[i].invert_yaxis()
    if i: ax[i].set_yticklabels([])
plt.tight_layout(); plt.savefig("model_comparison.png"); plt.close()
fig,ax=plt.subplots(figsize=(12,4.5)); ax.plot(yte.index,yte/1e3,color="#222",lw=2,label="Actual")
for k,c in zip([champ+" (tuned)","Linear Regression (tuned)","Random Forest (tuned)"],C):
    if k in preds: ax.plot(yte.index,preds[k]/1e3,color=c,lw=1.3,label=k)
ax.set_title("Hold-out: actual vs predicted daily sales (thousand $)"); ax.legend(frameon=False,ncol=2); plt.tight_layout(); plt.savefig("actual_vs_pred.png"); plt.close()
fi=pd.Series(best["XGBoost"].feature_importances_,FEAT).sort_values().tail(12)
fig,ax=plt.subplots(figsize=(7,4.5)); ax.barh(fi.index,fi.values,color=C[0]); ax.set_title("XGBoost feature importance (top 12)"); plt.tight_layout(); plt.savefig("feature_importance.png"); plt.close()

# ---------- 6. final fit on all labelled data + 16-day forecast ----------
fin=best[champ]; fin.fit(lab[FEAT],lab.sales); fc=pd.Series(fin.predict(fut[FEAT]),fut.index,name="Forecast")
print(fc.round(0).to_string())
fig,ax=plt.subplots(figsize=(12,4.5)); hist=lab.sales[-120:]; ax.plot(hist.index,hist/1e3,color="#222",lw=1.5,label="History")
ax.plot(fc.index,fc/1e3,color=C[1],lw=2,marker="o",ms=3,label=f"Forecast 16 days ({champ})"); ax.axvline(lab.index.max(),color="#888",ls="--")
ax.set_title("Future sales forecast: 16-Aug to 31-Aug-2017 (thousand $)"); ax.legend(frameon=False); plt.tight_layout(); plt.savefig("forecast.png"); plt.close()

# ---------- 7. excel ----------
overview=pd.DataFrame({"Item":["Dataset","Rows (train.csv)","Daily rows used","Date range","Target","Horizon","Hold-out","Features","Champion model","Hold-out MAE","Hold-out RMSE","Hold-out R2","Hold-out MAPE %"],
 "Value":["Kaggle Store Sales - Time Series Forecasting",len(tr),len(alld),f"{alld.index.min().date()} to {daily.index.max().date()}","Total daily sales (all stores, all families)",f"{H} days (16-Aug to 31-Aug-2017)",f"last {TEST_DAYS} days ({cut.date()} to {lab.index.max().date()})",len(FEAT),champ+" (tuned)"]+[round(float(tuned.iloc[0][k]),3) for k in ["MAE","RMSE","R2","MAPE"]]})
with pd.ExcelWriter("Experiment_12_Sales_Forecast_Results.xlsx") as xw:
    overview.to_excel(xw,sheet_name="Summary",index=False); res.round(4).to_excel(xw,sheet_name="Model comparison",index=False)
    pd.DataFrame(bp).to_excel(xw,sheet_name="Tuning",index=False)
    pd.DataFrame({"Date":yte.index,"Actual":yte.values,**{k:v for k,v in preds.items()}}).to_excel(xw,sheet_name="Holdout predictions",index=False)
    fc.reset_index().rename(columns={"index":"Date"}).to_excel(xw,sheet_name="Forecast",index=False)
    fi[::-1].reset_index().rename(columns={"index":"Feature",0:"Importance"}).to_excel(xw,sheet_name="Feature importance",index=False)
    pd.DataFrame({"Year":yr.index,"Total sales (M)":yr.values}).to_excel(xw,sheet_name="Yearly sales",index=False)
    for sh in xw.sheets.values(): sh.set_column(0,8,22)
print("done")
