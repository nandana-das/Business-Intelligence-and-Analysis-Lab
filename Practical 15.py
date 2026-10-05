"""Experiment 15 - Emerging BI technologies for a future retail scenario (UCI Online Retail)
Traditional BI KPIs -> AI-driven analytics (RFM + K-Means, key influencers, anomaly detection, forecast)
-> simulated real-time feed (Live_Sales.csv) -> cloud/immersive notes -> comparison + ROI tables"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"font.size":10})
BL,OR,GR,PU="#2a78d6","#eb6834","#1baf7a","#8a63d2"
raw=pd.read_csv("data.csv",encoding="latin1"); n0=len(raw)
raw["InvoiceDate"]=pd.to_datetime(raw.InvoiceDate,format="%m/%d/%Y %H:%M")
log={"Rows loaded":n0,"Missing CustomerID":int(raw.CustomerID.isna().sum()),"Missing Description":int(raw.Description.isna().sum()),
     "Cancelled invoices (C...)":int(raw.InvoiceNo.str.startswith("C").sum()),"Quantity <= 0":int((raw.Quantity<=0).sum()),"UnitPrice <= 0":int((raw.UnitPrice<=0).sum()),"Duplicate rows":int(raw.duplicated().sum())}
d=raw[~raw.InvoiceNo.str.startswith("C")&(raw.Quantity>0)&(raw.UnitPrice>0)].drop_duplicates().copy()
d=d[~d.StockCode.str.match(r"^(POST|D|M|DOT|BANK CHARGES|AMAZONFEE|S|CRUK|B)$",na=False)]
d["Description"]=d.Description.fillna("UNKNOWN").str.strip()
d["Sales"]=d.Quantity*d.UnitPrice; d["Date"]=d.InvoiceDate.dt.normalize(); d["Month"]=d.InvoiceDate.dt.to_period("M").astype(str)
log["Rows after cleaning (sales analysis)"]=len(d); print(log)
cust=d.dropna(subset=["CustomerID"]).copy(); cust["CustomerID"]=cust.CustomerID.astype(int).astype(str); log["Rows with CustomerID (customer analysis)"]=len(cust)

# ---------- KPIs ----------
K={"Total Sales":d.Sales.sum(),"Total Orders":d.InvoiceNo.nunique(),"Total Customers":cust.CustomerID.nunique(),"Total Quantity":int(d.Quantity.sum())}
K["Avg Order Value"]=K["Total Sales"]/K["Total Orders"]; print({k:round(v,1) for k,v in K.items()})
mon=d.groupby("Month").Sales.sum(); ctry=d.groupby("Country").Sales.sum().sort_values(ascending=False); prod=d.groupby("Description").agg(Sales=("Sales","sum"),Qty=("Quantity","sum")).sort_values("Sales",ascending=False)
uk=ctry["United Kingdom"]/ctry.sum()*100; print("UK share %",round(uk,1),"| top5 countries",ctry.head(5).round(0).to_dict()); print(prod.head(5).round(0))
mon_p=mon.copy(); mon_p.index=pd.PeriodIndex(mon_p.index,freq="M")
fig,ax=plt.subplots(2,2,figsize=(14,8.5))
ax[0,0].plot(mon.index,mon.values/1e3,color=BL,marker="o"); ax[0,0].set_title("Monthly sales (thousand £) – Dec-2011 partial (to 9-Dec)"); plt.setp(ax[0,0].get_xticklabels(),rotation=60,ha="right")
c=ctry.drop("United Kingdom").head(8)[::-1]/1e3; ax[0,1].barh(c.index,c.values,color=GR); ax[0,1].set_title("Top 8 export markets (thousand £, excl. UK)")
p=prod.head(10)[::-1]/1e3; ax[1,0].barh([s[:28] for s in p.index],p.Sales,color=OR); ax[1,0].set_title("Top 10 products by sales (thousand £)")
hr=d.groupby(d.InvoiceDate.dt.hour).Sales.sum()/1e3; ax[1,1].bar(hr.index,hr.values,color=PU); ax[1,1].set_title("Sales by hour of day (thousand £)")
plt.tight_layout(); plt.savefig("bi_overview.png"); plt.close()

# ---------- AI 1: RFM + K-Means ----------
snap=cust.InvoiceDate.max().normalize()+pd.Timedelta(days=1)
rfm=cust.groupby("CustomerID").agg(Recency=("InvoiceDate",lambda x:(snap-x.max()).days),Frequency=("InvoiceNo","nunique"),Monetary=("Sales","sum"))
Z=StandardScaler().fit_transform(np.log1p(rfm)); inert=[];sil=[]
from sklearn.metrics import silhouette_score
for k in range(2,8):
    km=KMeans(k,n_init=10,random_state=0).fit(Z); inert.append(km.inertia_); sil.append(silhouette_score(Z,km.labels_,sample_size=3000,random_state=0))
km=KMeans(4,n_init=10,random_state=0).fit(Z); rfm["Cluster"]=km.labels_
prof=rfm.groupby("Cluster").agg(Customers=("Recency","size"),Recency=("Recency","mean"),Frequency=("Frequency","mean"),Monetary=("Monetary","mean"),Revenue=("Monetary","sum"))
order=prof.sort_values("Monetary",ascending=False).index.tolist(); names={}
names[order[0]]="Champions"; names[order[-1]]="Lost / At risk"
mid=[c for c in order[1:-1]]; mid=sorted(mid,key=lambda c:prof.Recency[c]); names[mid[0]]="New / Recent (small)"; names[mid[1]]="Potential / Cooling"
rfm["Segment"]=rfm.Cluster.map(names); prof["Segment"]=prof.index.map(names); prof["Revenue %"]=prof.Revenue/prof.Revenue.sum()*100; prof=prof.sort_values("Monetary",ascending=False)
print(prof.round(1)); print("silhouette k=4:",round(sil[2],3))
fig,ax=plt.subplots(1,3,figsize=(16,4.6)); col={"Champions":GR,"New / Recent (small)":BL,"Potential / Cooling":"#e0a800","Lost / At risk":OR}
for s,g in rfm.groupby("Segment"): ax[0].scatter(g.Frequency,g.Monetary,s=6,color=col[s],label=s,alpha=.6)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Frequency (orders)"); ax[0].set_ylabel("Monetary (£)"); ax[0].set_title("Customer segments (K-Means on log-RFM)"); ax[0].legend(frameon=False,markerscale=3)
ax[1].plot(range(2,8),inert,marker="o",color=BL); ax[1].set_title("Elbow (inertia) – k=4 chosen"); ax[1].set_xlabel("k")
pr=prof.set_index("Segment"); ax[2].bar(pr.index,pr["Revenue %"],color=[col[s] for s in pr.index]); ax[2].set_title("Share of revenue by segment (%)"); plt.setp(ax[2].get_xticklabels(),rotation=12)
for i,v in enumerate(pr["Revenue %"]): ax[2].text(i,v+1,f"{v:.0f}%",ha="center")
plt.tight_layout(); plt.savefig("rfm_segments.png"); plt.close()

# ---------- AI 2: key influencers of ORDER VALUE (random forest, invoice level) ----------
o=d.groupby("InvoiceNo").agg(value=("Sales","sum"),lines=("StockCode","nunique"),avg_price=("UnitPrice","mean"),dt=("InvoiceDate","first"),ctry=("Country","first"),cust=("CustomerID",lambda x:x.notna().all()))
o["month"]=o.dt.dt.month; o["hour"]=o.dt.dt.hour; o["dow"]=o.dt.dt.dayofweek; o["is_uk"]=(o.ctry=="United Kingdom").astype(int); o["known_customer"]=o.cust.astype(int)
F=["lines","avg_price","month","hour","dow","is_uk","known_customer"]
rf=RandomForestRegressor(200,max_depth=10,min_samples_leaf=10,random_state=0,n_jobs=2).fit(o[F],np.log1p(o.value)); imp=pd.Series(rf.feature_importances_,F).sort_values()
imp.index=imp.index.map({"lines":"Distinct products in order","avg_price":"Average unit price","month":"Month","hour":"Hour of day","dow":"Day of week","is_uk":"UK customer","known_customer":"Registered customer"})
print("influencers",imp.round(3).to_dict())
fig,ax=plt.subplots(figsize=(7,4)); ax.barh(imp.index,imp.values,color=BL); ax.set_title("Key influencers of order value (random forest)"); plt.tight_layout(); plt.savefig("key_influencers.png"); plt.close()

# ---------- AI 3: anomaly detection on daily sales ----------
day=d.groupby("Date").Sales.sum().asfreq("D")
dd=day.dropna(); roll=dd.rolling(14,center=True,min_periods=7).median(); mad=(dd-roll).abs().rolling(28,center=True,min_periods=10).median()*1.4826
z=(dd-roll)/mad; an=dd[z.abs()>3.5]; print("anomalies",len(an)); 
iso=IsolationForest(contamination=.03,random_state=0).fit(np.c_[dd.values,dd.index.dayofweek]); an_iso=dd[iso.predict(np.c_[dd.values,dd.index.dayofweek])==-1]
fig,ax=plt.subplots(figsize=(12,4.2)); ax.plot(dd.index,dd/1e3,color=BL,lw=1); ax.scatter(an.index,an/1e3,color=OR,s=28,zorder=3,label=f"Anomaly (robust z > 3.5): {len(an)}"); ax.set_title("Daily sales (thousand £) with detected anomalies"); ax.legend(frameon=False); plt.tight_layout(); plt.savefig("anomalies.png"); plt.close()
an_tbl=pd.DataFrame({"Date":an.index.date,"Sales":an.values.round(0),"Robust z":z[an.index].round(2).values,"Type":np.where(an.values>roll[an.index].values,"Spike","Drop")})

# ---------- AI 4: forecast 30 days (Holt-Winters, weekly seasonality) ----------
ser=dd.copy(); ser=ser.reindex(pd.date_range(ser.index.min(),ser.index.max())).fillna(0)   # closed days = 0 (no Saturday trading)
wk=ser.resample("W").sum().iloc[:-1]                                                      # weekly totals (drop partial last week)
test=8; tr,te=wk.iloc[:-test],wk.iloc[-test:]
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
mae=lambda a,b:float(np.mean(np.abs(a-b))); mape=lambda a,b:float(np.mean(np.abs((a-b)/a))*100)
cand={"Holt damped trend":lambda y,h:ExponentialSmoothing(y,trend="add",damped_trend=True).fit().forecast(h),
      "Simple exp. smoothing":lambda y,h:SimpleExpSmoothing(y,initialization_method="estimated").fit().forecast(h),
      "Naive (last-4-week mean)":lambda y,h:pd.Series(y.iloc[-4:].mean(),pd.date_range(y.index[-1]+pd.Timedelta(days=7),periods=h,freq="W"))}
fm={};ph_all={}
for k,f in cand.items(): p=f(tr,test); p.index=te.index; ph_all[k]=p; fm[k]=(mae(te,p),mape(te,p))
print(fm); best=min(fm,key=lambda k:fm[k][1]); ph=ph_all["Holt damped trend"]; naive=ph_all["Naive (last-4-week mean)"]
prof_dow=ser[-70:].groupby(ser[-70:].index.dayofweek).mean(); prof_dow/=prof_dow.sum()
wf=cand[best](wk,5); fdays=pd.date_range(ser.index.max()+pd.Timedelta(days=1),periods=30); fvals=[]
for dte in fdays:
    wi=min(int((dte-ser.index.max()).days//7),4); v=wf.iloc[wi]*prof_dow[dte.dayofweek]
    if (dte.month==12 and dte.day>=24) or (dte.month==1 and dte.day<=3): v=0.0      # annual shutdown seen in 2010/11 data (23-Dec to 4-Jan)
    fvals.append(v)
fc=pd.Series(fvals,fdays,name="Forecast sales"); print("best model",best)
fig,ax=plt.subplots(1,2,figsize=(14,4.4)); ax[0].plot(wk.index,wk/1e3,color="#222",label="Weekly actual"); ax[0].plot(te.index,ph/1e3,color=OR,marker="o",label="Holt (8-wk hold-out)"); ax[0].plot(te.index,naive/1e3,color="#9aa0a6",ls="--",label="Naive (last-4-wk mean)"); ax[0].set_title("Forecast validation – weekly sales (thousand £)"); ax[0].legend(frameon=False)
ax[1].plot(ser.index[-60:],ser[-60:]/1e3,color="#222",label="Actual (last 60 days)"); ax[1].plot(fc.index,fc/1e3,color=OR,marker="o",ms=3,label=f"30-day forecast ({best})"); ax[1].set_title("30-day sales forecast (thousand £/day)"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig("forecast.png"); plt.close()

# ---------- Real-time simulation feed ----------
rng=np.random.default_rng(15); top=prod.head(12).index.tolist(); price=d.groupby("Description").UnitPrice.median()
cs=ctry.head(8).index.tolist(); pw=ctry.head(8).values/ctry.head(8).sum()
t0=pd.Timestamp("2026-10-02 09:00"); n=600; times=t0+pd.to_timedelta(np.cumsum(rng.exponential(1.6,n)),unit="m")
live=pd.DataFrame({"DateTime":times.round("min"),"Product":rng.choice(top,n),"Country":rng.choice(cs,n,p=pw),"Quantity":rng.integers(1,25,n)})
live["UnitPrice"]=live.Product.map(price).round(2); live["Sales"]=(live.Quantity*live.UnitPrice).round(2); live.to_csv("Live_Sales.csv",index=False)

# ---------- tables ----------
comp=pd.DataFrame([["AI-driven Analytics","Customer segmentation, anomaly detection, forecasting","High","Moderate","Automated insights"],["Cloud BI","Online reporting, sharing, scheduled refresh","High","Moderate","Anywhere access"],["Real-Time BI","Continuous monitoring of live sales","High","High","Faster response"],["Advanced Visualization","Interactive dashboards, map, drill-down","High","Low–Moderate","Better understanding"]],columns=["Technology","Application","Scalability","Implementation effort","Business benefit"])
roi=pd.DataFrame([["Data analysis","Manual / interactive","AI-assisted (RFM, anomalies, key influencers)"],["Reporting","Static / interactive","Interactive + automated refresh"],["Forecasting","Separate analysis","Integrated in dashboard"],["Monitoring","Periodic","Near-real-time architecture"],["Collaboration","Local files","Cloud-based sharing"],["Decision support","Historical","Historical + predictive"]],columns=["Factor","Traditional BI","Emerging BI"])
ins=[f"UK = {uk:.1f}% of sales; top export markets: {', '.join(ctry.drop('United Kingdom').head(3).index)}.",
 f"Peak month {mon.idxmax()} (£{mon.max():,.0f}); Nov is highest, Q4 seasonal build-up before Christmas.",
 f"Top product: {prod.index[0]} (£{prod.Sales.iloc[0]:,.0f}).",
 f"Champions segment = {prof['Revenue %'].iloc[0]:.0f}% of revenue from {int(prof.Customers.iloc[0])} customers ({prof.Customers.iloc[0]/prof.Customers.sum()*100:.0f}% of base).",
 f"{len(an)} anomalous days detected (spikes/drops) – candidates for stock/promo/outage review.",
 f"30-day forecast ≈ £{fc.sum():,.0f} using {best} (hold-out MAPE {fm[best][1]:.1f}%); Christmas shutdown (24-Dec to 3-Jan) applied; only 1 year of history, so seasonality is not learnt – treat as indicative."]
print("\n".join(ins))
with pd.ExcelWriter("Experiment_15_Retail_BI_Results.xlsx") as xw:
    pd.DataFrame({"Item":list(log),"Value":list(log.values())}).to_excel(xw,sheet_name="Data cleaning",index=False)
    pd.DataFrame({"KPI":list(K),"Value":[round(v,2) for v in K.values()]}).to_excel(xw,sheet_name="KPIs",index=False)
    mon.reset_index().to_excel(xw,sheet_name="Monthly sales",index=False); ctry.reset_index().to_excel(xw,sheet_name="Sales by country",index=False); prod.head(25).reset_index().to_excel(xw,sheet_name="Top products",index=False)
    prof.round(2).reset_index().to_excel(xw,sheet_name="RFM segments",index=False); rfm.reset_index().to_excel(xw,sheet_name="Customer RFM",index=False)
    imp[::-1].reset_index().rename(columns={"index":"Feature",0:"Importance"}).to_excel(xw,sheet_name="Key influencers",index=False)
    an_tbl.to_excel(xw,sheet_name="Anomalies",index=False); fc.reset_index().rename(columns={"index":"Date"}).to_excel(xw,sheet_name="Forecast 30d",index=False)
    pd.DataFrame(fm,index=["MAE","MAPE %"]).T.reset_index().to_excel(xw,sheet_name="Forecast validation",index=False)
    comp.to_excel(xw,sheet_name="Tech comparison",index=False); roi.to_excel(xw,sheet_name="ROI (qualitative)",index=False); pd.DataFrame({"Insight":ins}).to_excel(xw,sheet_name="Insights",index=False)
    for sh in xw.sheets.values(): sh.set_column(0,0,34); sh.set_column(1,8,24)
import json; json.dump({"K":K,"ins":ins,"log":log,"fm":fm,"uk":uk},open("summary.json","w"),default=float)
d[["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country","Sales"]].to_pickle("clean.pkl")
print("done")
