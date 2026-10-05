import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

df = pd.read_csv("customer-segmentation.csv")
print(df.shape); print(df.isnull().sum()[lambda s: s>0])
features = ['Income','MntWines','MntMeatProducts','MntFruits','MntFishProducts',
            'MntSweetProducts','MntGoldProds','NumWebPurchases','NumStorePurchases','NumCatalogPurchases']
X = df[features].copy().dropna()
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df = df.loc[X.index].copy(); df['Cluster'] = clusters
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
centers.insert(0,'Customers',np.bincount(clusters)); centers.index.name='Cluster'
print(centers.round(1).to_string()); print("silhouette",silhouette_score(X_scaled,clusters))
distances = pairwise_distances(X_scaled, kmeans.cluster_centers_)
df['Distance'] = distances.min(axis=1)
threshold = df['Distance'].quantile(0.95); df['Outlier'] = df['Distance'] > threshold
outliers = df[df['Outlier']]; print("threshold",threshold,"outliers",len(outliers))
print(outliers.groupby('Cluster').size())
# elbow
inertia=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_scaled).inertia_ for k in range(1,9)]
cols=['#1f77b4','#ff7f0e','#2ca02c']
plt.figure(figsize=(7,5))
for c in range(3): m=df.Cluster==c; plt.scatter(df.Income[m],df.MntWines[m],s=12,c=cols[c],label=f"Cluster {c}",alpha=.7)
plt.xlabel("Income"); plt.ylabel("Wine Spending"); plt.title("Customer Clusters"); plt.legend(); plt.tight_layout(); plt.savefig("clusters.png",dpi=150); plt.close()
plt.figure(figsize=(7,5))
for c in range(3): m=df.Cluster==c; plt.scatter(df.Income[m],df.MntWines[m],s=12,c=cols[c],label=f"Cluster {c}",alpha=.6)
plt.scatter(outliers.Income,outliers.MntWines,marker='x',c='red',s=40,label="Potential outlier (top 5% distance)")
plt.xlabel("Income"); plt.ylabel("Wine Spending"); plt.title("Customer Clusters and Potential Outliers"); plt.legend(); plt.tight_layout(); plt.savefig("outliers.png",dpi=150); plt.close()
plt.figure(figsize=(6,4)); plt.plot(range(1,9),inertia,'o-'); plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow (supporting)"); plt.tight_layout(); plt.savefig("elbow.png",dpi=150); plt.close()
df.to_pickle("res.pkl"); centers.to_pickle("centers.pkl")
