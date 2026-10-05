"""Experiment 13 - Text analytics on business documents (IMDb 50K reviews as customer-feedback corpus)
Preprocess -> frequency/word clouds -> TF-IDF -> sentiment classification (LR, NB, SVM) -> keywords
-> topic modelling (LDA) -> entity extraction (rule-based) -> new-review scoring -> dashboard"""
import warnings; warnings.filterwarnings("ignore")
import re, collections, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"font.size":10})
BL,OR,GR="#2a78d6","#eb6834","#1baf7a"

# 1 load + inspect
df=pd.read_csv("IMDB Dataset.csv"); print("shape",df.shape,"| missing",int(df.isna().sum().sum()),"| duplicates",int(df.duplicated().sum()))
print(df.sentiment.value_counts().to_dict())
df=df.drop_duplicates().reset_index(drop=True); print("after dedupe",df.shape)
df["words_raw"]=df.review.str.split().str.len()

# 2 preprocessing: lower, strip HTML, expand n't, keep letters, tokenise, stop-words (negations kept), Porter stem
STOP=set(ENGLISH_STOP_WORDS)-{"not","no","nor","never","cannot","nobody","nothing","none","neither"}|{"br","movie","film","films","movies"} if False else set(ENGLISH_STOP_WORDS)-{"not","no","nor","never","cannot","nobody","nothing","none","neither"}
ps=PorterStemmer(); cache={}
def stem(w):
    r=cache.get(w)
    if r is None: r=cache[w]=ps.stem(w)
    return r
def clean(t):
    t=t.lower(); t=re.sub(r"<.*?>"," ",t); t=t.replace("n't"," not"); t=re.sub(r"[^a-z\s]"," ",t)
    return " ".join(stem(w) for w in t.split() if w not in STOP and len(w)>2)
def clean_nostem(t):
    t=t.lower(); t=re.sub(r"<.*?>"," ",t); t=t.replace("n't"," not"); t=re.sub(r"[^a-z\s]"," ",t)
    return " ".join(w for w in t.split() if w not in STOP and len(w)>2)
df["clean"]=df.review.map(clean); df["clean_ns"]=df.review.map(clean_nostem)   # stemmed (model) / readable (clouds, topics)
print("\nORIGINAL :",df.review[1][:230]); print("CLEANED  :",df.clean_ns[1][:230])
df["y"]=(df.sentiment=="positive").astype(int)
df["words_clean"]=df.clean_ns.str.split().str.len()

# 3 frequency + clouds (readable unstemmed text)
cnt=collections.Counter(" ".join(df.clean_ns).split()); top=cnt.most_common(20)
fig,ax=plt.subplots(figsize=(10,5)); ax.bar([w for w,_ in top],[c/1e3 for _,c in top],color=BL); plt.xticks(rotation=55,ha="right"); ax.set_ylabel("frequency (thousand)"); ax.set_title("Top 20 words after cleaning"); plt.tight_layout(); plt.savefig("top_words.png"); plt.close()
extra={"movie","film","one","br","like","just","really","even","see","get","make"}
def cloud(txt,cm,fn,title):
    wc=WordCloud(width=1000,height=500,background_color="white",colormap=cm,stopwords=extra,max_words=120,random_state=1).generate(txt)
    plt.figure(figsize=(11,5.5)); plt.imshow(wc,interpolation="bilinear"); plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(fn); plt.close()
cloud(" ".join(df.clean_ns[df.y==1]),"Greens","wc_positive.png","Word cloud – positive reviews")
cloud(" ".join(df.clean_ns[df.y==0]),"Reds","wc_negative.png","Word cloud – negative reviews")

# 4 TF-IDF + split
tf=TfidfVectorizer(max_features=5000,ngram_range=(1,2)); X=tf.fit_transform(df.clean); y=df.y
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); print("TF-IDF",X.shape,"train",Xtr.shape[0],"test",Xte.shape[0])

# 5 models
models={"Logistic Regression":LogisticRegression(max_iter=1000,C=2.0),"Naive Bayes":MultinomialNB(alpha=.5),"Linear SVM":LinearSVC(C=.3)}
rows=[];cms={}
for n,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte); pr,rc,f1,_=precision_recall_fscore_support(yte,p,average="binary")
    rows.append({"Model":n,"Accuracy":accuracy_score(yte,p),"Precision":pr,"Recall":rc,"F1":f1}); cms[n]=confusion_matrix(yte,p)
res=pd.DataFrame(rows); print(res.round(4).to_string())
lr=models["Logistic Regression"]; plr=lr.predict(Xte); print(classification_report(yte,plr,target_names=["negative","positive"],digits=4))
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
cm=cms["Logistic Regression"]; im=ax[0].imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2): ax[0].text(j,i,f"{cm[i,j]:,}",ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "#222",fontsize=13)
ax[0].set_xticks([0,1],["negative","positive"]); ax[0].set_yticks([0,1],["negative","positive"]); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual"); ax[0].set_title("Confusion matrix – Logistic Regression"); ax[0].grid(False)
ax[1].bar(res.Model,res.Accuracy*100,color=[BL,"#9aa0a6","#9aa0a6"]); ax[1].set_ylim(80,95); ax[1].set_title("Accuracy by model (%)")
for i,v in enumerate(res.Accuracy*100): ax[1].text(i,v+.15,f"{v:.2f}",ha="center")
plt.xticks(rotation=10); plt.tight_layout(); plt.savefig("confusion_models.png"); plt.close()

# 6 keywords (LR coefficients; stems -> readable via most common surface form)
surf={}
for w in cnt: surf.setdefault(stem(w),[]).append((cnt[w],w))
rd=lambda s: max(surf[s])[1] if s in surf else s
names=tf.get_feature_names_out(); co=lr.coef_[0]; o=np.argsort(co)
kw_pos=[(" ".join(rd(t) for t in names[i].split()),co[i]) for i in o[::-1][:20]]; kw_neg=[(" ".join(rd(t) for t in names[i].split()),co[i]) for i in o[:20]]
fig,ax=plt.subplots(1,2,figsize=(11,5.5))
ax[0].barh([k for k,_ in kw_pos][::-1],[v for _,v in kw_pos][::-1],color=GR); ax[0].set_title("Top words → positive")
ax[1].barh([k for k,_ in kw_neg][::-1],[-v for _,v in kw_neg][::-1],color=OR); ax[1].set_title("Top words → negative")
plt.tight_layout(); plt.savefig("keywords.png"); plt.close()

# 7 topic modelling (LDA, 8 topics, 20K sample)
samp=df.sample(20000,random_state=1); cv=CountVectorizer(max_features=4000,max_df=.4,min_df=10,stop_words=list(extra)); Xc=cv.fit_transform(samp.clean_ns)
lda=LatentDirichletAllocation(n_components=8,random_state=0,learning_method="online",max_iter=12,n_jobs=2).fit(Xc)
vocab=cv.get_feature_names_out(); topics=[]
tdist=lda.transform(Xc).argmax(1)
for k,c in enumerate(lda.components_): topics.append({"Topic":k+1,"Top words":", ".join(vocab[c.argsort()[::-1][:10]]),"Reviews (share %)":round((tdist==k).mean()*100,1),"Positive share %":round(samp.y.values[tdist==k].mean()*100,1)})
tp=pd.DataFrame(topics); print(tp.to_string())

# 8 entity extraction (rule-based: capitalised words not at sentence start; no NER model available offline)
cap=collections.Counter(); low=collections.Counter()
for t in df.review.sample(15000,random_state=2):
    t=re.sub(r"<.*?>"," ",t)
    for sent in re.split(r"(?<=[.!?])\s+",t):
        ws=re.findall(r"[A-Za-z']+",sent)
        for w in ws[1:]:
            (cap if w[0].isupper() else low)[w.lower()]+=1 if True else 0
ent_c=collections.Counter()
for t in df.review.sample(15000,random_state=2):
    t=re.sub(r"<.*?>"," ",t)
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b",t): ent_c[m.group(1)]+=1
ent=[(e,c) for e,c in ent_c.most_common(60) if all(low[w.lower()]<c*0.3 for w in e.split()) and not e.startswith(("The ","This ","I ","It ","A ","In "))][:20]
ent_df=pd.DataFrame(ent,columns=["Entity (names/titles)","Mentions"]); print(ent_df.head(10).to_string())

# 9 business insights
lens=df.groupby("sentiment").words_raw.mean().round(1).to_dict(); print("avg words",lens)
df["len_bin"]=pd.cut(df.words_raw,[0,100,200,300,500,3000],labels=["<100","100-200","200-300","300-500",">500"])
lb=df.groupby("len_bin").y.mean()*100
fig,ax=plt.subplots(figsize=(6,4)); ax.bar(lb.index.astype(str),lb.values,color=BL); ax.set_ylim(0,100); ax.axhline(50,color="#888",ls="--"); ax.set_title("% positive by review length (words)"); plt.tight_layout(); plt.savefig("length_vs_sentiment.png"); plt.close()

# 10 score new business reviews
new=["The product quality is excellent and delivery was fast. Very happy with the service!","Terrible support. The item arrived broken and nobody replied to my emails.","It was okay, nothing special but not bad either.","I love the design but the battery life is disappointing."]
pn=lr.predict_proba(tf.transform([clean(t) for t in new]))[:,1]
newdf=pd.DataFrame({"Review":new,"P(positive)":pn.round(3),"Prediction":np.where(pn>=.5,"Positive","Negative")}); print(newdf.to_string())

# 11 dashboard
fig=plt.figure(figsize=(16,9)); g=fig.add_gridspec(2,3,hspace=.4,wspace=.3)
a=fig.add_subplot(g[0,0]); vc=df.sentiment.value_counts(); a.bar(vc.index,vc.values/1e3,color=[GR,OR]); a.set_title("Sentiment distribution (thousand)")
a=fig.add_subplot(g[0,1]); a.bar(res.Model,res.Accuracy*100,color=[BL,"#9aa0a6","#9aa0a6"]); a.set_ylim(80,95); a.set_title("Model accuracy (%)"); plt.setp(a.get_xticklabels(),rotation=10)
a=fig.add_subplot(g[0,2]); a.barh([k for k,_ in kw_pos][:10][::-1],[v for _,v in kw_pos][:10][::-1],color=GR); a.set_title("Top positive keywords")
a=fig.add_subplot(g[1,0]); a.barh([k for k,_ in kw_neg][:10][::-1],[-v for _,v in kw_neg][:10][::-1],color=OR); a.set_title("Top negative keywords")
a=fig.add_subplot(g[1,1]); a.barh([f"T{r.Topic}: {r['Top words'].split(', ')[0]}, {r['Top words'].split(', ')[1]}" for _,r in tp.iterrows()][::-1],tp["Reviews (share %)"][::-1],color=BL); a.set_title("Topic share of reviews (%)")
a=fig.add_subplot(g[1,2]); a.barh([e for e,_ in ent[:10]][::-1],[c for _,c in ent[:10]][::-1],color="#8a63d2"); a.set_title("Most mentioned names/titles")
fig.suptitle("Experiment 13 – Text analytics dashboard (IMDb 50K reviews)",fontsize=14,weight="bold"); plt.savefig("dashboard.png",bbox_inches="tight"); plt.close()

# 12 excel
summ=pd.DataFrame({"Item":["Rows loaded","Duplicates removed","Rows analysed","Positive / Negative","Avg words (positive)","Avg words (negative)","Vocabulary after cleaning","TF-IDF features","Train / test split","Best model","Best accuracy"],
 "Value":[50000,50000-len(df),len(df),f"{int(df.y.sum())} / {int((1-df.y).sum())}",lens["positive"],lens["negative"],len(cnt),X.shape[1],f"{Xtr.shape[0]} / {Xte.shape[0]}",res.sort_values("Accuracy").iloc[-1].Model,round(res.Accuracy.max(),4)]})
with pd.ExcelWriter("Experiment_13_Text_Analytics_Results.xlsx") as xw:
    summ.to_excel(xw,sheet_name="Summary",index=False); res.round(4).to_excel(xw,sheet_name="Model comparison",index=False)
    pd.DataFrame(kw_pos,columns=["Positive keyword","LR coefficient"]).round(3).to_excel(xw,sheet_name="Keywords positive",index=False)
    pd.DataFrame(kw_neg,columns=["Negative keyword","LR coefficient"]).round(3).to_excel(xw,sheet_name="Keywords negative",index=False)
    tp.to_excel(xw,sheet_name="Topics (LDA)",index=False); ent_df.to_excel(xw,sheet_name="Entities",index=False)
    pd.DataFrame(top,columns=["Word","Frequency"]).to_excel(xw,sheet_name="Top words",index=False); newdf.to_excel(xw,sheet_name="New review scoring",index=False)
    for sh in xw.sheets.values(): sh.set_column(0,0,34); sh.set_column(1,8,22)
print("done")
