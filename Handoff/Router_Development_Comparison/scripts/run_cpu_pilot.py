import gzip,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(4)
rows=[json.loads(s) for s in gzip.decompress((ROOT/'data/paired_full.jsonl.gz').read_bytes()).split(b'\n') if s]
rows=[r for r in rows if r['measurement_valid'] and r['same_prompt'] and r['cheap_score'] in [0,1] and r['strong_score'] in [0,1] and r['cheap_cost']>0 and r['strong_cost']>0]
x=[r['prompt'] for r in rows]; groups=[r['group_id'] for r in rows]
y=np.array([r['strong_score']-r['cheap_score'] for r in rows]); sc=np.array([r['strong_score'] for r in rows]); cc=np.array([r['cheap_score'] for r in rows]); cs=np.array([r['strong_cost'] for r in rows]); cl=np.array([r['cheap_cost'] for r in rows])
print('AUDIT',json.dumps({'n':len(rows),'groups':len(set(groups)),'strong_acc':sc.mean(),'cheap_acc':cc.mean(),'strong_cost':cs.sum(),'cheap_cost':cl.sum(),'Y_counts':{str(k):int(sum(y==k)) for k in [-1,0,1]}}),flush=True)
res=[]
for seed in [17,29,43]:
 tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(x,y,groups))
 vec=TfidfVectorizer(max_features=12000,min_df=3,ngram_range=(1,2),sublinear_tf=True)
 a=vec.fit_transform([x[i] for i in tr]); b=vec.transform([x[i] for i in va])
 for name,model in [('logistic',LogisticRegression(C=1,max_iter=250)),('forest',RandomForestRegressor(n_estimators=160,max_depth=20,min_samples_leaf=10,max_features=.1,n_jobs=4,random_state=seed))]:
  start=time.time(); model.fit(a,y[tr]); seconds=time.time()-start
  if name=='logistic':
   probs=model.predict_proba(b); pred=probs@model.classes_
  else: pred=model.predict(b)
  thresholds=np.r_[-np.inf,np.unique(pred)]
  for delta in [0,.01,.02]:
   best=None
   for tau in thresholds:
    strong=pred>tau
    acc=np.where(strong,sc[va],cc[va]).mean(); cost=np.where(strong,cs[va],cl[va]).sum()
    if acc>=sc[va].mean()-delta-1e-12 and (best is None or cost<best['cost']):
     best={'cost':cost,'accuracy':acc,'threshold':float(tau),'cheap_fraction':float((~strong).mean())}
   savings=1-best['cost']/cs[va].sum(); drop=(sc[va].mean()-best['accuracy'])*100
   out={'seed':seed,'model':name,'allowance_pp':delta*100,'n_train':len(tr),'n_validation':len(va),'fit_seconds':seconds,'savings_pct':100*savings,'loss_pp':drop,**best}
   res.append(out); print(json.dumps(out),flush=True)
(ROOT/'results').mkdir(exist_ok=True)
with (ROOT/'results/development_rerun.json').open('w') as f:
 json.dump(res,f,indent=2)
