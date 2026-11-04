"""Fit on failure rows, select on policy rows, and score unseen images."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import D,Q,U
from .images import FAMILIES

COVERAGES = (.8,.9)
LEARNED = {"U":U,"P5":U+Q,"UD":U+D,"P6":U+D+Q}
ACQUISITION = {"defocus","motion","vignette","exposure"}


def read_table(path):
    frame = pd.read_csv(path,dtype={"image":str,"patient":str})
    meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
    if frame.empty or frame.duplicated(["image","variant"]).any():
        raise ValueError("Empty feature file or duplicate image/variant pairs")
    return frame,meta


def errors(table,target):
    if target == "grade":
        return (table.prediction != table.grade).to_numpy()
    return ((table.ref_probability >= .5) != (table.grade >= 2)).to_numpy()


def fit_one(table,columns,y,mlp,seed):
    if len(set(y)) < 2:
        raise ValueError("Failure fitting needs both correct and incorrect predictions")
    if mlp:
        return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),
                             random_state=seed,max_iter=500)).fit(table[columns],y)
    model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    # Do not select regularisation on policy or test data. Use the default C if
    # there are too few patients of either outcome to cross-validate.
    groups = table.patient
    folds = min(5,groups.nunique(),min(groups[y == label].nunique() for label in (False,True)))
    if folds >= 2:
        cv = StratifiedGroupKFold(folds,shuffle=True,random_state=seed)
        splits = list(cv.split(table[columns],y,groups))
        if all(len(np.unique(y[a]))==2 and len(np.unique(y[b]))==2 for a,b in splits):
            model = GridSearchCV(model,{"logisticregression__C":[.01,.1,1.,10.,100.]},
                                 cv=splits,scoring="neg_log_loss",error_score="raise")
    return model.fit(table[columns],y)


def scores(table,bundle):
    s = {"P1_max":table.unc_max.to_numpy(),"P1_entropy":table.unc_entropy.to_numpy(),
         "P2":table.unc_tta.to_numpy(),"P3":1-table.q_good.to_numpy(),
         "P4":bundle["scaler"].transform(table[U]).mean(axis=1)}
    for name,model in bundle["models"].items():
        cols = LEARNED.get(name,U+D+Q)
        s[name] = model.predict_proba(table[cols])[:,1]
    return s


def acquisition(table):
    strongest = table[D].to_numpy().argmax(axis=1)
    present = table[["present_"+f for f in FAMILIES]].to_numpy()
    return (present[np.arange(len(table)),strongest]>=.5) & np.isin(strongest,
                [i for i,f in enumerate(FAMILIES) if f in ACQUISITION])


def retake_threshold(table,deferred,args):
    subset = table[deferred & acquisition(table) & (table.grade>=0)].sort_values("ref_probability")
    if subset.empty:
        return -1.
    count = np.arange(1,len(subset)+1)
    share = (subset.grade.to_numpy()>=2).cumsum()/count
    # Rule is inclusive. Check complete tied groups only.
    prob = subset.ref_probability.to_numpy()
    eligible = (count>=args.min_retakes) & (share<=args.max_referable_share) & np.r_[prob[:-1]!=prob[1:],True]
    return float(prob[eligible].max()) if eligible.any() else -1.


def fit(args):
    train, meta = read_table(args.fit)
    policy, other = read_table(args.policy)
    if meta != other or set(train.split)!={"fit"} or set(policy.split)!={"policy"}:
        raise ValueError("Expected matching model checkpoints and fit/policy partitions")
    if set(train.patient)&set(policy.patient):
        raise ValueError("Failure-fitting and threshold-selection patients overlap")
    train = train[train.grade>=0].reset_index(drop=True)
    if len(train)==0:
        raise ValueError("No reference grades in fit data")
    bundle = {"meta":meta,"targets":{},"seed":args.seed}
    for target in ("grade","referral"):
        part = {"models":{},"scaler":StandardScaler().fit(train[U]),"thresholds":{},"retake":{}}
        y = errors(train,target)
        for name,cols in LEARNED.items():
            part["models"][name] = fit_one(train,cols,y,False,args.seed)
        if args.mlp:
            part["models"]["MLP"] = fit_one(train,U+D+Q,y,True,args.seed)
        for name,score in scores(policy,part).items():
            part["thresholds"][name] = {c:float(np.quantile(score,c,method="inverted_cdf")) for c in COVERAGES}
        for c in COVERAGES:
            risk = scores(policy,part)["P6"]
            part["retake"][c] = retake_threshold(policy,risk>part["thresholds"]["P6"][c],args)
        bundle["targets"][target] = part
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,path)


def route(table,part,c):
    risk = scores(table,part)["P6"]
    accepted = risk <= part["thresholds"]["P6"][c]
    retake = (~accepted) & acquisition(table) & (table.ref_probability.to_numpy()<=part["retake"][c])
    return risk, accepted, np.where(accepted,"accept",np.where(retake,"retake","review"))
