"""Patient-paired evaluation with separate subgroup and fixed-threshold reports."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, cohen_kappa_score, f1_score,
                             roc_auc_score)
from .features import PROBS
from .images import seed_for
from .policy import COVERAGES, errors, read_table, route, scores


def augrc(y,score):
    # Include (0,0). Between two coverage points F rises linearly, which
    # averages over all orderings of equally scored examples.
    if not len(y):
        return np.nan
    order = np.argsort(score,kind="stable")
    sorted_score = np.asarray(score)[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_score)!=0),len(order)-1]
    x = np.r_[0,(endpoints+1)/len(y)]
    f = np.r_[0,np.cumsum(np.asarray(y)[order])[endpoints]/len(y)]
    return float(np.sum(np.diff(x)*(f[1:]+f[:-1])/2))


def ece(conf,observed):
    groups = np.minimum((np.asarray(conf)*15).astype(int),14)
    return sum(np.mean(groups==i)*abs(np.mean(conf[groups==i])-np.mean(observed[groups==i]))
               for i in np.unique(groups))


def metrics(frame,accepted,target):
    known = frame.grade.to_numpy()>=0
    ref = frame.grade.to_numpy()>=2
    ref_hat = frame.ref_probability.to_numpy()>=.5
    e = errors(frame,target)
    a = accepted & known
    bad = a & e
    def ratio(x,y):
        return x.sum()/y.sum() if y.any() else np.nan
    return {"n":len(frame),"coverage_all":accepted.mean(),"coverage_evaluable":ratio(a,known),
            "selective_risk":ratio(bad,a),"silent_failure":ratio(bad,known),
            "ref_sensitivity_accepted":ratio(a & ref & ref_hat,a & ref),
            "missed_referable_rate":ratio(a & ref & ~ref_hat,ref),
            "unknown_grade_accepted":int((accepted & ~known).sum())}


def classifier_summary(frame):
    t = frame[frame.grade>=0]
    if t.empty:
        return {"evaluable":0}
    y = t.grade.to_numpy()
    p = t[PROBS].to_numpy()
    pred = p.argmax(1)
    b = t.ref_probability.to_numpy()
    truth = y>=2
    return {"evaluable":len(y),"accuracy":(pred==y).mean(),"grade_mae":np.abs(pred-y).mean(),
            "qwk":cohen_kappa_score(y,pred,labels=range(5),weights="quadratic") if len(set(y)|set(pred))>1 else np.nan,
            "macro_f1":f1_score(y,pred,labels=range(5),average="macro",zero_division=0),
            "grade_ece":ece(p.max(1),y==pred),"ref_ece":ece(b,truth),
            "ref_auroc":roc_auc_score(truth,b) if truth.min()!=truth.max() else np.nan,
            "ref_auprc":average_precision_score(truth,b) if truth.min()!=truth.max() else np.nan,
            "grade_nll":-np.log(p[np.arange(len(y)),y].clip(1e-12,1)).mean(),
            "grade_brier":np.square(p-np.eye(5)[y]).sum(1).mean(),
            "ref_brier":np.square(b-truth).mean(),
            "ref_nll":-np.log(np.where(truth,b,1-b).clip(1e-12,1)).mean()}


def paired_bootstrap(frame,y,main,other,count,rng):
    blocks = list(frame.groupby("patient",sort=False).indices.values())
    diffs = []
    for _ in range(count):
        indices = np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])
        diffs.append(augrc(y[indices],main[indices])-augrc(y[indices],other[indices]))
    return np.quantile(diffs,[.025,.975])


def evaluate(args):
    table,meta = read_table(args.features)
    bundle = joblib.load(args.checker)
    if bundle["meta"] != meta or not set(table.split)<={"test","external"}:
        raise ValueError("Evaluation requires held-out rows and the checker's original ensemble")
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(bundle["seed"])
    results, grades, strata, overlaps, intervals = [], [], [], [], []
    groups = {"all":table, **{"family/"+str(k):v for k,v in table.groupby("family")},
              **{"variant/"+str(k):v for k,v in table.groupby("variant")}}
    for quality,t in table[table.variant=="original"].groupby("quality"):
        groups[f"original/quality-{quality}"] = t
    for group,subset in groups.items():
        t = subset.reset_index(drop=True)
        grades.append({"group":group,**classifier_summary(t)})
        known = t.grade.to_numpy()>=0
        tie = np.array([seed_for(bundle["seed"],i,v) for i,v in zip(t.image,t.variant)],dtype=np.uint64)
        for target,part in bundle["targets"].items():
            e = errors(t,target)
            y = e[known]
            all_scores = scores(t,part)
            all_scores.update(P0=tie.astype(float)/float(2**64),P7_oracle=e.astype(float))
            sets = {}
            for name,score in all_scores.items():
                score = np.asarray(score,dtype=float)
                evaluable_scores = score[known]
                detection = {"augrc":augrc(y,evaluable_scores),
                             "auroc_failure":roc_auc_score(y,evaluable_scores) if len(set(y))==2 else np.nan,
                             "auprc_failure":average_precision_score(y,evaluable_scores) if len(set(y))==2 else np.nan,
                             "full_coverage_error":y.mean() if len(y) else np.nan}
                if name in part["models"] and len(y):
                    detection["failure_ece"] = ece(evaluable_scores, y)
                    detection["failure_brier"] = np.square(evaluable_scores-y).mean()
                for c in COVERAGES:
                    order = np.lexsort((tie,score))
                    accepted = np.zeros(len(t),dtype=bool)
                    accepted[order[:int(np.floor(c*len(t)+.5))]]=True
                    modes = {"rank":accepted}
                    if name in part["thresholds"]:
                        modes["fixed"] = score<=part["thresholds"][name][c]
                    for mode,a in modes.items():
                        label = {"group":group,"target":target,"policy":name,"budget":c,"mode":mode}
                        results.append({**label,**metrics(t,a,target),**detection})
                        if name in ("P2","P3","P6") and mode=="rank":
                            sets[name,c] = ~a
                            if group=="all":
                                tmp = t.assign(deferred=(~a).astype(int))
                                for (g,q),sub in tmp.groupby(["grade","quality"]):
                                    strata.append({**label,"grade":g,"quality":q,"n":len(sub),
                                                   "deferred_fraction":sub.deferred.mean()})
            for c in COVERAGES:
                for x,z in (("P2","P3"),("P2","P6"),("P3","P6")):
                    a,b = sets[x,c],sets[z,c]
                    overlaps.append({"group":group,"target":target,"budget":c,"A":x,"B":z,
                                     "jaccard":(a&b).sum()/(a|b).sum() if (a|b).any() else np.nan})
            if args.bootstrap and group=="all" and known.any():
                for other in ("P2","P3","P5"):
                    lo,hi = paired_bootstrap(t[known].reset_index(drop=True),y,all_scores["P6"][known],
                                             all_scores[other][known],args.bootstrap,rng)
                    intervals.append({"target":target,"comparison":"P6-"+other,
                                      "augrc_difference":augrc(y,all_scores["P6"][known])-augrc(y,all_scores[other][known]),
                                      "lower95":lo,"upper95":hi})
    for key,values in (("policies",results),("grading",grades),("strata",strata),("overlap",overlaps),
                       ("bootstrap",intervals)):
        if values:
            pd.DataFrame(values).to_csv(out/(key+".csv"),index=False)
    decisions = table.copy()
    retakes = []
    for c in COVERAGES:
        risk,_,routing = route(table,bundle["targets"]["grade"],c)
        decisions["grade_failure_risk"] = risk
        decisions[f"route_{c}"] = routing
        selected = (routing=="retake") & (table.grade>=0)
        retakes.append({"budget":c,"retake_n":int((routing=="retake").sum()),
                        "labelled_retake_n":int(selected.sum()),
                        "referable_share":(table.grade[selected]>=2).mean()})
    pd.DataFrame(retakes).to_csv(out/"retakes.csv",index=False)
    decisions.to_csv(out/"decisions.csv",index=False)
