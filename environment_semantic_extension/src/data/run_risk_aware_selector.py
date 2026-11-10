"""
Nested-CV risk-aware utility-adaptive semantic beam selector.
Run: python src/data/run_risk_aware_selector.py
Uses aligned prediction + existing per-sample utility artifacts.
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "results/integration/semantic_predictions_unit3_test.csv"
UTILITY_PATH = ROOT / "results/integration/adaptive_beam_sample_results.csv"
OUT = ROOT / "results/analysis/risk_aware_adaptive_selector"
KS = (1, 3, 5)
TAUS_DB = (0.5, 1.0, 2.0, 3.0, 5.0)
BETAS = (0.0, 0.5, 1.0, 1.645, 2.0)
OUTER_FOLDS, INNER_FOLDS, RIDGE_ALPHA, SEED = 5, 4, 3.0, 41

def ridge_predict(xtr, ytr, xte):
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-8] = 1.0
    a, b = (xtr-mu)/sd, (xte-mu)/sd
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    penalty = np.eye(a.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ ytr)
    return b @ coef

def make_features(df):
    p = np.column_stack([df[f"prob_{i}"].to_numpy(float) for i in range(1, 6)])
    p = np.clip(p, 1e-12, 1.0)
    pn = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    ent = -np.sum(pn * np.log(np.maximum(pn, 1e-12)), axis=1)
    sp = np.sort(p, axis=1)[:, ::-1]
    return np.column_stack([p, sp[:,0]-sp[:,1], sp[:,0]-sp[:,2],
        p[:,:2].sum(1), p[:,:3].sum(1), p.sum(1), ent, ent/np.log(5), np.log(p[:,0])])

def metrics(name, r, c, oi):
    r, c, oi = np.asarray(r,float), np.asarray(c,float), np.asarray(oi,bool)
    return {"method":name, "mean_regret_db":r.mean(), "median_regret_db":np.median(r),
        "p95_regret_db":np.percentile(r,95), "within_1db_pct":100*np.mean(r<=1),
        "within_3db_pct":100*np.mean(r<=3), "oracle_in_candidates_pct":100*np.mean(oi),
        "mean_candidates":c.mean(), "candidate_reduction_pct":100*(1-c.mean()/64),
        "exhaustive_fallback_pct":100*np.mean(c>=64)}

def split_folds(indices, n, rng):
    ids = np.asarray(indices).copy()
    rng.shuffle(ids)
    return [a for a in np.array_split(ids, min(n,len(ids))) if len(a)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred, util = pd.read_csv(PRED_PATH), pd.read_csv(UTILITY_PATH)
    if pred.duplicated(["test_row","original_sample_id"]).any():
        raise ValueError("Duplicate prediction keys")
    if util.duplicated(["test_row","original_sample_id","strategy"]).any():
        raise ValueError("Duplicate utility keys")
    expected = {1:"Semantic_Top1",3:"Semantic_Top3",5:"Semantic_Top5"}
    pv = util.pivot(index=["test_row","original_sample_id"], columns="strategy",
        values=["regret_db","candidate_count","oracle_in_candidates"])
    pv.columns = [f"{a}__{b}" for a,b in pv.columns]
    df = pred.merge(pv.reset_index(), on=["test_row","original_sample_id"],
        how="inner", validate="one_to_one")
    if len(df) != len(pred):
        raise ValueError(f"Alignment failure: {len(df)}/{len(pred)}")
    pp=[]
    for v in df.top_k_probabilities:
        a=ast.literal_eval(v) if isinstance(v,str) else list(v)
        if len(a)<5: raise ValueError("Top-K probability list shorter than five")
        pp.append([float(z) for z in a[:5]])
    pp=np.asarray(pp,float)
    for j in range(5): df[f"prob_{j+1}"]=pp[:,j]
    ex=util[util.strategy=="Exhaustive_64"]
    if len(ex)!=len(df) or not np.allclose(ex.regret_db.to_numpy(float),0,atol=1e-8):
        raise ValueError("Exhaustive baseline integrity check failed")
    x=make_features(df)
    y={k:df[f"regret_db__{expected[k]}"].to_numpy(float) for k in KS}
    c={k:df[f"candidate_count__{expected[k]}"].to_numpy(float) for k in KS}
    oi={k:df[f"oracle_in_candidates__{expected[k]}"].astype(bool).to_numpy() for k in KS}
    rng=np.random.default_rng(SEED)
    outer=split_folds(np.arange(len(df)),OUTER_FOLDS,rng)
    policies=[(t,b) for t in TAUS_DB for b in BETAS]
    pr={z:np.full(len(df),np.nan) for z in policies}
    pc={z:np.full(len(df),np.nan) for z in policies}
    poi={z:np.zeros(len(df),bool) for z in policies}
    choice={z:np.full(len(df),-1,int) for z in policies}
    pred_oof={k:np.full(len(df),np.nan) for k in KS}
    sig_oof={k:np.full(len(df),np.nan) for k in KS}
    fold_id=np.full(len(df),-1,int)
    for fno, test in enumerate(outer):
        train=np.setdiff1d(np.arange(len(df)),test)
        fold_id[test]=fno
        inner=split_folds(train,INNER_FOLDS,rng)
        inner_oof={k:np.full(len(df),np.nan) for k in KS}
        for iv in inner:
            it=np.setdiff1d(train,iv)
            for k in KS: inner_oof[k][iv]=ridge_predict(x[it],y[k][it],x[iv])
        sig={}
        for k in KS:
            resid=y[k][train]-inner_oof[k][train]
            med=np.median(resid)
            mad=1.4826*np.median(np.abs(resid-med))
            rmse=np.sqrt(np.mean(resid**2))
            sig[k]=float(max(mad,0.5*rmse,1e-6))
            pred_oof[k][test]=ridge_predict(x[train],y[k][train],x[test])
            sig_oof[k][test]=sig[k]
        for tau,beta in policies:
            chosen=np.full(len(test),64,int)
            for j,idx in enumerate(test):
                for k in KS:
                    if pred_oof[k][idx]+beta*sig[k] <= tau:
                        chosen[j]=k
                        break
            z=(tau,beta)
            for j,idx in enumerate(test):
                k=int(chosen[j])
                if k==64:
                    pr[z][idx],pc[z][idx],poi[z][idx]=0.0,64.0,True
                else:
                    pr[z][idx],pc[z][idx],poi[z][idx]=y[k][idx],c[k][idx],oi[k][idx]
                choice[z][idx]=k
    if any(np.isnan(v).any() for v in pred_oof.values()):
        raise RuntimeError("Incomplete nested cross-validation predictions")
    rows=[]
    for k in KS: rows.append(metrics(f"Fixed_Top{k}",y[k],c[k],oi[k]))
    rows.append(metrics("Exhaustive_64",ex.regret_db,ex.candidate_count,ex.oracle_in_candidates))
    policy_rows=[]
    for tau,beta in policies:
        z=(tau,beta); r,cc,o=pr[z],pc[z],poi[z]
        rows.append(metrics(f"RiskAware_tau{tau:g}_beta{beta:g}",r,cc,o))
        policy_rows.append({"tau_db":tau,"beta":beta,"mean_regret_db":r.mean(),
            "p95_regret_db":np.percentile(r,95),"mean_candidates":cc.mean(),
            "candidate_reduction_pct":100*(1-cc.mean()/64),
            "within_1db_pct":100*np.mean(r<=1),"within_3db_pct":100*np.mean(r<=3),
            "exhaustive_fallback_pct":100*np.mean(cc>=64),
            "top1_selection_pct":100*np.mean(choice[z]==1),
            "top3_selection_pct":100*np.mean(choice[z]==3),
            "top5_selection_pct":100*np.mean(choice[z]==5)})
    comp,pdf=pd.DataFrame(rows),pd.DataFrame(policy_rows)
    sample=df.copy(); sample["outer_fold"]=fold_id
    for k in KS:
        sample[f"oof_predicted_regret_top{k}_db"]=pred_oof[k]
        sample[f"inner_residual_sigma_top{k}_db"]=sig_oof[k]
    for tau,beta in policies:
        tag=f"tau{tau:g}_beta{beta:g}".replace(".","p")
        sample[f"selected_k_{tag}"]=choice[(tau,beta)]
        sample[f"realized_regret_{tag}_db"]=pr[(tau,beta)]
    comp.to_csv(OUT/"method_comparison.csv",index=False)
    pdf.to_csv(OUT/"risk_policy_sweep.csv",index=False)
    sample.to_csv(OUT/"per_sample_results.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,6))
    for _,r in comp.iterrows():
        ax.scatter(r.mean_candidates,r.mean_regret_db)
        if r.method.startswith("Fixed") or r.method=="Exhaustive_64" or r.method in {
            "RiskAware_tau1_beta1","RiskAware_tau2_beta1","RiskAware_tau3_beta1","RiskAware_tau5_beta1"}:
            ax.annotate(r.method,(r.mean_candidates,r.mean_regret_db),fontsize=7,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("Mean candidate beams"); ax.set_ylabel("Mean realized candidate-set regret (dB)")
    ax.set_title("Nested-CV risk-aware semantic beam selection"); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/"risk_aware_regret_overhead.png",dpi=180); plt.close(fig)
    notes=["RISK-AWARE UTILITY-ADAPTIVE SELECTOR","====================================",
        f"Aligned samples: {len(df)}","Join: test_row + original_sample_id",
        f"Outer folds: {OUTER_FOLDS}; inner folds: {INNER_FOLDS}",
        "Residual uncertainty estimated only within each outer training fold.",
        "Rule: choose smallest K with predicted regret + beta*sigma <= tau; else exhaustive.",
        "Sigma: robust global residual scale from inner OOF errors.",
        "Development-only evaluation on existing 449-sample test export.",
        "Candidate count is a proxy, not measured latency/energy.",
        "Candidate-set regret assumes strongest measured candidate can be selected.",
        "POLICY SWEEP",pdf.to_string(index=False),"METHOD COMPARISON",comp.to_string(index=False)]
    (OUT/"experiment_notes.txt").write_text("\n".join(notes),encoding="utf-8")
    print(f"Aligned samples: {len(df)}")
    print("\n=== BASELINES + RISK-AWARE POLICIES ==="); print(comp.to_string(index=False))
    print("\n=== RISK POLICY SWEEP ==="); print(pdf.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")

if __name__=="__main__": main()
