"""Entry-point script for SOC pipeline commands (scaffolding)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .io.insitu import load_insitu_csv
from .models.ngb import make_ngb, NGBoostConfig
from .models.transformations import LogTransform, IdentityTransform
from .evaluation.metrics import negative_log_likelihood, crps, prediction_interval_coverage, mae_rmse
from .evaluation.calibration import pit_values
from .evaluation.spatial_cv import leave_group_out_indices
from .predict.raster import raster_predict
from .utils.ood import fit_ood_models, ood_scores
import numpy as np
import joblib


def select_transform(dist: str):
    if dist == "lognormal" or dist == "normal":
        return LogTransform()
    return IdentityTransform()


def cmd_train(args):
    df = load_insitu_csv(args.insitu)
    feature_cols = [c for c in df.columns if c not in {"soc","latitude","longitude","top_depth_cm","bottom_depth_cm", args.group_col}]
    if args.group_col in df.columns:
        groups = df[args.group_col]
    else:
        groups = None
    X = df[feature_cols].values
    y_raw = df["soc"].values.astype(float)
    transform = select_transform(args.dist)
    y = transform.forward(y_raw)
    # Optional sample weighting from measurement variance/STD
    sample_weight = None
    if args.weight_col and args.weight_col in df.columns:
        w = df[args.weight_col].astype(float).values
        # If provided is variance -> weight = 1/var; if std -> 1/std^2 ; assume var if magnitude small
        if args.weight_col.lower().endswith("var"):
            sample_weight = 1.0 / np.clip(w, 1e-8, None)
        elif args.weight_col.lower().endswith("std"):
            sample_weight = 1.0 / np.clip(w, 1e-8, None) ** 2
        else:  # treat as provided weights directly
            sample_weight = w

    cfg = NGBoostConfig(distribution=args.dist if args.dist != "lognormal" else "normal")
    model = make_ngb(cfg)

    # Spatial CV for early stopping (simple manual loop)
    if groups is not None:
        splits = list(leave_group_out_indices(df, args.group_col))
    else:
        # fallback: simple holdout last 20%
        n = len(df)
        idx = np.arange(n)
        np.random.default_rng(42).shuffle(idx)
        cut = int(0.8*n)
        splits = [(idx[:cut], idx[cut:])]

    best_score = np.inf
    best_iter = cfg.n_estimators
    # Fit incrementally capturing validation NLL
    model.fit(X, y, sample_weight=sample_weight)  # NGBoost supports sample_weight
    # Evaluate
    for train_idx, test_idx in splits:
        nll = negative_log_likelihood(model, X[test_idx], y[test_idx])
        if nll < best_score:
            best_score = nll
            best_iter = model.n_estimators

    bundle_base = {"features": feature_cols, "config": cfg, "transform": transform}
    # OOD models (Mahalanobis + IsolationForest optional)
    if args.fit_ood:
        ood_bundle = fit_ood_models(X, methods=args.ood_methods.split(","))
        bundle_base["ood"] = ood_bundle

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bootstrap > 1:
        rng = np.random.default_rng(42)
        # Simple spatial block bootstrap if groups present else standard
        models = []
        for b in range(args.bootstrap):
            if groups is not None:
                unique_groups = np.unique(groups)
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                sel = np.isin(groups, sampled_groups)
            else:
                sel = rng.integers(0, len(X), size=len(X))
            Xb = X[sel]
            yb = y[sel]
            swb = sample_weight[sel] if sample_weight is not None else None
            mb = make_ngb(cfg)
            mb.fit(Xb, yb, sample_weight=swb)
            models.append(mb)
        bundle = {**bundle_base, "ensemble": True, "models": models}
        joblib.dump(bundle, out)
        print(f"Saved ensemble ({len(models)} members) to {out}")
    else:
        bundle = {**bundle_base, "ensemble": False, "model": model}
        joblib.dump(bundle, out)
        print(f"Saved model to {out}; best_iter={best_iter} val_nll={best_score:.4f}")


def cmd_evaluate(args):
    bundle = joblib.load(args.model)
    feature_cols = bundle["features"]
    transform = bundle.get("transform")
    df = load_insitu_csv(args.insitu)
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols_present].values
    y_raw = df["soc"].values.astype(float)
    y = transform.forward(y_raw) if transform else y_raw
    metrics = {}
    def eval_model(m):
        out = {
            "nll": negative_log_likelihood(m, X, y),
            "crps": crps(m, X, y),
            **mae_rmse(m, X, y),
        }
        for a in (0.5, 0.8, 0.95):
            cov = prediction_interval_coverage(m, X, y, alpha=a)
            out[f"coverage_{a}"] = cov["empirical_coverage"]
        pit = pit_values(m, X, y)
        out["pit_mean"], out["pit_std"] = float(pit.mean()), float(pit.std())
        return out
    if bundle.get("ensemble"):
        model_metrics = [eval_model(m) for m in bundle["models"]]
        # Aggregate
        avg = {k: float(np.mean([mm[k] for mm in model_metrics])) for k in model_metrics[0]}
        metrics["ensemble_mean"] = avg
        metrics["members"] = model_metrics
    else:
        metrics.update(eval_model(bundle["model"]))
    # OOD score stats if available
    if "ood" in bundle:
        ood = ood_scores(bundle["ood"], X)
        metrics["ood_stats"] = {k: float(np.mean(v)) for k, v in ood.items()}
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.metrics_out}")
    else:
        print(json.dumps(metrics, indent=2))


def cmd_predict(args):
    bundle = joblib.load(args.model)
    transform = bundle.get("transform")
    raster_predict(
        bundle=bundle,
        raster_config=args.raster_config,
        out_dir=args.out_dir,
        transform=transform,
        mask_path=args.mask,
        tile_size=args.tile_size,
        ood_action=args.ood_action,
        ood_threshold=args.ood_threshold,
    )


def build_parser():
    p = argparse.ArgumentParser("soc-pipeline")
    sub = p.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--insitu", required=True)
    p_train.add_argument("--dist", choices=["lognormal","gamma"], default="lognormal")
    p_train.add_argument("--model-out", required=True)
    p_train.add_argument("--group-col", default="site_id", help="Column for leave-group-out CV")
    p_train.add_argument("--weight-col", default=None, help="Column name with measurement variance/std or weights")
    p_train.add_argument("--bootstrap", type=int, default=1, help="Number of bootstrap ensemble members")
    p_train.add_argument("--fit-ood", action="store_true", help="Fit OOD detectors (Mahalanobis, IsolationForest)")
    p_train.add_argument("--ood-methods", default="mahalanobis,isolation_forest")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--insitu", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--metrics-out", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--raster-config", required=True, help="CSV with columns feature,path")
    p_pred.add_argument("--out-dir", required=True)
    p_pred.add_argument("--mask", default=None, help="Optional raster mask (1=keep,0=mask)")
    p_pred.add_argument("--tile-size", type=int, default=512)
    p_pred.add_argument("--ood-action", choices=["none","mask","nan"], default="none")
    p_pred.add_argument("--ood-threshold", type=float, default=0.99, help="Quantile threshold for OOD distance")
    p_pred.set_defaults(func=cmd_predict)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
