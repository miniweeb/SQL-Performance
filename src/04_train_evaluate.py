# -*- coding: utf-8 -*-
"""
Bước 4: Huấn luyện và đánh giá bằng cross-validation chia theo query.

Đầu vào: data/interim/features_clean.csv (feature), data/labeled/labels_df.csv (label).
Đầu ra: bảng trong reports/tables/results/, hình trong reports/figures/results/, models/model.pkl và models/run_info.json.
Đánh giá: Stratified Group K-Fold 5 fold, lặp 10 lần; các phiên bản của cùng một query luôn nằm cùng một fold.
Mọi bước có học từ dữ liệu (kể cả baseline hai ngưỡng) chỉ học trên phần train của từng fold.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import common
import config

LABEL_LEVELS = ["Normal", "Warning", "Critical"]
ML_MODELS = ["Decision Tree", "Logistic Regression", "Random Forest", "Gradient Boosting"]
LEARNING_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
COLOR_PALETTE = {"main": "#2F5D8A", "line": "#C0504D", "grey": "#8C8C8C", "light": "#DCE6F1"}


class TimeRule(BaseEstimator, ClassifierMixin):
    """Baseline 2: hai ngưỡng trên estimated time; chọn ngưỡng cho macro-F1 lớn nhất trên tập train."""

    def fit(self, X, y):
        t = np.asarray(X, dtype=float).ravel()
        # np.quantile(t, dãy các mức) trả về các giá trị cắt ở 5%, 10%, ..., 95% của estimated time.
        candidate_cuts = np.unique(np.quantile(t, np.linspace(0.05, 0.95, 19)))
        best, self.low_, self.high_ = -1.0, candidate_cuts[0], candidate_cuts[-1]
        for i, a in enumerate(candidate_cuts):
            for b in candidate_cuts[i + 1:]:
                score = f1_score(y, self._level(t, a, b), average="macro")
                if score > best:
                    best, self.low_, self.high_ = score, a, b
        self.classes_ = np.array([0, 1, 2])           # sklearn yêu cầu estimator phân lớp có thuộc tính classes_
        return self

    @staticmethod
    def _level(t, a, b):
        return np.where(t < a, 0, np.where(t < b, 1, 2))

    def predict(self, X):
        return self._level(np.asarray(X, dtype=float).ravel(), self.low_, self.high_)


def make_models(features):
    """Trả về {tên: (estimator, các cột đầu vào, loại)}. Siêu tham số cố định từ trước, không tinh chỉnh theo kết quả."""
    return {
        "Majority class": (DummyClassifier(strategy="most_frequent"), features, "baseline"),
        "Estimated time rule": (TimeRule(), ["ref_est_time_s"], "baseline"),
        "Decision Tree": (DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0), features, "ml"),
        # np.log1p là hàm của numpy nên pipeline lưu và gửi sang tiến trình khác được (khác với hàm lambda).
        "Logistic Regression": (make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                              LogisticRegression(C=0.1, max_iter=5000)), features, "ml"),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                                 max_features="sqrt", random_state=0), features, "ml"),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2,
                                                         subsample=0.8, min_samples_leaf=3, random_state=0),
                              features, "ml"),
    }


def evaluate_fold(run_df, y, train_idx, test_idx, features, only_ml, train_fraction, importance, seed):
    """Huấn luyện và đánh giá mọi mô hình trên một fold. Trả về điểm, ma trận nhầm lẫn và độ quan trọng feature."""
    if train_fraction < 1.0:
        # Learning curve: chỉ giữ một phần các query của tập train (chọn ngẫu nhiên, cố định theo seed).
        rng = np.random.default_rng(seed)
        groups = run_df["query"].to_numpy()[train_idx]
        unique_groups = np.unique(groups)
        keep = rng.choice(unique_groups, size=max(2, int(len(unique_groups) * train_fraction)), replace=False)
        train_idx = train_idx[np.isin(groups, keep)]
    out = {"rows": [], "confusion_by_model": {}, "importance_by_model": {}}
    if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
        return out
    y_train, y_test = y[train_idx], y[test_idx]
    for name, (est, cols, kind) in make_models(features).items():
        if only_ml and kind != "ml":
            continue
        model = clone(est)
        model.fit(run_df.iloc[train_idx][cols], y_train)
        pred, pred_tr = model.predict(run_df.iloc[test_idx][cols]), model.predict(run_df.iloc[train_idx][cols])
        out["rows"].append({"model": name, "f1_test": f1_score(y_test, pred, average="macro"),
                            "qwk_test": cohen_kappa_score(y_test, pred, weights="quadratic"),
                            "acc_test": accuracy_score(y_test, pred),
                            "f1_train": f1_score(y_train, pred_tr, average="macro")})
        out["confusion_by_model"][name] = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        if importance and kind == "ml":
            pi = permutation_importance(model, run_df.iloc[test_idx][cols], y_test, scoring="f1_macro", n_repeats=3,
                                        random_state=seed)
            out["importance_by_model"][name] = pi.importances_mean
    return out


def split_group_folds_unstratified(groups, n_folds, seed):
    """Chia theo nhóm không phân tầng: xáo trộn danh sách query rồi phát lần lượt vào các fold."""
    unique_groups = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique_groups)
    fold_of = {q: i % n_folds for i, q in enumerate(unique_groups)}
    ids = np.array([fold_of[q] for q in groups])
    return [(np.where(ids != k)[0], np.where(ids == k)[0]) for k in range(n_folds)]


def cross_validate(run_df, y, features, n_repeats, n_folds, n_jobs, scenario, only_ml=False, train_fraction=1.0, importance=False,
                   shuffle=False, stratified=True):
    """Group K-Fold lặp lại (mặc định có phân tầng theo label); trả về (điểm từng fold, ma trận nhầm lẫn, độ quan trọng)."""
    groups = run_df["query"].to_numpy()
    tasks = []
    for rep in range(n_repeats):
        # Kịch bản kiểm tra rò rỉ: xáo trộn nhãn giữa các dòng; kết quả phải rơi về mức ngẫu nhiên.
        y_rep = np.random.default_rng(1000 + rep).permutation(y) if shuffle else y
        if stratified:
            splits = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rep).split(run_df, y_rep, groups)
        else:
            splits = split_group_folds_unstratified(groups, n_folds, rep)
        for fold, (train_idx, test_idx) in enumerate(splits):
            assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "mot query nam o ca tap train va tap test"
            tasks.append((rep, fold, train_idx, test_idx, y_rep))
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(run_df, y_rep, train_idx, test_idx, features, only_ml, train_fraction, importance, rep * 100 + fold)
        for rep, fold, train_idx, test_idx, y_rep in tasks)
    rows, confusion_by_model, importance_by_model = [], {}, {}
    for (rep, fold, _, _, _), res in zip(tasks, results):
        for r in res["rows"]:
            rows.append({"scenario": scenario, "rep": rep, "fold": fold, "fraction": train_fraction, **r})
        for name, m in res["confusion_by_model"].items():
            confusion_by_model[name] = confusion_by_model.get(name, 0) + m
        for name, v in res["importance_by_model"].items():
            importance_by_model.setdefault(name, []).append(v)
    return pd.DataFrame(rows), confusion_by_model, importance_by_model


def summarize_scores(scores):
    """Trung bình và độ lệch chuẩn (giữa các fold) của từng mô hình."""
    g = scores.groupby("model", sort=False)
    out = pd.DataFrame({
        "f1_test_mean": g["f1_test"].mean(), "f1_test_std": g["f1_test"].std(),
        "qwk_test_mean": g["qwk_test"].mean(), "qwk_test_std": g["qwk_test"].std(),
        "acc_test_mean": g["acc_test"].mean(), "acc_test_std": g["acc_test"].std(),
        "f1_train_mean": g["f1_train"].mean()})
    out["chenh_lech_train_test"] = out["f1_train_mean"] - out["f1_test_mean"]
    return out


def print_summary(title, summary):
    print(f"\n{title}")
    print(f"   {'Mo hinh':<22}{'F1 test':>14}{'QWK test':>12}{'Acc test':>10}{'F1 train':>10}{'Chenh':>8}")
    for name, r in summary.iterrows():
        print(f"   {name:<22}{r['f1_test_mean']:>8.3f} +-{r['f1_test_std']:.2f}{r['qwk_test_mean']:>10.3f}"
              f"{r['acc_test_mean']:>10.3f}{r['f1_train_mean']:>10.3f}{r['chenh_lech_train_test']:>8.2f}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--features", default=str(config.FEATURES_CLEAN_CSV))
    arg_parser.add_argument("--labels", default=str(config.LABELS_CSV))
    arg_parser.add_argument("--tab-dir", default=str(config.TAB_RESULTS))
    arg_parser.add_argument("--fig-dir", default=str(config.FIG_RESULTS))
    arg_parser.add_argument("--model-dir", default=str(config.MODELS_DIR))
    arg_parser.add_argument("--n-folds", type=int, default=5)
    arg_parser.add_argument("--n-repeats", type=int, default=10, help="so lan lap cross-validation")
    arg_parser.add_argument("--jobs", type=int, default=-1, help="so tien trinh chay song song (-1: tat ca)")
    arg_parser.add_argument("--final", default="auto", help="mo hinh luu cho app: auto (F1 test cao nhat) hoac ten mo hinh")
    args = arg_parser.parse_args()
    for p in (args.features, args.labels):
        if not Path(p).exists():
            sys.exit(f"Khong tim thay {p}. Chay cac buoc truoc (02_eda.py, 03_label_data.py).")
    tab_dir, fig_dir, model_dir = Path(args.tab_dir), Path(args.fig_dir), Path(args.model_dir)
    for d in (tab_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Xóa đầu ra của lần chạy trước để không còn file cũ lẫn với file mới.
    stale_tables = ["results_summary.csv", "results_without_peak.csv", "results_shuffled.csv", "results_unstratified.csv",
           "learning_curve.csv",
           "permutation_importance.csv", "fold_scores.csv", "confusion_matrices.csv"]
    stale_figures = ["01_so_sanh_mo_hinh.png", "02_ma_tran_nham_lan.png", "03_permutation_importance.png",
                "04_learning_curve.png"]
    for p in [tab_dir / n for n in stale_tables] + [fig_dir / n for n in stale_figures] + [model_dir / "model.pkl",
                                                                           model_dir / "run_info.json"]:
        p.unlink(missing_ok=True)

    features_df = pd.read_csv(args.features)
    labels_df = pd.read_csv(args.labels)
    # validate="one_to_one": báo lỗi nếu run_id bị lặp ở một trong hai bảng.
    run_df = features_df.merge(labels_df[["run_id", "label_ord", "peak_day"]], on="run_id", how="inner", validate="one_to_one")
    features = [c for c in common.FEATURE_COLS if c in run_df.columns]
    y = run_df["label_ord"].to_numpy()
    start_time = time.time()

    print("1. Du Lieu")
    print(f"   {len(run_df)} lan chay, {run_df['query'].nunique()} query, {len(features)} feature")
    print("   Label: " + ", ".join(f"{LABEL_LEVELS[i]} {int((y == i).sum())}" for i in range(3)))
    print(f"   Cross-validation: {args.n_folds} fold x {args.n_repeats} lan lap, chia theo query")

    # 2. Kịch bản chính: toàn bộ 64 lần chạy, kèm độ quan trọng feature
    scores, confusion_by_model, importance_by_model = cross_validate(run_df, y, features, args.n_repeats, args.n_folds, args.jobs, "tat_ca", importance=True)
    main_summary = summarize_scores(scores)
    main_summary.to_csv(tab_dir / "results_summary.csv")
    print_summary("2. Ket Qua (tat ca lan chay)", main_summary)

    # 3. Bỏ các lần chạy ngày cao điểm
    keep = ~run_df["peak_day"].to_numpy()
    run_df_no_peak = run_df[keep].reset_index(drop=True)
    scores_no_peak, _, _ = cross_validate(run_df_no_peak, y[keep], features, args.n_repeats, args.n_folds, args.jobs, "bo_ngay_cao_diem")
    summary_no_peak = summarize_scores(scores_no_peak)
    summary_no_peak.to_csv(tab_dir / "results_without_peak.csv")
    print_summary(f"3. Ket Qua (bo {int((~keep).sum())} lan chay ngay cao diem)", summary_no_peak)

    # 4. Kiểm tra rò rỉ: xáo trộn nhãn
    scores_shuffled, _, _ = cross_validate(run_df, y, features, args.n_repeats, args.n_folds, args.jobs, "nhan_xao_tron", shuffle=True)
    summary_shuffled = summarize_scores(scores_shuffled)
    summary_shuffled.to_csv(tab_dir / "results_shuffled.csv")
    print_summary("4. Kiem Tra Ro Ri (nhan bi xao tron, phai gan muc ngau nhien)", summary_shuffled)

    # 5. Độ nhạy với cách chia fold: chia theo nhóm nhưng không phân tầng theo label
    scores_unstratified, _, _ = cross_validate(run_df, y, features, args.n_repeats, args.n_folds, args.jobs, "khong_phan_tang",
                                     stratified=False)
    summary_unstratified = summarize_scores(scores_unstratified)
    summary_unstratified.to_csv(tab_dir / "results_unstratified.csv")
    print_summary("5. Ket Qua Khi Chia Fold Khong Phan Tang", summary_unstratified)

    # 6. Learning curve: chỉ giữ một phần số query của tập train
    curve_rows = []
    for train_fraction in LEARNING_FRACTIONS:
        s, _, _ = cross_validate(run_df, y, features, args.n_repeats, args.n_folds, args.jobs, "learning_curve",
                                 only_ml=True, train_fraction=train_fraction)
        g = s.groupby("model")["f1_test"]
        for name in ML_MODELS:
            curve_rows.append({"model": name, "fraction": train_fraction, "f1_test_mean": g.mean()[name],
                               "f1_test_std": g.std()[name]})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(tab_dir / "learning_curve.csv", index=False)
    print("\n6. Learning Curve (F1 test theo ti le query dung de huan luyen)")
    for name in ML_MODELS:
        c = curve[curve["model"] == name]
        print(f"   {name:<22}" + "  ".join(f"{int(r.fraction * 100)}%: {r.f1_test_mean:.2f}" for r in c.itertuples()))

    # 7. Bảng chi tiết
    pd.concat([scores, scores_no_peak, scores_shuffled, scores_unstratified]).to_csv(tab_dir / "fold_scores.csv", index=False)
    conf_rows = [{"model": n, "that": LABEL_LEVELS[i], "du_doan": LABEL_LEVELS[j], "so_luong": int(m[i, j])}
                 for n, m in confusion_by_model.items() for i in range(3) for j in range(3)]
    pd.DataFrame(conf_rows).to_csv(tab_dir / "confusion_matrices.csv", index=False)
    imp_rows = []
    for name, arrs in importance_by_model.items():
        a = np.vstack(arrs)                                   # mỗi dòng là một fold, mỗi cột là một feature
        for k, f in enumerate(features):
            imp_rows.append({"model": name, "feature": f, "importance_mean": a[:, k].mean(),
                             "importance_std": a[:, k].std()})
    importance_df = pd.DataFrame(imp_rows)
    importance_df.to_csv(tab_dir / "permutation_importance.csv", index=False)

    # 8. Hình
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    order = list(main_summary.index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pos = np.arange(len(order))[::-1]
    ax.barh(pos, main_summary["f1_test_mean"], xerr=main_summary["f1_test_std"], color=COLOR_PALETTE["main"],
            error_kw={"ecolor": COLOR_PALETTE["grey"], "capsize": 3})
    ax.axvline(summary_shuffled.loc[ML_MODELS, "f1_test_mean"].mean(), color=COLOR_PALETTE["line"], ls="--",
               label="Nhãn bị xáo trộn (mức ngẫu nhiên)")
    ax.set_yticks(pos); ax.set_yticklabels(order); ax.set_xlabel("Macro-F1 trên tập test (trung bình ± độ lệch chuẩn giữa các fold)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(fig_dir / "01_so_sanh_mo_hinh.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for a, name in zip(axes.ravel(), ML_MODELS):
        m = confusion_by_model[name].astype(float)
        pct = m / m.sum(axis=1, keepdims=True) * 100          # chuẩn hóa theo dòng: tỉ lệ nhận đúng từng lớp thật
        a.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center",
                       color="white" if pct[i, j] > 55 else "black")
        a.set_xticks(range(3)); a.set_xticklabels(LABEL_LEVELS, fontsize=8)
        a.set_yticks(range(3)); a.set_yticklabels(LABEL_LEVELS, fontsize=8)
        a.set_title(name, fontsize=10); a.set_xlabel("Dự báo"); a.set_ylabel("Thực tế")
    fig.tight_layout(); fig.savefig(fig_dir / "02_ma_tran_nham_lan.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for a, name in zip(axes.ravel(), ML_MODELS):
        d = importance_df[importance_df["model"] == name].sort_values("importance_mean", ascending=False).head(8)[::-1]
        a.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=COLOR_PALETTE["main"],
               error_kw={"ecolor": COLOR_PALETTE["grey"], "capsize": 2})
        a.set_title(name, fontsize=10); a.tick_params(axis="y", labelsize=8)
        a.set_xlabel("Giảm macro-F1 khi hoán vị feature")
    fig.tight_layout(); fig.savefig(fig_dir / "03_permutation_importance.png", dpi=150); plt.close(fig)

    fig, a = plt.subplots(figsize=(6.4, 4.2))
    for name, col in zip(ML_MODELS, [COLOR_PALETTE["grey"], "#4C9F70", COLOR_PALETTE["main"], COLOR_PALETTE["line"]]):
        c = curve[curve["model"] == name]
        a.plot(c["fraction"] * 100, c["f1_test_mean"], marker="o", color=col, label=name)
        a.fill_between(c["fraction"] * 100, c["f1_test_mean"] - c["f1_test_std"] / 2,
                       c["f1_test_mean"] + c["f1_test_std"] / 2, color=col, alpha=0.08)
    a.axhline(main_summary.loc["Estimated time rule", "f1_test_mean"], color=COLOR_PALETTE["grey"], ls="--", lw=1,
              label="Estimated time rule")
    a.set_xlabel("Tỉ lệ query dùng để huấn luyện (%)"); a.set_ylabel("Macro-F1 trên tập test")
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "04_learning_curve.png", dpi=150); plt.close(fig)

    # 9. Mô hình cuối cho app: huấn luyện lại trên toàn bộ dữ liệu
    final = main_summary.loc[ML_MODELS, "f1_test_mean"].idxmax() if args.final == "auto" else args.final
    est, cols, _ = make_models(features)[final]
    est = clone(est).fit(run_df[cols], y)
    joblib.dump({"model": est, "features": features, "levels": LABEL_LEVELS, "name": final}, model_dir / "model.pkl")
    info = {"trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "final_model": final,
            "selection": "F1 test cao nhat trong cross-validation" if args.final == "auto" else "chon thu cong",
            "n_runs": len(run_df), "n_queries": int(run_df["query"].nunique()), "features": features,
            "label_counts": {LABEL_LEVELS[i]: int((y == i).sum()) for i in range(3)},
            "cv": {"n_folds": args.n_folds, "n_repeats": args.n_repeats, "splitter": "StratifiedGroupKFold (nhom = query)"},
            "cv_f1_test_mean": round(float(main_summary.loc[final, "f1_test_mean"]), 4),
            "cv_f1_test_std": round(float(main_summary.loc[final, "f1_test_std"]), 4)}
    (model_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n7. Mo Hinh Cho App: {final} (F1 test cross-validation {info['cv_f1_test_mean']:.3f}); luu o {model_dir / 'model.pkl'}")
    print(f"   Thoi gian chay: {time.time() - start_time:.0f} giay")


if __name__ == "__main__":
    main()
