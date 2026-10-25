# -*- coding: utf-8 -*-
"""
Bước 2: Tiền xử lý và EDA trên data/interim/features.csv.

Việc thực hiện: điền giá trị thiếu, loại feature hằng và feature hiếm, phát hiện outlier (chỉ báo cáo, không sửa giá trị),
thống kê phân phối, so sánh estimated time với elapsed thực tế, tính Spearman giữa các feature và loại cặp trùng lặp.
Đầu ra: data/interim/features_clean.csv, bảng trong reports/tables/eda/, hình trong reports/figures/eda/.
Chạy lại sẽ xóa và ghi đè các file đầu ra của bước này.
Cột ref_est_time_s là estimated time gốc, giữ lại cho baseline dù est_time_s đã bị loại khỏi danh sách feature.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # vẽ ra file, không mở cửa sổ (chạy được trong notebook và terminal)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config

ID_COLS = ["run_id", "query", "version", "run_date"]
POST_COLS = ["post_elapsed_s", "post_cpu_s", "post_wall_s", "post_buffer_gets", "post_disk_reads",
             "post_read_bytes", "post_sorts_disk", "post_rows_returned"]
IQR_K = 1.5                               # hàng rào IQR: ngoài [Q1 - 1,5 IQR, Q3 + 1,5 IQR] là outlier
RARE_MIN_RUNS = 5                         # feature có ít hơn 5 lần chạy khác giá trị phổ biến nhất thì bị loại
DISCRETE_MAX = 4                          # feature có tối đa 4 giá trị khác nhau là rời rạc, không xét outlier
CORR_LIMIT = 0.90                         # cặp feature có |Spearman| từ ngưỡng này thì bỏ một feature
COLOR = {"main": "#2F5D8A", "line": "#C0504D", "grey": "#8C8C8C"}


def outlier_table(df, cols):
    """Thống kê outlier theo quy tắc IQR cho từng feature; không sửa giá trị."""
    rows = []
    for c in cols:
        x = df[c].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        n_unique = x.nunique()
        # Cờ 0/1 và feature chỉ có vài giá trị hợp lệ không có outlier: IQR của cờ bằng 0 nên mọi giá trị khác
        # trung vị đều bị coi là outlier, còn giá trị hiếm của feature rời rạc là giá trị bình thường.
        kind = "co_nhi_phan" if n_unique <= 2 else ("roi_rac" if n_unique <= DISCRETE_MAX else "lien_tuc")
        # Hàng rào IQR chỉ có nghĩa với feature liên tục; cờ và feature rời rạc để trống.
        low, high = (q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)) if kind == "lien_tuc" else (np.nan, np.nan)
        n_iqr = int(((x < low) | (x > high)).sum()) if kind == "lien_tuc" else 0
        rows.append({"feature": c, "loai": kind, "skew": x.skew(), "outlier_iqr": n_iqr,
                     "hang_rao_duoi": low, "hang_rao_tren": high})
    return pd.DataFrame(rows).set_index("feature")


def drop_correlated(df, cols):
    """Duyệt feature theo thứ tự; bỏ feature có |Spearman| từ CORR_LIMIT với một feature đã giữ."""
    rho = df[cols].corr(method="spearman")
    kept, dropped = [], []
    for c in cols:
        hit = [k for k in kept if abs(rho.loc[c, k]) >= CORR_LIMIT]
        if hit:
            partner = max(hit, key=lambda k: abs(rho.loc[c, k]))      # max(..., key=hàm): phần tử có giá trị hàm lớn nhất
            dropped.append({"bo": c, "giu": partner, "spearman": rho.loc[c, partner]})
        else:
            kept.append(c)
    return rho, kept, pd.DataFrame(dropped, columns=["bo", "giu", "spearman"])


def percentile_table(f):
    """Phân phối các chỉ số sau khi chạy: nhỏ nhất, P10, P25, trung vị, P75, P90, lớn nhất."""
    metrics = {
        "elapsed (phut)": f["post_elapsed_s"] / 60, "cpu (phut)": f["post_cpu_s"] / 60,
        "wall (phut)": f["post_wall_s"] / 60, "doc dia (GB)": f["post_read_bytes"] / 2 ** 30,
        "buffer gets (trieu)": f["post_buffer_gets"] / 1e6, "so dong tra ve (trieu)": f["post_rows_returned"] / 1e6,
    }
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = {k: [v.min()] + [v.quantile(q) for q in qs] + [v.max()] for k, v in metrics.items()}
    # Mỗi khóa của dict là một cột nên bảng ra có chỉ số ở cột; .T chuyển vị để mỗi chỉ số là một dòng.
    return pd.DataFrame(rows, index=["min", "P10", "P25", "P50", "P75", "P90", "max"]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.FEATURES_CSV))
    ap.add_argument("--clean", default=str(config.FEATURES_CLEAN_CSV))
    ap.add_argument("--tab-dir", default=str(config.TAB_EDA))
    ap.add_argument("--fig-dir", default=str(config.FIG_EDA))
    args = ap.parse_args()
    tab_dir, fig_dir, clean_path = Path(args.tab_dir), Path(args.fig_dir), Path(args.clean)
    for d in (tab_dir, fig_dir, clean_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    # Xóa đầu ra của lần chạy trước để không còn file cũ lẫn với file mới (missing_ok: không báo lỗi nếu chưa có file).
    old_figs = ["01_phan_phoi_elapsed_cpu.png", "02_time_uoc_luong_vs_thuc_te.png", "03_tuong_quan_dac_trung.png",
                "02_elapsed_vs_cpu.png", "03_time_uoc_luong_vs_thuc_te.png", "04_tuong_quan_dac_trung.png",
                "05_outlier_truoc_sau.png"]                      # gồm cả tên hình của các phiên bản trước
    old_tabs = ["outliers.csv", "percentiles.csv", "spearman_voi_elapsed.csv", "feature_filter.csv",
                "est_vs_actual.csv", "correlation_pairs.csv"]
    for p in [clean_path] + [tab_dir / n for n in old_tabs] + [fig_dir / n for n in old_figs]:
        p.unlink(missing_ok=True)

    f = pd.read_csv(args.features)
    feats = common.FEATURE_COLS

    # 1. Giá trị thiếu và cột hằng
    n_missing = int(f[feats].isna().sum().sum())
    f[feats] = f[feats].fillna(f[feats].median())             # điền trung vị theo từng cột
    # value_counts() đếm số lần mỗi giá trị xuất hiện, sắp giảm dần; iloc[0] là số lần của giá trị phổ biến nhất.
    # Số lần chạy khác giá trị phổ biến nhất = tổng số dòng trừ số lần của giá trị đó.
    n_other = {c: len(f) - int(f[c].value_counts().iloc[0]) for c in feats}
    filter_rows = [{"feature": c, "ly_do": "hang_so" if n_other[c] == 0 else "hiem",
                    "so_lan_chay_khac_mode": n_other[c], "feature_giu": "", "spearman": np.nan}
                   for c in feats if n_other[c] < RARE_MIN_RUNS]
    constant = [r["feature"] for r in filter_rows if r["ly_do"] == "hang_so"]
    rare = [r["feature"] for r in filter_rows if r["ly_do"] == "hiem"]
    active = [c for c in feats if n_other[c] >= RARE_MIN_RUNS]

    # 2. Outlier: chỉ báo cáo; các query lớn là giá trị thật nên giữ nguyên
    outliers = outlier_table(f, active)
    outliers.to_csv(tab_dir / "outliers.csv")

    # 3. Phân phối và so sánh estimated time với elapsed
    pct = percentile_table(f)
    pct.to_csv(tab_dir / "percentiles.csv")
    elapsed = f["post_elapsed_s"]
    est_actual = pd.DataFrame({
        "spearman_voi_elapsed": [f["est_time_s"].corr(elapsed, method="spearman"),
                                 f["est_cost"].corr(elapsed, method="spearman"),
                                 f["post_cpu_s"].corr(elapsed, method="spearman")],
        "trung_vi_ty_le_uoc_luong_tren_thuc_te": [(f["est_time_s"] / elapsed).median(), np.nan, np.nan],
    }, index=["est_time_s", "est_cost", "post_cpu_s"])
    est_actual.to_csv(tab_dir / "spearman_voi_elapsed.csv")

    # 4. Tương quan giữa các feature và loại feature trùng lặp
    rho, kept, dropped = drop_correlated(f, active)
    for _, r in dropped.iterrows():
        filter_rows.append({"feature": r["bo"], "ly_do": "trung_lap", "so_lan_chay_khac_mode": np.nan,
                            "feature_giu": r["giu"], "spearman": r["spearman"]})
    pd.DataFrame(filter_rows).to_csv(tab_dir / "feature_filter.csv", index=False)

    # 5. Ghi dữ liệu đã làm sạch: định danh, feature giữ lại, số liệu sau khi chạy
    # pd.concat(axis=1) ghép các bảng cạnh nhau theo cột (cùng thứ tự dòng); rename đổi tên cột est_time_s.
    ref = f[["est_time_s"]].rename(columns={"est_time_s": "ref_est_time_s"})
    out = pd.concat([f[ID_COLS], f[kept], ref, f[POST_COLS]], axis=1)
    out.to_csv(clean_path, index=False)

    # 6. Hình
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    el_min, cpu_min = elapsed / 60, f["post_cpu_s"] / 60

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, v, title in ((ax[0], el_min, "Elapsed time"), (ax[1], cpu_min, "CPU time")):
        a.hist(v, bins=16, color=COLOR["main"], edgecolor="white")
        a.axvline(v.median(), color=COLOR["line"], ls="--", label=f"Trung vị {v.median():.1f} phút")
        a.set_xlabel("phút"); a.set_ylabel("số lần chạy"); a.set_title(title); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "01_phan_phoi_elapsed_cpu.png", dpi=150); plt.close(fig)

    fig, a = plt.subplots(figsize=(5.2, 4.6))
    est_min = f["est_time_s"] / 60
    a.scatter(est_min, el_min, s=26, color=COLOR["main"], alpha=0.8)
    lim = [min(est_min.min(), el_min.min()) * 0.8, max(est_min.max(), el_min.max()) * 1.2]
    a.plot(lim, lim, color=COLOR["grey"], ls="--", lw=1, label="Ước lượng = thực tế")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Estimated time (phút, thang log)"); a.set_ylabel("Elapsed time thực tế (phút, thang log)")
    a.set_title(f"Spearman = {est_actual.loc['est_time_s', 'spearman_voi_elapsed']:.2f}"); a.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "02_time_uoc_luong_vs_thuc_te.png", dpi=150); plt.close(fig)

    fig, a = plt.subplots(figsize=(9, 7.6))
    im = a.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(active))); a.set_xticklabels(active, rotation=90, fontsize=7)
    a.set_yticks(range(len(active))); a.set_yticklabels(active, fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="Spearman")
    a.set_title("Tương quan Spearman giữa các feature")
    fig.tight_layout(); fig.savefig(fig_dir / "03_tuong_quan_dac_trung.png", dpi=150); plt.close(fig)

    # 7. Tóm tắt
    print("1. Kiem Tra Du Lieu")
    print(f"   Dong x cot        : {f.shape[0]} x {f.shape[1]}")
    print(f"   Gia tri thieu     : {n_missing} (da dien trung vi)")
    print(f"   Feature hang so   : {constant if constant else 'khong co'}")
    print(f"   Feature hiem (< {RARE_MIN_RUNS} lan chay khac gia tri pho bien nhat): {rare if rare else 'khong co'}")
    print("\n2. Outlier")
    cont = outliers[outliers["loai"] == "lien_tuc"]
    cont_n = len(cont)
    kinds = outliers["loai"].value_counts()
    print(f"   Feature lien tuc  : {cont_n}; roi rac: {kinds.get('roi_rac', 0)}; nhi phan: {kinds.get('co_nhi_phan', 0)}")
    print(f"   Feature co outlier IQR: {int((cont['outlier_iqr'] > 0).sum())} / {cont_n}")
    print(f"   Feature lech phai (skew > 2): {int((cont['skew'] > 2).sum())} / {cont_n}")
    print(f"   So gia tri outlier (IQR): {int(cont['outlier_iqr'].sum())} (chi bao cao, khong sua)")
    print("\n3. Phan Phoi (phut)")
    for k in ("elapsed (phut)", "cpu (phut)"):
        r = pct.loc[k]
        print(f"   {k:<15}: min {r['min']:.1f}, P50 {r['P50']:.1f}, P90 {r['P90']:.1f}, max {r['max']:.1f}")
    print("\n4. Estimated Time So Voi Elapsed")
    print(f"   Spearman est_time_s : {est_actual.loc['est_time_s', 'spearman_voi_elapsed']:.2f}")
    print(f"   Spearman est_cost   : {est_actual.loc['est_cost', 'spearman_voi_elapsed']:.2f}")
    print(f"   Spearman CPU        : {est_actual.loc['post_cpu_s', 'spearman_voi_elapsed']:.2f}")
    print(f"   Trung vi (est_time / elapsed): {est_actual.loc['est_time_s', 'trung_vi_ty_le_uoc_luong_tren_thuc_te']:.2f}")
    print("\n5. Feature Trung Lap (|Spearman| >= %.2f)" % CORR_LIMIT)
    for _, r in dropped.iterrows():
        print(f"   bo {r['bo']:<18} giu {r['giu']:<18} rho {r['spearman']:+.2f}")
    print(f"\n6. Ket Qua: {len(kept)} feature giu lai / {len(feats)}")
    print(f"   {clean_path}  ({out.shape[0]} dong, {out.shape[1]} cot)")


if __name__ == "__main__":
    main()
