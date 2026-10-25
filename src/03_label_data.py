# -*- coding: utf-8 -*-
"""
Bước 3: Gán label theo quy tắc trong docs/LABEL_RULES.md và đo độ nhạy của ngưỡng.

Đầu vào: data/interim/features_clean.csv (cột post_elapsed_s, post_sorts_disk).
Đầu ra (data/labeled/): labels.csv (label và quy tắc quyết định của từng lần chạy),
label_sensitivity.csv (số lần chạy đổi label khi dời ngưỡng).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

LABEL_LEVELS = ["Normal", "Warning", "Critical"]
WARNING_MIN, CRITICAL_MIN = 11.0, 21.0           # ngưỡng elapsed (phút); phải khớp với bảng ngưỡng trong docs/LABEL_RULES.md


def assign(elapsed_min, spill, warn, crit):
    """Trả về (mức 0/1/2 theo từng lần chạy, cờ lần chạy được nâng bậc vì tràn đĩa)."""
    # np.where(điều kiện, a, b) chọn a hoặc b theo từng phần tử; lồng hai lần để chia ba mức.
    level = np.where(elapsed_min < warn, 0, np.where(elapsed_min < crit, 1, 2))
    # & là "và" theo từng phần tử của mảng; từ khóa and không dùng được cho mảng.
    raised = (spill > 0) & (level < 2)
    return np.where(raised, level + 1, level), raised


def sensitivity_table(elapsed, spill, peak, base_level, warn, crit):
    """Đếm số lần chạy đổi label khi dời từng ngưỡng ±10% và khi giả định tải ngày cao điểm gấp đôi."""
    elapsed_halved = np.where(peak, elapsed / 2, elapsed)      # tải gấp đôi thì elapsed ngày cao điểm giảm một nửa
    scenarios = [
        ("Warning -10%", elapsed, warn * 0.9, crit, len(elapsed)),
        ("Warning +10%", elapsed, warn * 1.1, crit, len(elapsed)),
        ("Critical -10%", elapsed, warn, crit * 0.9, len(elapsed)),
        ("Critical +10%", elapsed, warn, crit * 1.1, len(elapsed)),
        ("Ngay cao diem, tai gap doi", elapsed_halved, warn, crit, int(peak.sum())),
    ]
    rows = []
    for name, e, w, c, total in scenarios:
        level, _ = assign(e, spill, w, c)
        changed = int((level != base_level).sum())
        counts = np.bincount(level, minlength=3)
        rows.append({"kich_ban": name, "nguong_warning_phut": round(w, 2), "nguong_critical_phut": round(c, 2),
                     "so_ban_ghi_doi_label": changed, "tong_so_ban_ghi_xet": total,
                     "ty_le_doi_pct": round(changed / total * 100, 1),
                     "normal": int(counts[0]), "warning": int(counts[1]), "critical": int(counts[2])})
    return pd.DataFrame(rows).set_index("kich_ban")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--features", default=str(config.FEATURES_CLEAN_CSV))
    arg_parser.add_argument("--labels", default=str(config.LABELS_CSV))
    arg_parser.add_argument("--sensitivity", default=str(config.SENSITIVITY_CSV))
    arg_parser.add_argument("--warn", type=float, default=WARNING_MIN, help="nguong Warning (phut)")
    arg_parser.add_argument("--crit", type=float, default=CRITICAL_MIN, help="nguong Critical (phut)")
    args = arg_parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        sys.exit(f"Khong tim thay {features_path}. Chay 02_eda.py truoc.")
    labels_path, sens_path = Path(args.labels), Path(args.sensitivity)
    for p in (labels_path, sens_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)              # xóa đầu ra của lần chạy trước (không báo lỗi nếu chưa có)

    run_df = pd.read_csv(features_path)
    elapsed = run_df["post_elapsed_s"].to_numpy() / 60
    spill = run_df["post_sorts_disk"].to_numpy()
    peak = (run_df["run_date"] == config.PEAK_DAY).to_numpy()

    level, raised = assign(elapsed, spill, args.warn, args.crit)
    labels = pd.DataFrame({
        "run_id": run_df["run_id"], "query": run_df["query"], "version": run_df["version"], "run_date": run_df["run_date"],
        "elapsed_min": elapsed.round(3), "sorts_disk": spill,
        "label": [LABEL_LEVELS[i] for i in level], "label_ord": level,
        "rule": np.where(raised, "elapsed+spill", "elapsed"), "peak_day": peak,
    })
    labels.to_csv(labels_path, index=False)

    sensitivity_df = sensitivity_table(elapsed, spill, peak, level, args.warn, args.crit)
    sensitivity_df.to_csv(sens_path)

    # Đối chiếu: nếu chỉ dùng elapsed (không nâng bậc) thì phân phối label thế nào.
    only_elapsed, _ = assign(elapsed, np.zeros_like(spill), args.warn, args.crit)
    by_query = labels.groupby("query").agg(runs=("label", "size"), labels=("label", "nunique"))
    multi = by_query[by_query["runs"] > 1]

    print("1. Nguong Su Dung")
    print(f"   Warning tu {args.warn:g} phut; Critical tu {args.crit:g} phut; tran dia (sorts disk > 0) nang mot bac")
    print("\n2. Phan Phoi Label")
    counts = labels["label"].value_counts().reindex(LABEL_LEVELS)
    for name, n in counts.items():
        print(f"   {name:<9}: {n:>3} ({n / len(labels) * 100:.0f}%)")
    only_counts = np.bincount(only_elapsed, minlength=3)
    print(f"   Neu chi dung elapsed: Normal {only_counts[0]}, Warning {only_counts[1]}, Critical {only_counts[2]}")
    print("\n3. Quy Tac Quyet Dinh")
    print(f"   Lan chay tran dia      : {int((spill > 0).sum())}")
    print(f"   Duoc nang bac (doi nhan): {int(raised.sum())}")
    critical = labels["label"] == "Critical"
    print(f"   Critical: {int(critical.sum())} (thuan elapsed {int((critical & ~raised).sum())}, "
          f"nho nang bac {int((critical & raised).sum())})")
    print("\n4. Do Nhay")
    for name, r in sensitivity_df.iterrows():
        print(f"   {name:<28}: {int(r['so_ban_ghi_doi_label']):>2} / {int(r['tong_so_ban_ghi_xet']):<3} doi label "
              f"({r['ty_le_doi_pct']:.1f}%)")
    print("\n5. Cung Mot Query")
    print(f"   Query co nhieu lan chay: {len(multi)}; trong do co label khac nhau giua cac phien ban: "
          f"{int((multi['labels'] > 1).sum())}")
    print(f"   Lan chay ngay cao diem : {int(peak.sum())}")
    print(f"\n   {labels_path}  ({len(labels)} dong)")


if __name__ == "__main__":
    main()
