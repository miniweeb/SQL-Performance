# -*- coding: utf-8 -*-
"""
Bước 1: Trích xuất feature từ các folder lần chạy trong data/raw/.

Đầu vào mỗi folder: plan_estimated.txt, plan_allstats.txt, summary.json.
Đầu ra (data/interim/): features.csv, check_report.csv, errors.csv (chỉ có khi có folder lỗi).
Các cột tiền tố post_ là số liệu sau khi chạy, dùng để gán label và không dùng làm feature.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import common
import config

ID_COLS = ["run_id", "query", "version", "run_date"]
POST_COLS = ["post_elapsed_s", "post_cpu_s", "post_wall_s", "post_buffer_gets", "post_disk_reads",
             "post_read_bytes", "post_sorts_disk", "post_rows_returned"]


def read_run(folder):
    """Đọc một folder lần chạy; trả về (dòng của features.csv, dòng của check_report.csv)."""
    summary = common.read_summary(folder / "summary.json")        # toán tử / nối đường dẫn của pathlib
    est_text = (folder / "plan_estimated.txt").read_text(encoding="utf-8")
    act_text = (folder / "plan_allstats.txt").read_text(encoding="utf-8")

    feats = common.compute_features(est_text, bind_range_days=summary["bind_range_days"])
    est_rows = common.parse_plan_table(est_text)
    act_rows = common.parse_plan_table(act_text)

    # {k: summary[k] for k in ...} tạo dict mới chỉ gồm các khóa trong danh sách; update() gộp thêm vào dict.
    feature_row = {k: summary[k] for k in ID_COLS}
    feature_row.update(feats)
    feature_row.update({k: summary[k] for k in POST_COLS})

    check_row = {
        "run_id": summary["run_id"], "query": summary["query"], "version": summary["version"],
        "plan_hash_khop": summary["estimated_plan_hash"] == summary["actual_plan_hash"],
        "n_steps_est": len(est_rows), "n_steps_act": len(act_rows),
        # sum() trên chuỗi True/False: True tính là 1 nên kết quả là số bước thỏa điều kiện.
        "n_px_est": sum(r["op"].startswith("PX") for r in est_rows),
        "n_px_act": sum(r["op"].startswith("PX") for r in act_rows),
        "structure": common.compare_structure(est_rows, act_rows),
    }
    return feature_row, check_row


def main():
    ap = argparse.ArgumentParser()
    # Tham số "--out-dir" được đọc ở args.out_dir (argparse đổi dấu - thành _).
    ap.add_argument("--raw", default=str(config.RAW_DIR), help="thu muc chua cac folder lan chay")
    ap.add_argument("--out-dir", default=str(config.INTERIM_DIR), help="thu muc ghi ket qua")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in raw.iterdir() if p.is_dir())
    if not folders:
        sys.exit(f"Khong co folder lan chay nao trong {raw}")    # sys.exit(chuỗi): in chuỗi ra stderr rồi thoát với mã lỗi 1

    feature_rows, check_rows, errors = [], [], []
    for folder in folders:
        try:
            f_row, c_row = read_run(folder)
        except Exception as exc:                  # một folder lỗi không làm dừng cả lần chạy; lỗi được ghi lại
            errors.append({"folder": folder.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        feature_rows.append(f_row)
        check_rows.append(c_row)

    # Tham số columns cố định thứ tự cột: định danh, feature, rồi số liệu sau khi chạy.
    features = pd.DataFrame(feature_rows, columns=ID_COLS + common.FEATURE_COLS + POST_COLS)
    check = pd.DataFrame(check_rows)
    features.to_csv(out / "features.csv", index=False)            # index=False: không ghi cột số thứ tự
    check.to_csv(out / "check_report.csv", index=False)
    errors_path = out / "errors.csv"
    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False)
    elif errors_path.exists():
        errors_path.unlink()                      # xóa errors.csv của lần chạy trước để file không còn lỗi cũ

    print("1. Ket Qua Doc Du Lieu")
    print(f"   Folder lan chay : {len(folders)}")
    print(f"   Doc thanh cong  : {len(features)}")
    print(f"   Loi             : {len(errors)}")
    print("\n2. Du Lieu Dau Ra")
    print(f"   features.csv     : {features.shape[0]} dong, {features.shape[1]} cot "
          f"({len(ID_COLS)} dinh danh, {len(common.FEATURE_COLS)} feature, {len(POST_COLS)} so lieu sau khi chay)")
    print(f"   Query goc        : {features['query'].nunique()}")
    print(f"   Phien ban        : {sorted(features['version'].unique().tolist())}")
    # isna() đánh dấu ô thiếu; sum() lần đầu cộng theo cột, sum() lần hai cộng tất cả các cột.
    print(f"   Gia tri thieu    : {int(features[common.FEATURE_COLS].isna().sum().sum())} (trong cac feature)")
    print("\n3. Doi Chieu Estimated Plan Va Actual Plan")
    for name, n in check["structure"].value_counts().items():
        print(f"   {name:<11}: {n}")                # :<11 căn trái trong 11 ký tự cho thẳng cột
    print(f"   Plan hash khop  : {int(check['plan_hash_khop'].sum())} / {len(check)}")
    print(f"\n   Thu muc: {out.resolve()}")
    if errors:
        print("\n4. Loi (xem errors.csv)")
        for e in errors[:5]:                      # [:5] chỉ in 5 lỗi đầu tiên
            print(f"   {e['folder']}: {e['error']}")


# Khối này chỉ chạy khi chạy trực tiếp file (python src/01_extract_features.py), không chạy khi file được import.
if __name__ == "__main__":
    main()
