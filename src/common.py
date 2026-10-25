# -*- coding: utf-8 -*-
"""
Hàm dùng chung cho trích xuất feature, huấn luyện và Streamlit app: đọc bảng execution plan,
tính 30 feature, đọc summary.json, so sánh cấu trúc estimated plan với actual plan.
"""
import json
import re
from datetime import datetime

# 30 feature. Thứ tự trong danh sách là thứ tự cột của features.csv.
FEATURE_COLS = [
    "est_cost", "est_cpu_pct", "est_time_s",
    "est_rows_root", "est_bytes_root", "est_rows_max", "est_bytes_max", "est_bytes_table_access",
    "est_temp_total", "has_temp",
    "n_steps", "max_depth", "n_objects",
    "n_full_scan", "n_table_full_scan", "n_index_access", "n_hash_join", "n_nested_loops", "n_merge_join",
    "n_sort", "n_group_by", "n_view", "n_filter",
    "n_partition_ops", "n_partition_all",
    "n_px",
    "n_hints", "note_dynamic_sampling", "note_adaptive", "bind_range_days",
]

# Nhận dòng dữ liệu của bảng plan, ví dụ "|   3 | ..." hoặc "|*  4 | ...".
#   ^\|      dòng bắt đầu bằng dấu |
#   [ *]     một khoảng trắng hoặc một dấu * (dấu * đánh dấu bước có điều kiện lọc)
#   \s*\d+   khoảng trắng rồi các chữ số của Id
#   " \|"    khoảng trắng và dấu | đóng ô Id
# re.compile biên dịch mẫu một lần để dùng lại cho nhiều dòng.
ROW_RE = re.compile(r"^\|[ *]\s*\d+ \|")
# Hậu tố rút gọn số của DBMS_XPLAN: "5304K" là 5304 x 1000.
UNIT = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_number(text):
    """Đổi số kiểu DBMS_XPLAN ("5304K", "1686M", "132") về số thực; ô trống thì trả None."""
    # Nhóm 1 ([\d.]+) là phần số, nhóm 2 ([KMGT]?) là hậu tố (có thể rỗng).
    m = re.match(r"^([\d.]+)([KMGT]?)$", text.replace(",", "").strip())
    if not m:
        return None
    return float(m.group(1)) * UNIT[m.group(2)]


def parse_time(text):
    """Đổi "HH:MM:SS" về số giây."""
    h, m, s = text.strip().split(":")          # tách chuỗi theo dấu ":" rồi gán lần lượt vào 3 biến
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_plan_table(text):
    """Đọc bảng plan thành danh sách dict, mỗi dict là một bước (một Id).

    Cột được lấy theo TÊN ở dòng tiêu đề chứ không theo vị trí: bảng estimated plan và actual plan có cột
    khác nhau, và các cột TempSpc, Pstart, Pstop chỉ xuất hiện khi plan có dùng.
    Mỗi dict có thêm "op" (tên operation đã bỏ khoảng trắng), "depth" (độ sâu của bước) và "name".
    """
    lines = text.splitlines()
    # next(biểu thức generator, giá trị mặc định): lấy dòng đầu tiên thỏa điều kiện, không có thì trả None.
    header = next((ln for ln in lines if ln.startswith("| Id")), None)
    if header is None:
        raise ValueError("khong tim thay dong tieu de cua bang plan")
    cols = [c.strip() for c in header.strip("|").split("|")]
    rows = []
    for ln in lines:
        if not ROW_RE.match(ln):
            continue
        cells = ln.strip("|").split("|")
        # zip ghép tên cột với giá trị cùng vị trí; dict() biến các cặp thành {tên cột: giá trị}.
        row = dict(zip(cols, cells))
        raw_op = row["Operation"]
        row["op"] = raw_op.strip()
        # Operation được thụt đầu dòng theo độ sâu của bước trong cây plan, nên số khoảng trắng đầu dòng
        # (độ dài chuỗi trừ độ dài sau khi bỏ khoảng trắng bên trái) chính là độ sâu.
        row["depth"] = len(raw_op) - len(raw_op.lstrip())
        row["name"] = row.get("Name", "").strip()     # .get(khóa, mặc định): dòng không có cột Name thì dùng ""
        rows.append(row)
    if not rows:
        raise ValueError("bang plan khong co dong du lieu")
    return rows


def compute_features(est_text, bind_range_days=0):
    """Tính 30 feature từ nội dung plan_estimated.txt. Trả về dict {tên feature: giá trị}.

    bind_range_days là số ngày giữa hai bind ngày; giá trị bind đã biết trước khi chạy query.
    """
    rows = parse_plan_table(est_text)
    ops = [r["op"] for r in rows]
    root = rows[0]                                    # Id 0 là SELECT STATEMENT: số liệu của cả câu lệnh

    def count(prefix=None, contains=None):
        """Đếm số bước có operation bắt đầu bằng prefix hoặc chứa chuỗi contains."""
        n = 0
        for o in ops:
            if (prefix is not None and o.startswith(prefix)) or (contains is not None and contains in o):
                n += 1
        return n

    # Bước như FILTER không có Rows/Bytes: parse_number trả None, lọc bỏ trước khi lấy max.
    rows_all = [v for v in (parse_number(r["Rows"]) for r in rows) if v is not None]
    bytes_all = [v for v in (parse_number(r["Bytes"]) for r in rows) if v is not None]
    # startswith nhận một tuple: đúng nếu chuỗi bắt đầu bằng một trong các giá trị.
    # "x or 0.0": parse_number trả None (không phải số) thì thay bằng 0.0 để cộng được.
    table_bytes = [parse_number(r["Bytes"]) or 0.0 for r in rows
                   if r["op"].startswith(("TABLE ACCESS", "MAT_VIEW ACCESS"))]
    # Cột TempSpc chỉ có khi plan dự kiến dùng vùng tạm; r.get(..., "") trả "" khi cột vắng mặt.
    temp = sum(parse_number(r.get("TempSpc", "")) or 0.0 for r in rows)

    # Ô Cost có dạng "11M  (1)": số cost rồi %CPU trong ngoặc.
    cost_m = re.match(r"\s*([\d.]+[KMGT]?)\s*\((\d+)\)", root["Cost (%CPU)"])
    hints = re.search(r"Total hints for statement:\s*(\d+)", est_text)

    f = {
        "est_cost": parse_number(cost_m.group(1)) if cost_m else None,
        "est_cpu_pct": float(cost_m.group(2)) if cost_m else None,
        "est_time_s": parse_time(root["Time"]),
        "est_rows_root": parse_number(root["Rows"]),
        "est_bytes_root": parse_number(root["Bytes"]),
        "est_rows_max": max(rows_all) if rows_all else None,
        "est_bytes_max": max(bytes_all) if bytes_all else None,
        "est_bytes_table_access": sum(table_bytes),
        "est_temp_total": temp,
        "has_temp": int(temp > 0),                    # int(True/False) cho 1/0
        "n_steps": len(rows),
        "max_depth": max(r["depth"] for r in rows),
        # {...} với điều kiện là tập hợp: mỗi tên đối tượng chỉ được tính một lần dù xuất hiện nhiều bước.
        "n_objects": len({r["name"] for r in rows if r["name"]}),
        # "FULL" có trong cả "TABLE ACCESS STORAGE FULL" và "MAT_VIEW ACCESS STORAGE FULL".
        "n_full_scan": count(contains="FULL"),
        # sum(1 for ... if ...) là cách đếm số phần tử thỏa điều kiện.
        "n_table_full_scan": sum(1 for o in ops if o.startswith("TABLE ACCESS") and "FULL" in o),
        "n_index_access": count(prefix="INDEX"),
        "n_hash_join": count(prefix="HASH JOIN"),
        "n_nested_loops": count(prefix="NESTED LOOPS"),
        "n_merge_join": count(prefix="MERGE JOIN"),
        "n_sort": count(prefix="SORT"),                # gồm SORT ORDER BY và SORT GROUP BY
        "n_group_by": count(contains="GROUP BY"),      # gồm HASH GROUP BY và SORT GROUP BY
        "n_view": count(prefix="VIEW"),
        "n_filter": count(prefix="FILTER"),
        "n_partition_ops": count(prefix="PARTITION"),
        # PARTITION RANGE ALL: quét mọi partition (không cắt tỉa được partition)
        "n_partition_all": sum(1 for o in ops if o.startswith("PARTITION") and o.endswith("ALL")),
        "n_px": count(prefix="PX"),
        "n_hints": int(hints.group(1)) if hints else 0,
        "note_dynamic_sampling": int("dynamic statistics used" in est_text),
        "note_adaptive": int("this is an adaptive plan" in est_text),
        "bind_range_days": bind_range_days,
    }
    # dict giữ thứ tự thêm vào; assert dừng chương trình nếu tên hoặc thứ tự feature lệch FEATURE_COLS.
    assert list(f) == FEATURE_COLS
    return f


def bind_days(binds):
    """Số ngày giữa hai bind ngày đầu tiên trong summary["binds"]."""
    dates = []
    for v in binds.values():
        try:
            dates.append(datetime.strptime(v, "%Y-%m-%d %H:%M:%S"))
        except (TypeError, ValueError):
            continue                                  # bind không phải ngày (ví dụ "None") thì bỏ qua
    if len(dates) < 2:
        return 0
    return abs((dates[1] - dates[0]).days)            # hiệu hai datetime là timedelta, .days là số ngày


def read_summary(path):
    """Đọc summary.json, trả về dict phẳng: định danh và số liệu sau khi chạy (tên cột có tiền tố post_)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    cur, sess = d["cursor"], d["session_stat_delta"]
    # sql_file có dạng "sql\ten_query_v3.sql" (dấu \ của Windows).
    #   [^\\/]+   chuỗi ký tự không chứa \ hoặc / (tên query gốc)
    #   _v(\d+)   số phiên bản; \.sql$ kết thúc bằng .sql
    m = re.search(r"([^\\/]+)_v(\d+)\.sql$", d["sql_file"])
    return {
        "run_id": d["run_id"],
        "query": m.group(1) if m else d["sql_file"],
        "version": int(m.group(2)) if m else int(d["label"].lstrip("V")),
        "run_date": datetime.strptime(d["run_id"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d"),
        "bind_range_days": bind_days(d["binds"]),
        "estimated_plan_hash": d["estimated_plan_hash"],
        "actual_plan_hash": cur["plan_hash_value"],
        "post_elapsed_s": cur["elapsed_us"] / 1e6,    # elapsed_us tính bằng micro giây
        "post_cpu_s": cur["cpu_us"] / 1e6,
        "post_wall_s": d["wall_total_sec"],
        "post_buffer_gets": cur["buffer_gets"],
        "post_disk_reads": cur["disk_reads"],
        "post_read_bytes": sess["physical read total bytes"],
        "post_sorts_disk": sess["sorts (disk)"],
        "post_rows_returned": d["rows_returned"],
    }


def compare_structure(est_rows, act_rows):
    """So sánh cấu trúc estimated plan với actual plan, bỏ qua các operation PX của parallel execution.

    "giong": cùng dãy (operation, tên đối tượng) theo cùng thứ tự.
    "doi_thu_tu": cùng các bước nhưng khác thứ tự (ví dụ đổi bên dựng của HASH JOIN).
    "khac": khác các bước.
    """
    def keys(rows):
        return [(r["op"], r["name"]) for r in rows if not r["op"].startswith("PX")]

    a, b = keys(est_rows), keys(act_rows)
    if a == b:
        return "giong"
    # sorted() xếp hai danh sách theo cùng thứ tự rồi so sánh: bằng nhau nghĩa là cùng tập bước.
    if sorted(a) == sorted(b):
        return "doi_thu_tu"
    return "khac"
