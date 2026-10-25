# -*- coding: utf-8 -*-
"""
So sánh các phiên bản của một query từ estimated execution plan.

Chạy: streamlit run app/streamlit_app.py
Đầu vào: models/model.pkl (src/04_train_evaluate.py) và file plan_estimated.txt của từng phiên bản.
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
import config

LEVEL_COLORS = {"Normal": "#4C9F70", "Warning": "#E0A030", "Critical": "#C0504D"}
PLAN_METRICS = [
    ("est_time_s", "Estimated time (s)"), ("est_cost", "Estimated cost"), ("n_steps", "Số bước plan"),
    ("n_full_scan", "Full scan"), ("n_hash_join", "Hash join"), ("n_nested_loops", "Nested loops"),
    ("n_sort", "Sort"), ("n_partition_all", "Quét mọi partition"), ("est_temp_total", "Temp space dự kiến"),
]


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PKL)


def risk_index(proba):
    """Kỳ vọng mức (Normal=0, Warning=1, Critical=2) đổi về thang 0-100."""
    return float(proba @ np.array([0, 1, 2]) / 2 * 100)


def logit_contributions(model, x, feature_names):
    """Đóng góp của từng feature vào chênh lệch logit Critical - Normal (chỉ với pipeline hồi quy logistic)."""
    try:
        scaler, clf = model.steps[1][1], model.steps[2][1]
        z = scaler.transform(np.log1p(x.to_numpy(dtype=float)))[0]
        return pd.Series(z * (clf.coef_[2] - clf.coef_[0]), index=feature_names)
    except (AttributeError, IndexError):
        return None


def parse_version(bundle, name, text, bind_days):
    feats = common.compute_features(text, bind_days)
    x = pd.DataFrame([feats])[bundle["features"]]
    proba = bundle["model"].predict_proba(x)[0]
    return {"name": name, "feats": feats, "proba": proba, "risk": risk_index(proba),
            "level": bundle["levels"][int(proba.argmax())], "text": text,
            "contrib": logit_contributions(bundle["model"], x, bundle["features"])}


st.set_page_config(page_title="So sánh phiên bản SQL query", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;max-width:1200px}</style>", unsafe_allow_html=True)
st.title("So sánh phiên bản SQL query")
st.caption("Dự báo mức tiêu thụ tài nguyên (Normal / Warning / Critical) từ estimated plan, trước khi chạy query.")

if not Path(config.MODEL_PKL).exists():
    st.error("Chưa có models/model.pkl. Chạy src/04_train_evaluate.py trước.")
    st.stop()
bundle = load_model()
levels = bundle["levels"]

with st.sidebar:
    st.header("Đầu vào")
    files = st.file_uploader("plan_estimated.txt (mỗi phiên bản một file)", type=["txt"], accept_multiple_files=True)
    bind_days = st.number_input("Số ngày giữa hai bind ngày", min_value=0, value=0, step=1,
                                help="Dùng chung cho mọi phiên bản; 0 nếu query không có bind ngày.")
    names = {}
    if files:
        st.subheader("Tên phiên bản")
        for i, f in enumerate(files):
            names[i] = st.text_input(f"File {i + 1}", value=Path(f.name).stem, key=f"name_{i}")
    st.divider()
    st.caption(f"Mô hình: {bundle['name']}. Điểm rủi ro = kỳ vọng mức trên thang 0-100.")

if not files:
    st.info("Tải lên từ hai file plan_estimated.txt trở lên ở thanh bên để so sánh.")
    st.stop()

versions = []
for i, f in enumerate(files):
    try:
        versions.append(parse_version(bundle, names[i], f.getvalue().decode("utf-8", errors="replace"), bind_days))
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        st.warning(f"{f.name}: không đọc được plan ({exc})")
if not versions:
    st.stop()
if len({v["name"] for v in versions}) < len(versions):
    st.warning("Tên phiên bản bị trùng; đổi tên ở thanh bên.")

versions.sort(key=lambda v: v["risk"])
by_name = {v["name"]: v for v in versions}
best = versions[0]
base_name = st.selectbox("Phiên bản gốc để so sánh", [v["name"] for v in versions],
                         index=len(versions) - 1, help="Mặc định là phiên bản có điểm rủi ro cao nhất.")
base = by_name[base_name]

cols = st.columns(min(len(versions), 4))
for col, v in zip(cols, versions):
    delta = v["risk"] - base["risk"]
    col.metric(v["name"], f"{v['risk']:.0f} / 100", None if v is base else f"{delta:+.0f} so với {base_name}",
               delta_color="inverse")
    col.markdown(f"<span style='color:{LEVEL_COLORS[v['level']]};font-weight:600'>{v['level']}</span>",
                 unsafe_allow_html=True)
if len(versions) > 1:
    st.success(f"Rủi ro thấp nhất: **{best['name']}** ({best['risk']:.0f}/100, {best['level']}).")

tab_rank, tab_diff, tab_why, tab_plan = st.tabs(["Xếp hạng", "Khác biệt plan", "Vì sao", "Xem plan"])

with tab_rank:
    long = pd.DataFrame([{"Phiên bản": v["name"], "Mức": lv, "Xác suất": float(p)}
                         for v in versions for lv, p in zip(levels, v["proba"])])
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y("Phiên bản:N", sort=[v["name"] for v in versions], title=None),
        x=alt.X("Xác suất:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Mức:N", scale=alt.Scale(domain=levels, range=[LEVEL_COLORS[x] for x in levels]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Phiên bản", "Mức", alt.Tooltip("Xác suất:Q", format=".0%")]).properties(
        height=60 + 40 * len(versions)), use_container_width=True)
    table = pd.DataFrame([{"Phiên bản": v["name"], "Dự báo": v["level"], "Điểm rủi ro": round(v["risk"]),
                           **{f"P({lv})": f"{p:.0%}" for lv, p in zip(levels, v["proba"])}} for v in versions])
    st.dataframe(table, hide_index=True, use_container_width=True)

with tab_diff:
    rows = []
    for key, label in PLAN_METRICS:
        row = {"Chỉ số": label, base_name: f"{base['feats'][key]:,.0f}"}
        for v in versions:
            if v is not base:
                d = v["feats"][key] - base["feats"][key]
                row[v["name"]] = f"{v['feats'][key]:,.0f} ({d:+,.0f})"
        rows.append(row)
    st.caption(f"Giá trị của từng phiên bản; trong ngoặc là chênh lệch so với {base_name}.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_why:
    others = [v for v in versions if v is not base]
    if base["contrib"] is None or not others:
        st.write("Cần ít nhất hai phiên bản và mô hình hồi quy logistic để giải thích.")
    else:
        pick = st.selectbox("Phiên bản cần giải thích", [v["name"] for v in others])
        diff = (by_name[pick]["contrib"] - base["contrib"]).sort_values(key=abs, ascending=False).head(8)
        chart_df = pd.DataFrame({"Feature": diff.index, "Tác động": diff.values})
        st.caption(f"Feature làm điểm rủi ro của {pick} cao hơn (đỏ) hoặc thấp hơn (xanh) {base_name}. "
                   "Đơn vị: đóng góp vào logit Critical so với Normal.")
        st.altair_chart(alt.Chart(chart_df).mark_bar().encode(
            y=alt.Y("Feature:N", sort=None, title=None), x=alt.X("Tác động:Q", title=None),
            color=alt.condition(alt.datum["Tác động"] > 0, alt.value(LEVEL_COLORS["Critical"]),
                                alt.value(LEVEL_COLORS["Normal"]))).properties(height=300),
            use_container_width=True)

with tab_plan:
    pick = st.selectbox("Phiên bản", [v["name"] for v in versions], key="plan_pick")
    st.code(by_name[pick]["text"], language="text")

st.caption("Kết quả là dự báo từ mô hình proof of concept (64 lần chạy, 32 query); dùng để xếp thứ tự ưu tiên "
           "kiểm tra, không thay thế việc chạy thử.")
