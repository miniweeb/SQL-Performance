"""
Khai báo tập trung mọi đường dẫn của dự án.

ROOT được tính từ vị trí của chính file này (src/config.py -> thư mục cha là gốc repo),
nên các script chạy đúng dù được gọi từ thư mục nào.
"""
from pathlib import Path

# __file__ là đường dẫn của file này; resolve() đổi thành đường dẫn tuyệt đối;
# parents[0] là src/, parents[1] là thư mục gốc repo.
ROOT = Path(__file__).resolve().parents[1]

# 1. Dữ Liệu
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            
INTERIM_DIR = DATA_DIR / "interim"    # kết quả trích xuất: features, features_clean, check_report
LABELED_DIR = DATA_DIR / "labeled"    # label gán tự động và kiểm tra độ nhạy ngưỡng

FEATURES_CSV = INTERIM_DIR / "features.csv"
FEATURES_CLEAN_CSV = INTERIM_DIR / "features_clean.csv"
CHECK_REPORT_CSV = INTERIM_DIR / "check_report.csv"
ERRORS_CSV = INTERIM_DIR / "errors.csv"
LABELS_CSV = LABELED_DIR / "labels.csv"
SENSITIVITY_CSV = LABELED_DIR / "label_sensitivity.csv"

# Ngày hệ thống tải nặng: elapsed của các lần chạy trong ngày này cao hơn ngày thường.
PEAK_DAY = "2026-09-03"

# 2. Mô Hình
MODELS_DIR = ROOT / "models"
MODEL_PKL = MODELS_DIR / "model.pkl"
RUN_INFO_JSON = MODELS_DIR / "run_info.json"

# 3. Kết Quả Dùng Cho Báo Cáo
REPORTS_DIR = ROOT / "reports"
FIG_EDA = REPORTS_DIR / "figures" / "eda"
FIG_RESULTS = REPORTS_DIR / "figures" / "results"
TAB_EDA = REPORTS_DIR / "tables" / "eda"
TAB_RESULTS = REPORTS_DIR / "tables" / "results"


def ensure_dirs():
    """Tạo các thư mục đầu ra nếu chưa có.
    parents=True: tạo luôn thư mục cha còn thiếu; exist_ok=True: không báo lỗi nếu đã tồn tại."""
    for d in (RAW_DIR, INTERIM_DIR, LABELED_DIR, MODELS_DIR, FIG_EDA, FIG_RESULTS, TAB_EDA, TAB_RESULTS):
        d.mkdir(parents=True, exist_ok=True)