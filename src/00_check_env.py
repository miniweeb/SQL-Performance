# -*- coding: utf-8 -*-
"""
Bước 0: Kiểm tra phiên bản Python và các thư viện cần dùng.

Logic: nạp từng thư viện theo tên dạng chuỗi bằng importlib.import_module.
Thư viện nào chưa cài sẽ phát sinh lỗi; lỗi được bắt bằng try/except và thư viện
được ghi vào danh sách còn thiếu, nên chương trình không dừng giữa chừng.
"""
import importlib
import sys

# Tên thư viện dùng khi cài bằng pip (khớp với environment/requirements.txt).
NEED = ["pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "streamlit", "ipykernel"]
# Thư viện có tên import khác tên cài đặt; thư viện không có trong bảng này thì hai tên giống nhau.
IMPORT_NAME = {"scikit-learn": "sklearn"}

print("1. Phien Ban Python")
# sys.version có dạng "3.13.0 (main, ...)"; split()[0] chỉ lấy số phiên bản.
# sys.executable là đường dẫn trình thông dịch đang chạy.
print(f"   {sys.version.split()[0]}  ({sys.executable})")

print("\n2. Thu Vien")
missing = []
for pip_name in NEED:
    # dict.get(khóa, mặc định): có trong bảng thì lấy tên import, không có thì dùng luôn tên pip.
    mod = IMPORT_NAME.get(pip_name, pip_name)
    try:
        m = importlib.import_module(mod)
        # :<13 căn trái trong 13 ký tự cho thẳng cột; getattr trả "?" nếu thư viện không có __version__.
        print(f"   [OK]    {pip_name:<13} {getattr(m, '__version__', '?')}")
    except Exception:
        print(f"   [THIEU] {pip_name}")
        missing.append(pip_name)

print("\n3. Ket Luan")
if not missing:
    print("   Du thu vien.")
else:
    print("   Cai bang lenh: python -m pip install -r environment/requirements.txt")
    # set(missing) <= {...}: tập thư viện còn thiếu nằm trọn trong 2 thư viện tùy chọn.
    if set(missing) <= {"streamlit", "ipykernel"}:
        print("   Cac script trong src/ van chay duoc; streamlit can cho app, ipykernel can cho main.ipynb.")