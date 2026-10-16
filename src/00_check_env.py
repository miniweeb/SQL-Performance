"""
Bước 0: kiểm tra phiên bản Python và các thư viện cần dùng.
Dùng importlib.import_module để nạp thư viện theo tên dạng chuỗi; thư viện nào chưa cài
sẽ phát sinh lỗi và được ghi vào danh sách còn thiếu, thay vì làm dừng cả chương trình.
"""
import importlib
import sys

# (tên dùng khi cài bằng pip, tên dùng khi import) - hai tên này có thể khác nhau, ví dụ scikit-learn / sklearn.
NEED = ["pandas", "numpy", "scikit-learn", "matplotlib", "joblib", "streamlit", "ipykernel"]
# (tên dùng khi cài bằng pip, tên dùng khi import) - hai tên này có thể khác nhau, ví dụ scikit-learn / sklearn.
IMPORT_NAME = {"scikit-learn": "sklearn"} 

for pip_name in NEED:
    mod = IMPORT_NAME.get(pip_name, pip_name)

print("1. Phien Ban Python")
print(f"   {sys.version.split()[0]}  ({sys.executable})")

print("\n2. Thu Vien")
missing = []
for pip_name, mod in NEED:
    try:
        m = importlib.import_module(mod)
        print(f"   [OK]    {pip_name:<13} {getattr(m, '__version__', '?')}")
    except Exception:
        print(f"   [THIEU] {pip_name}")
        missing.append(pip_name)

print("\n3. Ket Luan")
if not missing:
    print("   Du thu vien.")
else:
    print("   Cai bang lenh: python -m pip install -r environment/requirements.txt")
    if set(missing) <= {"streamlit", "ipykernel"}:
        print("   Cac script trong src/ van chay duoc; streamlit can cho app, ipykernel can cho main.ipynb.")