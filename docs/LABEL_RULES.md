# Quy Tắc Gán Label

Label có ba mức có thứ bậc: **Normal < Warning < Critical**. Label được tính từ số liệu sau khi chạy (`summary.json`); feature của model chỉ lấy từ estimated plan, nên label không rò rỉ vào feature. Ngưỡng trong `03_label_data.py` khớp với bảng ở mục 3.

## 1. Dữ liệu tham chiếu (64 lần chạy, 32 query gốc)

| Chỉ số | P10 | P25 | Trung vị | P75 | P90 | Lớn nhất |
|---|---|---|---|---|---|---|
| Elapsed (phút) | 2,4 | 7,2 | 10,8 | 15,1 | 25,2 | 30,1 |
| CPU (phút) | 2,2 | 6,7 | 9,2 | 13,5 | 18,7 | 23,9 |
| Đọc từ đĩa (GB) | 13,5 | 46,4 | 78,6 | 162,0 | 604,7 | 1081,2 |

Có 18 lần chạy tràn ra đĩa khi sắp xếp (`sorts (disk) > 0`) và 7 lần chạy vào ngày cao điểm. Các phiên bản của cùng một query chung một nhóm; có 12 query chỉ có một lần chạy.

Phân phối elapsed có hai cụm đông: 8,5 đến 10,2 phút (15 bản ghi) và 12,2 đến 13,8 phút (11 bản ghi). Giữa các bản ghi có hai khoảng trống: 10,2 đến 11,3 phút và 19,7 đến 22,0 phút.

## 2. Chỉ số dùng để gán label

| Vai trò | Chỉ số | Nguồn | Lý do |
|---|---|---|---|
| Chính | `elapsed_us` | `summary.json` → `cursor` | Thời gian người tuning chờ và thời gian chiếm database dùng chung; có ở mọi lần chạy |
| Nâng bậc | `sorts (disk)` | `summary.json` → `session_stat_delta` | Tràn ra vùng tạm là dấu hiệu thiếu bộ nhớ và tốn tài nguyên ngoài thời gian chạy |
| Không dùng | `physical read total bytes` | `summary.json` | Spearman 0,69 với elapsed, phần lớn trùng thông tin |
| Không dùng | `cpu_us`, `buffer_gets` | `summary.json` | Trùng thông tin với elapsed (Spearman giữa CPU và elapsed là 0,98) |
| Không dùng | `rows_returned` | `summary.json` | Là kích thước kết quả, không phải mức tiêu thụ tài nguyên |

## 3. Ngưỡng

| Ngưỡng | Giá trị | Căn cứ |
|---|---|---|
| Warning từ (elapsed) | 11 phút | Nằm trong khoảng trống 10,2 đến 11,3 phút, giữa hai cụm bản ghi |
| Critical từ (elapsed) | 21 phút | Nằm trong khoảng trống 19,7 đến 22,0 phút, thuộc nửa trên của khoảng 5 đến 30 phút mà query báo cáo thường chạy |
| Nâng một bậc | `sorts (disk) > 0` | Mục 4 |

Tên hai mức Warning và Critical đặt theo hai mức cảnh báo của OEM. Các giá trị ngưỡng là tham số của dự án. Hai ngưỡng đặt trong các khoảng trống của phân phối để không cắt đôi một cụm bản ghi: chỉ 4 bản ghi nằm trong ±10% quanh ngưỡng Warning và 6 bản ghi quanh ngưỡng Critical. Kết quả là ba lớp có 26 / 21 / 17 bản ghi (41%, 33%, 27%).

## 4. Quy tắc nâng bậc và thứ tự ưu tiên

1. Tính mức theo `elapsed` (Normal / Warning / Critical).
2. Nếu `sorts (disk) > 0` và mức hiện tại chưa phải Critical thì nâng một mức (mã quy tắc `elapsed+spill`); các trường hợp còn lại có mã `elapsed`.
3. Cột `rule` trong `labels.csv` ghi mã quy tắc quyết định; cột `peak_day` (True/False) ghi bản ghi chạy vào ngày cao điểm.

Quy tắc kích hoạt cho 18 bản ghi và làm đổi label của 14 bản ghi (4 bản ghi còn lại đã ở Critical). Critical nghĩa là chạy từ 21 phút trở lên, hoặc chạy từ 11 phút trở lên và có tràn đĩa.

## 5. Kết quả gán label

| Cấu hình | Normal | Warning | Critical |
|---|---|---|---|
| Chỉ theo elapsed (11 / 21 phút) | 32 | 23 | 9 |
| Theo elapsed và nâng bậc khi tràn đĩa | 26 | 21 | 17 |

8 trong 17 bản ghi Critical là các query chạy từ 11 phút trở lên có tràn đĩa; 9 bản ghi còn lại là Critical thuần theo elapsed.

## 6. Độ nhạy và giới hạn

- **Độ nhạy ngưỡng:** dời ngưỡng Warning ±10% (9,9 và 12,1 phút) làm đổi label của 2 và 2 bản ghi (3,1%); dời ngưỡng Critical ±10% (18,9 và 23,1 phút) làm đổi 3 và 1 bản ghi (4,7% và 1,6%).
- **Ngày cao điểm:** elapsed trong ngày cao điểm cao hơn ngày thường khoảng 1,6 đến 2,6 lần. Bản ghi được gắn cờ `peak_day` và kết quả model được báo cáo cả khi có và khi bỏ 7 bản ghi này. Nếu giả định tải gấp đôi thì 6 trong 7 bản ghi đổi label, nên label của các bản ghi này phụ thuộc nhiều vào tải của ngày chạy.
- **Cùng một query:** 11 trong 20 query có nhiều phiên bản có label khác nhau giữa các phiên bản.
- **Quy mô dữ liệu:** 64 bản ghi từ 32 nhóm query. Cross-validation chia theo nhóm nên số đơn vị độc lập là 32.
- **Phạm vi của nhãn:** model dự báo kết quả của chính quy tắc này, nên mọi kết quả đi kèm ngưỡng ở mục 3.

## 7. Kết quả đánh giá

Stratified Group K-Fold 5 fold lặp 10 lần, chia theo query, 22 feature từ `plan_estimated.txt` (`04_train_evaluate.py`). Macro-F1 trung bình ± độ lệch chuẩn giữa các fold:

| Mô hình | F1 train | F1 test | Chênh lệch | QWK test |
|---|---|---|---|---|
| Luôn đoán lớp phổ biến | 0,19 | 0,19 ± 0,01 | 0,00 | 0,00 |
| Hai ngưỡng trên estimated time | 0,57 | 0,41 ± 0,16 | 0,16 | 0,30 |
| Cây quyết định | 0,71 | 0,33 ± 0,12 | 0,39 | 0,16 |
| Hồi quy logistic | 0,85 | 0,59 ± 0,13 | 0,26 | 0,60 |
| Random Forest | 0,95 | 0,43 ± 0,12 | 0,52 | 0,30 |
| Gradient Boosting | 1,00 | 0,48 ± 0,12 | 0,52 | 0,35 |
| Mọi mô hình với nhãn bị xáo trộn | | 0,27 đến 0,30 | | gần 0 |

- Nhãn bị xáo trộn cho kết quả ngẫu nhiên, không có dấu hiệu rò rỉ.
- Hồi quy logistic cao hơn baseline hai ngưỡng trên estimated time 0,18 điểm F1 (khoảng 1,2 độ lệch chuẩn); Gradient Boosting cao hơn 0,07, Random Forest cao hơn 0,02, cây quyết định thấp hơn 0,08.
- F1 train 0,71 đến 1,00 so với F1 test 0,33 đến 0,59: có overfit. Learning curve (30% đến 100% số query huấn luyện): hồi quy logistic tăng từ 0,32 lên 0,59, Gradient Boosting từ 0,37 lên 0,48, Random Forest từ 0,29 lên 0,43, cây quyết định không tăng.
- Điểm phụ thuộc cách chia fold: hồi quy logistic 0,59 khi phân tầng và 0,48 khi không phân tầng. Bỏ 7 lần chạy ngày cao điểm thì hồi quy logistic đạt 0,49 (baseline 0,39).
- Phần lớn lỗi lệch một mức: hồi quy logistic đúng 62%, lệch một mức 33%, lệch hai mức (Normal với Critical) 5%.
