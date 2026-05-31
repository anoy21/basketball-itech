# basketballll

# SmartCoach v3: Nhận diện & Phân tích Chiến thuật Bóng rổ

Đây là mã nguồn thử nghiệm cho bài toán áp dụng Machine Learning vào việc tự động đọc vị chiến thuật bóng rổ dựa trên dữ liệu quỹ đạo (trajectory tracking) của cầu thủ.

Hệ thống được thiết kế theo luồng xử lý 3 bước:
1. **Feature Extraction (LSTM):** Đọc chuỗi tọa độ không gian - thời gian của 10 cầu thủ trong 14 giây và nén thành một vector đặc trưng 64 chiều. 
2. **Classification & Regression (Gradient Boosting):** Kết hợp vector trên với bối cảnh trận đấu (tỉ số, thời gian ném) để phân loại tên chiến thuật và tính toán chỉ số đánh giá EPV (Expected Possession Value).
3. **Automated Reporting (Anthropic API / LLM):** Dịch các thông số đầu ra thành một đoạn phân tích bằng ngôn ngữ tự nhiên để hỗ trợ người dùng cuối (HLV).

*(Lưu ý: File có tích hợp cơ chế fallback. Nếu môi trường thiếu thư viện `torch` hoặc chưa cấu hình API key của Claude, code vẫn chạy bằng trọng số mô phỏng và template báo cáo tĩnh để đảm bảo không bị crash).*

## Cài đặt

```bash
pip install numpy matplotlib scikit-learn torch anthropic