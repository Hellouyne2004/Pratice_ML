# Sử dụng base image là python gọn nhẹ
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements và cài đặt các thư viện (Track environment)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Expose port 5000 cho ứng dụng Flask
EXPOSE 5000

# Mặc định khi chạy container sẽ khởi chạy ứng dụng web API
CMD ["python", "app.py"]
