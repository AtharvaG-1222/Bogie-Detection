# 1. Base OS image with Python 3.10
FROM python:3.10-slim

# 2. OpenCV aur basic system tools ke liye required C-libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Container ke andar working directory set karo
WORKDIR /app

# 4. Layer caching: pehle requirements copy karke install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Apna code aur folders container ke andar copy karo
COPY . .

# 6. Default command jab container start ho
CMD ["python", "test_anomaly.py"]