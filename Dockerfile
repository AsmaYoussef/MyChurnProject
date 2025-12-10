# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port 80
EXPOSE 80

# Run the FastAPI app
CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "80"]
