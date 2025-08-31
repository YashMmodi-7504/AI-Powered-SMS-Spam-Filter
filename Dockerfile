# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
