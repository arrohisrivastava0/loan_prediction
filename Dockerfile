# Official Python image
FROM python:3.12-slim

# Working directory inside the container
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project into the container
COPY . .

# app runs on port 8000
EXPOSE 8000

# start the API when container runs
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]