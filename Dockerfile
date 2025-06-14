FROM cgr.dev/chainguard/python:latest-dev AS builder
  
WORKDIR /app
  
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
  
FROM cgr.dev/chainguard/python:latest
  
WORKDIR /app
  
COPY --chown=nonroot:nonroot --from=builder /home/nonroot/.local /home/nonroot/.local
  
COPY ./app ./app
COPY ./models ./models
  
ENV PATH=/home/nonroot/.local/bin:$PATH
  
EXPOSE 8000
  
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
