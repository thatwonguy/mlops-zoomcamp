FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install pandas numpy pyarrow scikit-learn

CMD ["python", "starter.py", "--year", "2023", "--month", "5"]
