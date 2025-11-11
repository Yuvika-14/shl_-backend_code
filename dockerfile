
FROM python
WORKDIR /app
COPY . /app
CMD ["uvicorn", "shl_ai.app:app", "python3","app.py"]