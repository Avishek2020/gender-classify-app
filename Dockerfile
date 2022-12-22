# FROM python:3.8.0-slim
FROM python:3.8.6

EXPOSE 8501
# set work directory
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD ["gender_app.py"]