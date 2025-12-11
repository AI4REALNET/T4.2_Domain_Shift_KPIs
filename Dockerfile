FROM python:3.10-slim-bullseye

WORKDIR /usr/src

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -U ".[powergrid]"

CMD ["python", "main.py", "--config_path", "./config.ini"]
