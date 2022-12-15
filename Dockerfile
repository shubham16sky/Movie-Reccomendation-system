FROM python:3.9-slim
LABEL maintainer="shubhampc16@gmail.com"

EXPOSE 8501

COPY . /app

WORKDIR /app


RUN pip3 install -r requirements.txt
RUN python3 data.py

ENTRYPOINT ["streamlit","run","app.py","--server.port=8501", "--server.address=0.0.0.0"]