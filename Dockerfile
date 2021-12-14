FROM python:3.7

COPY . ./

RUN pip install -r ./requirements.txt

RUN python ./downloads_dependencies.py

EXPOSE 8080

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]