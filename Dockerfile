FROM python

WORKDIR /app

RUN pip3 install -U numpy pandas matplotlib scikit-learn seaborn django graphviz

COPY ./ ./

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD python3 manage.py runserver 0.0.0.0:8080