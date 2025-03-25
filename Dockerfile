FROM python:3.12.4

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "/home/python/app:/home/python/app"

# ENV GOOGLE_CLOUD_PROJECT "ufg-prd-energygpt"
# ENV GOOGLE_APPLICATION_CREDENTIALS "ufg-prd-energygpt"
# ENV POSTGRES_DB "postgres"
# ENV POSTGRES_USER "postgres"
# ENV POSTGRES_PASSWORD "legacy"
# ENV POSTGRES_HOST "172.28.0.5"
# ENV POSTGRES_PORT 5432



WORKDIR /home/python/app

COPY . .

EXPOSE 8000

# ENTRYPOINT ["/home/python/app/entrypoint.sh"]