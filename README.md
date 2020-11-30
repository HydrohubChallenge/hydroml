# hydroml

## API 

Copy .env file from Google Drive hydroml_data -> env files -> .env to the root of project. Then run the commands below.

```bash
docker-compose up -d --build api db
docker-compose exec api python manage.py migrate
```


## API Jupyter

```bash
docker-compose exec api bash
python manage.py shell_plus --notebook
```
Open one of the URLs from the log on the browser.

## Jupyter

```bash
docker-compose up -d --build jupyter

docker-compose logs jupyter
```

Open one of the URLs from the log on the browser.

## Injects Anomalies: Water Levels

```bash
docker-compose up -d --build injects_anomalies_wl

docker-compose logs injects_anomalies_wl
```

Open https:hostip/8050 on the browser.