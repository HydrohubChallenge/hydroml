## Integration using docker to connect Hydroml to Surface

Create a new docker network

```bash
docker network create integration_net
```

Connect both api to the network

```bash
docker connect integration_net surface_api_1
docker connect integration_net hydroml_api_1
```

Inspect the network to check the IP of each api

```bash
docker network inspect integration_net
```

or inspect directly the api

```bash
docker inspect surface_api_1 | grep IPAddress
docker inspect hydroml_api_1 | grep IPAddress
```

To test the network you can access the container via browser using the IP followed by the port of the API, in this case, both api uses the port :8000

Or you can ping the container via docker

```bash
docker exec -ti surface_api_1 ping -c 5 hydroml_api_1
docker exec -ti hydroml_api_1 ping -c 5 surface_api_1
```
