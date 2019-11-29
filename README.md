# Vivid

Support Tools for Machine Learning Vividly 🚀

## Install

```bash
pip install git+https://gitlab.com/nyker510/vivid
```

## Developer

### Recomended

create same docker-image as test enviroment.

```bash
docker-compose build
docker-compose up -d
docker exec -it vivid-test bash
```

### Test

use `pytest` for test tool (see [gitlab-ci.yml](./gitlab-ci.yml)).

```bash
pytest tests
```
