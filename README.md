# Vivid

Support Tools for Machine Learning Vividly 🚀

## Install

```bash
pip install git+https://gitlab.com/nyker510/vivid
```

## Sample Code

In `/vivid/smaples`, Some sample script codes exist.

## Developer

### Requiremtns

* docker
* docker-compose

create docker-image from docker-compose file

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
