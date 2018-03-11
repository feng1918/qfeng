
DEFAULT_PYTHON := /usr/local/bin/python3
VIRTUALENV := /usr/local/bin/virtualenv

VENV := .venv
PYTHON := env PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python

venv:
	test -d $(VENV) || $(VIRTUALENV) -p $(DEFAULT_PYTHON) -q $(VENV)

docker: 
	docker build -t qfeng/zipline .

.PHONY: venv docker
