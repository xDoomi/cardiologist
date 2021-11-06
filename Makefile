## Lint your code using pylint
.PHONY: lint
lint:
	python -m pylint --version
	python -m pylint users restapi board cardiologist
## Run tests using pytest
.PHONY: test
test:
	python -m pytest --version
	python -m pytest tests
## Format your code using black
.PHONY: black
black:
	python -m black --version
	python -m black .
## Format imports in your code using isort
.PHONY: isort
isort:
	python -m isort --version
	python -m isort .
## Run ci part
.PHONY: ci
ci:
	pre-commit run lint
	pre-commit run test
