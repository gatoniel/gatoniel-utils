SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	poetry run mypy gatoniel_utils tests/**/*.py
	poetry run flake8 .
	poetry run doc8 -q docs

.PHONY: unit
unit:
	poetry run pytest

.PHONY: package
package:
	poetry check
	poetry run pip check
	poetry run safety check --full-report

.PHONY: test
test: lint package unit

