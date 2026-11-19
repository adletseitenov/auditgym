# Thin aliases. The official single command is: python -m auditgym check
PY ?= python3
SEEDS ?= 1000
IMAGE ?= auditgym

.PHONY: check test score fuzz episodes trajectories report demo docker-check

check:
	$(PY) -m auditgym check

test:
	$(PY) -m unittest discover -s tests -q

score:
	$(PY) -m tools.score

# Targets below get their tools as the plan is implemented (see docs/TASKS.md).
fuzz:
	$(PY) -m tools.fuzz --seeds $(SEEDS)

episodes:
	$(PY) -m tools.episodes --out episodes

trajectories:
	$(PY) -m tools.trajectories --out trajectories.jsonl

report:
	$(PY) -m tools.report --out docs/RESULTS.md

demo:
	$(PY) -m tools.demo

docker-check:
	docker build -t $(IMAGE) .
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
