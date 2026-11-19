# Thin aliases. The official single command is: python3 -m auditgym check (fast, about 100 s);
# `make check-full` runs everything.
PY ?= python3
SEEDS ?= 1000
IMAGE ?= auditgym

.PHONY: check check-full test score fuzz episodes trajectories report demo docker-check

check:
	$(PY) -m auditgym check

check-full:
	$(PY) -m auditgym check --full

test:
	$(PY) -m unittest discover -s tests -q

# Strict: exit 1 on any TODO, pending: check or missing check module. `make score ARGS=--allow-todo`
# is the development mode; `AUDITGYM_STRICT=1 make check` is the strict submission gate.
score:
	$(PY) -m tools.score $(ARGS)

# Targets below get their tools as the plan is implemented (see docs/TASKS.md).
fuzz:
	$(PY) -m tools.fuzz --seeds $(SEEDS)

episodes:
	$(PY) -m tools.episodes --out episodes

trajectories:
	$(PY) -m tools.trajectories --out trajectories.jsonl --oracle-out runs/trajectories.oracle.jsonl

report:
	$(PY) -m tools.report --out docs/RESULTS.md

demo:
	$(PY) -m tools.demo

docker-check:
	docker build -t $(IMAGE) .
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
