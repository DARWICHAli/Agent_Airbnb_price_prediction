.PHONY: all fetch preprocess eda train eval report clean

all: fetch preprocess eda train eval report

fetch:
	python -u src/fetcher.py --city $(city)

preprocess:
	python -u src/preprocess.py --city $(city)

eda:
	python -u src/eda.py --city $(city)

train:
	python -u src/train.py --city $(city) --config configs/base.yaml

eval:
	python -u src/evaluate.py --city $(city)

report:
	python -u src/agent.py --city $(city) --stage report

clean:
	rm -rf data/raw/* data/processed/* artifacts/* reports/* mlruns/* mlflow.db
