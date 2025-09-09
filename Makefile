
.PHONY: setup data features train app test
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
data:
	python scripts/pull_statcast.py --config config.yaml
features:
	python scripts/build_features.py --config config.yaml
train:
	python scripts/train_model.py --config config.yaml
app:
	streamlit run app.py
test:
	pytest -q
