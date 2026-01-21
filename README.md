# RA2 Inverse Modeling: JR ERP -> Parameters

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run pipeline
python data/generate_dataset.py
python data/prepare_training_data.py
python models/train.py
python eval/evaluate.py
python eval/plot_results.py
python eval/sensitivity_validation.py
# Bayesian-Transformer-inverse-model
