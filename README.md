# Premier League Match Result Prediction Using Logistic Regression and Gradient Boosting

Predicts the result of a Premier League match (Home win / Draw / Away win)
using Elo ratings and recent-form features.

## Project structure
- `data/raw/` - downloaded season CSVs (football-data.co.uk)
- `data/processed/` - engineered feature dataset
- `src/data_loader.py` - downloads and combines seasons
- `src/features.py` - Elo and recent-form features
- `src/train.py` - trains the models and saves the best one
- `src/evaluate.py` - accuracy, log loss, confusion matrix
- `models/` - saved model
- `results/` - confusion matrix images
- `notebooks/` - exploration

## How to run
```
pip install -r requirements.txt
python -m src.train
```
Run from the project root folder. An internet connection is needed the first time to download the data.

## Method
1. Load 6 seasons of match data
2. Build features: Elo difference, last-5 points, goals for/against
3. Time-based split: train on older seasons, test on the latest season
4. Compare against a baseline, Logistic Regression, and Gradient Boosting
5. Evaluate with accuracy, log loss, and a confusion matrix

## Results
Fill in after running (expect roughly 50-55% accuracy).

## Future work
Rest days, head-to-head record, bookmaker odds, SHAP, Streamlit app.
