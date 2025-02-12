# NBA Player Stats Predictor

A Python tool for predicting NBA player statistics and tracking prediction accuracy.

## Overview

This tool uses machine learning to predict NBA player statistics (Points, Rebounds, Assists) for upcoming games. It maintains a history of predictions and actual results, allowing for analysis of prediction accuracy over time.

## Features

- Predict PTS/REB/AST for NBA players
- Automatically detect upcoming games and matchups
- Track predictions in CSV format
- Update predictions with actual game results
- Analyze prediction accuracy
- Caching system for improved performance

## Requirements

- Python 3.8+
- pandas
- numpy
- xgboost
- nba_api

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd nba-stats-predictor
```

2. Install required packages:
```bash
pip install pandas numpy xgboost nba_api
```

## Usage

### Command Line Interface

The program provides three main commands:

1. Make predictions:
```bash
# Using default players (Josh Hart, Jalen Brunson, Pascal Siakam)
python main.py predict

# Specify custom players
python main.py predict --players "Josh Hart,Jalen Brunson,OG Anunoby"
```

2. Update previous predictions with actual results:
```bash
python main.py update
```

3. Analyze prediction accuracy:
```bash
python main.py analyze
```

### Getting Help

To see all available commands and options:
```bash
python main.py --help
```

To see help for a specific command:
```bash
python main.py predict --help
```

## File Structure

- `predictions-tracking.csv`: Stores predictions and actual results
- `prediction-analysis.csv`: Contains analysis of prediction accuracy
- `cache/`: Directory containing cached API responses and models

## Output Files

### predictions-tracking.csv
Contains:
- Date of prediction
- Player name
- Opponent
- Home/Away status
- Predicted and actual values for PTS/REB/AST
- Season and recent averages

### prediction-analysis.csv
Contains:
- Overall accuracy metrics (MAE, RMSE)
- Percentage of predictions within different ranges
- Biggest prediction misses
- Player-specific accuracy metrics

## Caching

The tool implements caching for:
- Player game data
- Team statistics
- Box scores
- Trained models

This significantly improves performance after initial data collection.
