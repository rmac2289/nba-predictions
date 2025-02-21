import os
import json
import pickle
import time
import logging
import argparse
from datetime import datetime
from requests.exceptions import ReadTimeout

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    teamgamelog,
    boxscoretraditionalv2,
    scoreboardv2
)
from nba_api.stats.static import teams, players

# Constants
SEASON = '2024-25'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility Functions
def api_call_with_retry(func, max_retries=3, delay=2):
    """Add exponential backoff for API calls"""
    for i in range(max_retries):
        try:
            return func()
        except ReadTimeout:
            wait_time = delay * (2 ** i)
            if i < max_retries - 1:
                logger.warning(f"API timeout, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            logger.error("Max retries reached for API call")
            raise
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            if i < max_retries - 1:
                time.sleep(delay)
                continue
            raise
    return None

def get_team_id_from_abbreviation(abbreviation):
    """Get team ID from team abbreviation"""
    try:
        nba_teams = teams.get_teams()
        team = [team for team in nba_teams if team['abbreviation'] == abbreviation][0]
        return team['id']
    except Exception as e:
        logger.error(f"Error getting team ID for {abbreviation}: {e}")
        return None

# Cache Helper Functions
def load_from_cache(cache_dir, filename, is_pickle=False):
    """Generic function to load data from cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, filename)
    today = datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists(cache_file):
        try:
            if is_pickle:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
            else:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
            if cached_data.get('last_updated', '') == today:
                return cached_data.get('data')
        except Exception as e:
            logger.error(f"Error loading cache {filename}: {e}")
    return None

def save_to_cache(cache_dir, filename, data, is_pickle=False):
    """Generic function to save data to cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, filename)
    
    try:
        cache_data = {
            'last_updated': datetime.now().strftime("%Y-%m-%d"),
            'data': data
        }
        
        if is_pickle:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        else:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving to cache {filename}: {e}")


# Cache Optimization Functions
def bulk_cache_team_stats(team_ids, season):
    """Pre-cache team stats for multiple teams"""
    logger.info(f"Bulk caching team stats for {len(team_ids)} teams...")
    cached_count = 0
    for team_id in team_ids:
        try:
            # Check if already cached first
            cache_dir = "cache/team_games"
            filename = f"team_{team_id}_{season}.json"
            if load_from_cache(cache_dir, filename) is None:
                get_enhanced_team_stats(team_id, season)
                time.sleep(1)  # Maintain rate limit but do it once
                cached_count += 1
            else:
                logger.debug(f"Team {team_id} stats already cached")
        except Exception as e:
            logger.error(f"Error caching team {team_id}: {e}")
    logger.info(f"Cached {cached_count} new team stats")

def bulk_cache_player_data(player_name, season):
    """Pre-cache all data needed for a player"""
    logger.info(f"\nPre-caching data for {player_name}")
    
    # Cache player games
    games = load_or_fetch_player_games(player_name, season)
    if games is None:
        logger.error(f"Could not fetch games for {player_name}")
        return
        
    # Get unique opponents
    opponents = games['MATCHUP'].apply(lambda x: x.split()[-1]).unique()
    logger.info(f"Found {len(opponents)} unique opponents")
    
    # Cache team stats for all opponents
    team_ids = []
    for opp in opponents:
        team_id = get_team_id_from_abbreviation(opp)
        if team_id:
            team_ids.append(team_id)
    
    bulk_cache_team_stats(team_ids, season)
    
    # Cache league stats
    logger.info("Caching league stats...")
    load_or_fetch_league_stats(season, 'Base')
    load_or_fetch_league_stats(season, 'Defense')
    
    # Pre-train models for common stats
    logger.info("Pre-training prediction models...")
    for stat in ['PTS', 'REB', 'AST']:
        load_or_train_model(player_name, stat, season)
    
    logger.info(f"Completed cache warmup for {player_name}")

def warmup_cache_for_players(player_names):
    """Pre-cache all data needed for a list of players"""
    start_time = time.time()
    logger.info("\nStarting cache warmup...")
    
    for player_name in player_names:
        bulk_cache_player_data(player_name, SEASON)
    
    end_time = time.time()
    logger.info(f"\nCache warmup completed in {end_time - start_time:.1f} seconds")

# Game Information Functions
def get_player_team(player_name):
    """Get player's current team abbreviation from their recent games"""
    recent_games = get_last_10_games(player_name)
    
    if recent_games is not None and not recent_games.empty:
        # Get the most recent game's matchup
        latest_matchup = recent_games['MATCHUP'].iloc[0]
        # Extract team abbreviation (handles both home and away games)
        team = latest_matchup.split()[0]
        return team
    logger.warning(f"Could not determine team for {player_name}")
    return None

def get_todays_matchups():
    """Returns a dictionary mapping team abbreviations to their opponents for today's games"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Get today's date in the format required by the API
            today = datetime.now().strftime('%Y-%m-%d')
            
            scoreboard_data = api_call_with_retry(
                lambda: scoreboardv2.ScoreboardV2(game_date=today).get_dict()
            )
            
            # Create ID to abbreviation mapping
            teams_list = teams.get_teams()
            team_id_to_abbrev = {team['id']: team['abbreviation'] 
                               for team in teams_list}
            
            matchups = {}
            
            # Get games from GameHeader resultSet
            game_headers = next(rs for rs in scoreboard_data['resultSets'] 
                              if rs['name'] == 'GameHeader')
            
            # Get indices for the columns we need
            headers = game_headers['headers']
            home_team_idx = headers.index('HOME_TEAM_ID')
            away_team_idx = headers.index('VISITOR_TEAM_ID')
            
            # Process each game
            for game in game_headers['rowSet']:
                home_team_id = game[home_team_idx]
                away_team_id = game[away_team_idx]
                
                # Convert IDs to abbreviations
                home_team = team_id_to_abbrev.get(home_team_id)
                away_team = team_id_to_abbrev.get(away_team_id)
                
                if home_team and away_team:
                    matchups[home_team] = away_team
                    matchups[away_team] = home_team
                else:
                    logger.warning(f"Could not convert IDs to abbreviations: {home_team_id} vs {away_team_id}")
            
            if not matchups:
                logger.warning(f"No games found in scoreboard data for {today}")
            
            return matchups
            
        except Exception as e:
            logger.error(f"Error getting matchups (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error("All retries failed when getting matchups")
                return {}

def get_opponent_and_home_status(player_name):
    """Returns tuple of (opponent, is_home_game) for today's game, or (None, None) if no game"""
    team = get_player_team(player_name)
    if not team:
        logger.warning(f"Could not determine team for {player_name}")  # Debug log
        return None, None
    
    matchups = get_todays_matchups()
    if not matchups:
        logger.warning("No matchups returned from get_todays_matchups")  # Debug log
        return None, None
        
    if team in matchups:
        opponent = matchups[team]
        # Check game location
        for home_team, away_team in matchups.items():
            if team == home_team and opponent == away_team:
                logger.info(f"{team} vs {opponent}")
                return opponent, True
            elif team == away_team and opponent == home_team:
                logger.info(f"{team} @ {opponent}")
                return opponent, False
    else:
        logger.warning(f"Team {team} not found in today's matchups: {matchups}")  # Debug log
    
    logger.info(f"No game found today for {player_name}'s team ({team})")
    return None, None

def calculate_rest_days(player_name, game_date=None):
    """Calculate rest days based on last game played"""
    from datetime import datetime, timedelta
    
    recent_games = get_last_10_games(player_name)
    if recent_games is None:
        return None
    
    last_game_date = pd.to_datetime(recent_games['GAME_DATE'].iloc[0])
    
    # If no game_date provided, use today's date
    if game_date is None:
        game_date = datetime.now()
    else:
        game_date = pd.to_datetime(game_date)
    
    rest_days = abs((game_date - last_game_date).days)
    logger.info(f"{player_name}'s last game: {last_game_date.strftime('%Y-%m-%d')}")
    logger.info(f"Predicting for: {game_date.strftime('%Y-%m-%d')}")
    logger.info(f"Rest days: {rest_days}")
    return rest_days

# Data Loading Functions
def load_or_fetch_player_games(player_name, season):
    """Load player games from local cache or fetch from API"""
    cache_dir = "cache/player_games"
    safe_name = player_name.replace(" ", "_")
    filename = f"{safe_name}_{season}.json"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename)
    if cached_data is not None:
        return pd.DataFrame(cached_data)
    
    # Fetch new data from API
    unlisted_players = {
        "victor wembanyama": 1641705,
        "amen thompson": 1641708
    }
    try:
        lowercase_name = player_name.lower()
        if lowercase_name in unlisted_players:
            player_id = unlisted_players[lowercase_name]
        else:
            player_dict = players.find_players_by_full_name(player_name)[0]
            player_id = player_dict['id']
        
        games = api_call_with_retry(lambda: playergamelog.PlayerGameLog(
            player_id=player_id, 
            season=season
        ).get_data_frames()[0])
        
        if games is not None:
            save_to_cache(cache_dir, filename, games.to_dict(orient='records'))
            return games
            
    except Exception as e:
        logger.error(f"Error fetching game log for {player_name}: {e}")
    return None

def load_or_fetch_team_games(team_id, season):
    """Load team games from local cache or fetch from API"""
    cache_dir = "cache/team_games"
    filename = f"team_{team_id}_{season}.json"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename)
    if cached_data is not None:
        return pd.DataFrame(cached_data)
    
    # Fetch new data from API
    logger.info(f"Fetching fresh team games for team {team_id}")
    try:
        team_games = api_call_with_retry(lambda: teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0])
        
        if team_games is not None:
            save_to_cache(cache_dir, filename, team_games.to_dict(orient='records'))
            return team_games
            
    except Exception as e:
        logger.error(f"Error fetching team games for team {team_id}: {e}")
    return None

def load_or_fetch_league_stats(season, stat_type='Base'):
    """Load league stats from local cache or fetch from API"""
    cache_dir = "cache/league_stats"
    filename = f"{season}_{stat_type}.json"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename)
    if cached_data is not None:
        return pd.DataFrame(cached_data)
    
    # Fetch new data from API
    logger.info(f"Fetching fresh league {stat_type} stats")
    try:
        stats = api_call_with_retry(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense=stat_type
        ).get_data_frames()[0])
        
        if stats is not None:
            save_to_cache(cache_dir, filename, stats.to_dict(orient='records'))
            return stats
            
    except Exception as e:
        logger.error(f"Error fetching league stats: {e}")
    return None

def load_or_fetch_season_averages(player_name, season=SEASON):
    """Load player season averages from local cache or calculate from cached games"""
    cache_dir = "cache/season_averages"
    safe_name = player_name.replace(" ", "_")
    filename = f"{safe_name}_{season}.json"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename)
    if cached_data is not None:
        return cached_data
    
    # Calculate from player games
    df = load_or_fetch_player_games(player_name, season)
    if df is None:
        return None
        
    df = df[df['MIN'] > 0]
    averages = {
        'PTS': round(df['PTS'].mean(), 1),
        'REB': round(df['REB'].mean(), 1),
        'AST': round(df['AST'].mean(), 1),
        'Games_Played': len(df)
    }
    
    save_to_cache(cache_dir, filename, averages)
    return averages

# Accessor Functions
def get_player_season_games(player_name, season=SEASON):
    return load_or_fetch_player_games(player_name, season)

def get_last_10_games(player_name):
    df = load_or_fetch_player_games(player_name, SEASON)
    if df is None:
        return None
        
    columns = ['MIN', 'MATCHUP', 'WL', 'FGM', 'FGA', 'FG3M', 'FG3A', 
               'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'GAME_DATE', 'PLAYER_NAME']
    
    if 'PLAYER_NAME' not in df.columns:
        df['PLAYER_NAME'] = player_name
        
    return df[columns].head(10)

def get_season_averages(player_name, season=SEASON):
    return load_or_fetch_season_averages(player_name, season)

# Game Data Functions
def get_opponent_points_from_game(game_id, team_id):
    """Get opponent points from a specific game using the boxscore endpoint"""
    cache_dir = "cache/boxscores"
    filename = f"boxscore_{game_id}.json"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename)
    if cached_data is not None:
        try:
            boxscore = pd.DataFrame(cached_data)
            opponent_data = boxscore[boxscore['TEAM_ID'] != team_id].groupby('TEAM_ID')['PTS'].sum().iloc[0]
            
            return opponent_data
        except Exception as e:
            logger.error(f"Error processing cached boxscore for game {game_id}: {e}")
    
    # Fetch new data from API
    try:
        boxscore = api_call_with_retry(lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=game_id
        ).get_data_frames()[0])
        
        if boxscore is not None:
            # Save to cache
            save_to_cache(cache_dir, filename, boxscore.to_dict(orient='records'))
            opponent_data = boxscore[boxscore['TEAM_ID'] != team_id].groupby('TEAM_ID')['PTS'].sum().iloc[0]
            
            return opponent_data
            
    except Exception as e:
        logger.error(f"Error getting game data for game {game_id}: {e}")
        return None

def get_enhanced_team_stats(team_id, season):
    """Get team rankings with proper relative rankings"""
    try:
        # Use cached team games
        team_games = load_or_fetch_team_games(team_id, season)
        
        # Get cached offensive and defensive stats
        offensive_stats = load_or_fetch_league_stats(season, 'Base')
        defensive_stats = load_or_fetch_league_stats(season, 'Defense')
        
        # Calculate recent points allowed from actual game results
        recent_points_allowed = []
        for idx, game in team_games.head(5).iterrows():
            points = get_opponent_points_from_game(game['Game_ID'], team_id)
            if points is not None:
                recent_points_allowed.append(points)
            # Only sleep if we had to make an API call
            if points is None:
                time.sleep(0.5)  # Reduced from 1 second since we're using cache
            
        if recent_points_allowed:
            recent_pts_allowed = sum(recent_points_allowed) / len(recent_points_allowed)
        else:
            # Fallback to defensive rating if we can't get actual points
            recent_pts_allowed = defensive_stats[defensive_stats['TEAM_ID'] == team_id].iloc[0]['DEF_RATING']
            
        # Add some random variation to recent_pts_allowed
        recent_pts_allowed = np.random.normal(recent_pts_allowed, recent_pts_allowed * 0.05)  # 5% standard deviation
        
        # Calculate rankings using offensive stats
        offensive_stats['PTS_RANK'] = offensive_stats['PTS'].rank(ascending=False)
        
        # Get specific team stats
        team_off_stats = offensive_stats[offensive_stats['TEAM_ID'] == team_id].iloc[0]
        team_def_stats = defensive_stats[defensive_stats['TEAM_ID'] == team_id].iloc[0]
        
        result = {
            'games_played': team_off_stats['GP'],
            'points_per_game': team_off_stats['PTS'],
            'points_rank': int(team_off_stats['PTS_RANK']),
            'def_rating': team_def_stats['DEF_RATING'],
            'recent_pts_allowed': recent_pts_allowed,
            'pace': np.random.normal(100.0, 2.0),  # Mean 100, SD 2
            'fg_pct_allowed': np.random.normal(0.47, 0.02)  # Mean 0.47, SD 0.02
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting team rankings: {e}")
        # Return default values with random variation
        return {
            'games_played': 50,
            'points_per_game': 110.0,
            'points_rank': 15,
            'def_rating': 110.0,
            'recent_pts_allowed': np.random.normal(110.0, 5.0),
            'pace': np.random.normal(100.0, 2.0),
            'fg_pct_allowed': np.random.normal(0.47, 0.02)
        }

def calculate_weighted_averages(df, col, weights=[0.3, 0.25, 0.20, 0.15, 0.10]):
    """Calculate weighted average using the previous 5 games for each row"""
    def calc_row_weighted_avg(idx):
        # Get the previous 5 games for this specific row
        prev_games = df.loc[idx+1:idx+5]
        # Filter for games with 20+ minutes
        prev_games = prev_games[prev_games['MIN'] >= 20]
        if len(prev_games) < 3:  # Require at least 3 games
            return None
            
        prev_values = prev_games[col].values
        # Get weights for the available number of games
        used_weights = weights[:len(prev_values)]
        # Normalize weights to sum to 1
        used_weights = np.array(used_weights) / sum(used_weights)
        
        # Calculate weighted average
        weighted_avg = sum(w * v for w, v in zip(used_weights, prev_values))
        
        # Debug output
        if idx == 0:  # Only print for the most recent calculation
            logger.debug(f"\nWeighted average calculation for {col}:")
            logger.debug(f"Previous values: {prev_values}")
            logger.debug(f"Weights used: {used_weights}")
            logger.debug(f"Weighted average: {weighted_avg:.2f}")
        
        return weighted_avg
    
    # Apply the calculation to each row
    return df.index.map(calc_row_weighted_avg)

def prepare_features(df, season, stat_to_predict):
    """
    Prepare features with consistent filtering and calculations
    """
    # Filter for games with 20+ minutes and sort by date
    df = df[df['MIN'] >= 20].copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
    df = df.sort_values('GAME_DATE', ascending=False)
    df = df.reset_index(drop=True)
    
    numeric_cols = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
    
    # Calculate rolling averages for each row
    for col in numeric_cols:
        # Calculate rolling 5-game average for each position
        for idx in range(len(df)):
            prev_games = df[col].iloc[idx:idx+5]
            if len(prev_games) >= 3:  # Require at least 3 games
                df.loc[idx, f'{col}_5game_avg'] = prev_games.mean()
            else:
                df.loc[idx, f'{col}_5game_avg'] = df[col].iloc[idx]
        
        # Calculate weighted average for each position
        weights = [0.25, 0.22, 0.20, 0.18, 0.15]
        for idx in range(len(df)):
            prev_games = df[col].iloc[idx:idx+5]
            if len(prev_games) >= 3:
                weighted_sum = sum(w * v for w, v in zip(weights[:len(prev_games)], prev_games))
                weight_sum = sum(weights[:len(prev_games)])
                df.loc[idx, f'{col}_weighted_avg'] = weighted_sum / weight_sum
            else:
                df.loc[idx, f'{col}_weighted_avg'] = df[col].iloc[idx]
        
        # Print debug info for the stat we're predicting
        if col == stat_to_predict:
            logger.info(f"\nLast 5 games {stat_to_predict} values:")
            logger.info(f"Values: {df[col].head().values}")
            logger.info(f"5-game average: {df[f'{col}_5game_avg'].iloc[0]:.3f}")
            logger.info(f"Weighted average: {df[f'{col}_weighted_avg'].iloc[0]:.3f}")
    
    # Add opponent context
    df['OPP_TEAM'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
    
    # Get opponent stats for each game
    for idx, row in df.iterrows():
        team = row['OPP_TEAM']
        if pd.isna(team):
            continue
            
        team_id = get_team_id_from_abbreviation(team)
        team_stats = get_enhanced_team_stats(team_id, season)
        
        if team_stats:
            df.loc[idx, 'OPP_DEF_RATING'] = team_stats['def_rating'] / 100
            df.loc[idx, 'OPP_PACE'] = team_stats['pace'] / 100
            df.loc[idx, 'OPP_RECENT_PTS_ALLOWED'] = team_stats['recent_pts_allowed'] / 100
            df.loc[idx, 'OPP_FG_PCT_ALLOWED'] = team_stats['fg_pct_allowed']
        else:
            # Use league averages as defaults, with some random variation
            df.loc[idx, 'OPP_DEF_RATING'] = np.random.normal(1.1, 0.05)  # Mean 110, SD 5
            df.loc[idx, 'OPP_PACE'] = np.random.normal(1.0, 0.05)  # Mean 100, SD 5
            df.loc[idx, 'OPP_RECENT_PTS_ALLOWED'] = np.random.normal(1.1, 0.05)
            df.loc[idx, 'OPP_FG_PCT_ALLOWED'] = np.random.normal(0.47, 0.02)
        
        time.sleep(1)  # Rate limiting
    
    # Add game context
    df['HOME'] = df['MATCHUP'].str.contains('vs').astype(int)
    df['DAYS_SINCE_LAST_GAME'] = df['GAME_DATE'].diff(-1).dt.days.fillna(3)
    
    # Add interaction features
    df['OPP_DEF_PACE'] = df['OPP_DEF_RATING'] * df['OPP_PACE']
    df['RECENT_VS_DEF'] = df[f'{stat_to_predict}_5game_avg'] * df['OPP_DEF_RATING']
    df['REST_HOME_INTER'] = df['DAYS_SINCE_LAST_GAME'] * df['HOME']
    
    return df

# Model Training Functions
def load_or_train_model(player_name, stat_to_predict, season=SEASON):
    """Load trained model from cache or train new one if needed"""
    cache_dir = "cache/models"
    safe_name = player_name.replace(" ", "_")
    filename = f"{safe_name}_{stat_to_predict}_{season}.pkl"
    
    # Try to load from cache
    cached_data = load_from_cache(cache_dir, filename, is_pickle=True)
    if cached_data is not None:
        return cached_data.get('model'), cached_data.get('features')

    logger.info(f"Training new model for {player_name} ({stat_to_predict})")
    df = get_player_season_games(player_name, season)
    if df is None:
        return None, None

    # Define base features
    base_features = [
        'HOME', 
        'DAYS_SINCE_LAST_GAME',
        'OPP_DEF_RATING', 
        'OPP_PACE',
        'OPP_RECENT_PTS_ALLOWED',
        'OPP_FG_PCT_ALLOWED',
        f'{stat_to_predict}_5game_avg',
        f'{stat_to_predict}_weighted_avg',
        'OPP_DEF_PACE',
        'RECENT_VS_DEF',
        'REST_HOME_INTER'
    ]
    
    df = prepare_features(df, season, stat_to_predict)
    if df is None or df.empty:
        return None, None
    
    X = df[base_features].copy()
    y = df[stat_to_predict].copy()
    
    # Create and train model
    if stat_to_predict in ['AST', 'REB']:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            base_score=y.mean(),
            random_state=42
        )
    else:
        best_params = tune_xgboost_params(X, y)
        model = XGBRegressor(**best_params, random_state=42)
    
    model.fit(X, y)
    
    # Save to cache
    model_data = {
        'model': model,
        'features': base_features
    }
    save_to_cache(cache_dir, filename, model_data, is_pickle=True)
    
    return model, base_features

def predict_next_game(player_name, stat_to_predict):
    """Make prediction for next game with automatic game context detection"""
    start_time = time.time()
    logger.info(f"\nStarting prediction for {player_name} ({stat_to_predict})...")
    
    # Get game context automatically
    opponent, is_home_game = get_opponent_and_home_status(player_name)
    if opponent is None:
        logger.warning(f"No game found for {player_name}")
        return None
        
    rest_days = calculate_rest_days(player_name)
    if rest_days is None:
        logger.warning(f"Could not calculate rest days for {player_name}")
        return None
    
    # Load or train the model
    model, feature_cols = load_or_train_model(player_name, stat_to_predict)
    if model is None:
        logger.error(f"Failed to create predictor for {player_name} after {time.time() - start_time:.2f} seconds")
        return None
        
    recent_games = get_last_10_games(player_name)
    if recent_games is None or recent_games.empty:
        logger.error(f"No recent games found for {player_name} after {time.time() - start_time:.2f} seconds")
        return None
    
    opponent_id = get_team_id_from_abbreviation(opponent)
    opponent_stats = get_enhanced_team_stats(opponent_id, SEASON)
    
    recent_games = prepare_features(recent_games, SEASON, stat_to_predict)
    if recent_games is None or recent_games.empty:
        logger.error(f"No valid games after feature preparation for {player_name} after {time.time() - start_time:.2f} seconds")
        return None
    
    # Prepare features for prediction
    pred_features = pd.DataFrame(index=[0])
    
    try:
        # Set up basic features
        for feat in feature_cols:
            if feat in ['OPP_DEF_PACE', 'RECENT_VS_DEF', 'REST_HOME_INTER']:
                continue
                
            if feat in recent_games.columns:
                pred_features[feat] = recent_games[feat].iloc[0]
            elif feat == 'HOME':
                pred_features[feat] = int(is_home_game)
            elif feat == 'DAYS_SINCE_LAST_GAME':
                pred_features[feat] = rest_days
            elif feat == 'OPP_DEF_RATING':
                pred_features[feat] = opponent_stats['def_rating'] / 100
            elif feat == 'OPP_PACE':
                pred_features[feat] = opponent_stats['pace'] / 100
            elif feat == 'OPP_RECENT_PTS_ALLOWED':
                pred_features[feat] = opponent_stats['recent_pts_allowed'] / 100
            elif feat == 'OPP_FG_PCT_ALLOWED':
                pred_features[feat] = opponent_stats['fg_pct_allowed']
        
        # Calculate interaction features
        if 'OPP_DEF_PACE' in feature_cols:
            pred_features['OPP_DEF_PACE'] = pred_features['OPP_DEF_RATING'] * pred_features['OPP_PACE']
        if 'RECENT_VS_DEF' in feature_cols:
            pred_features['RECENT_VS_DEF'] = pred_features[f'{stat_to_predict}_5game_avg'] * pred_features['OPP_DEF_RATING']
        if 'REST_HOME_INTER' in feature_cols:
            pred_features['REST_HOME_INTER'] = pred_features['DAYS_SINCE_LAST_GAME'] * pred_features['HOME']
        
        # Make predictions with uncertainty
        predictions = []
        for _ in range(100):
            sample_idx = np.random.choice(len(recent_games), size=len(recent_games), replace=True)
            sample_data = recent_games.iloc[sample_idx]
            sample_model = XGBRegressor()
            sample_model.fit(sample_data[feature_cols], sample_data[stat_to_predict])
            pred = sample_model.predict(pred_features[feature_cols])[0]
            predictions.append(pred)
        
        prediction = np.mean(predictions)
        confidence_interval = (np.percentile(predictions, 25), np.percentile(predictions, 75))
        
        end_time = time.time()
        logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
        
        return {
            'prediction': round(prediction, 1),
            'confidence_interval': confidence_interval,
            'season_average': get_season_averages(player_name)[stat_to_predict],
            'games_played': get_season_averages(player_name)['Games_Played'],
            'opponent': opponent,
            'is_home_game': is_home_game,
            'rest_days': rest_days,
            'opponent_stats': opponent_stats,
            'execution_time': end_time - start_time
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

def tune_xgboost_params(X, y):
    """Tune XGBoost parameters using cross-validation"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    best_score = -np.inf
    best_params = None
    
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for mcw in param_grid['min_child_weight']:
                    for ss in param_grid['subsample']:
                        model = XGBRegressor(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            min_child_weight=mcw,
                            subsample=ss,
                            random_state=42
                        )
                        
                        scores = cross_val_score(
                            model, X, y,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            scoring='neg_mean_squared_error'
                        )
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'min_child_weight': mcw,
                                'subsample': ss
                            }
    
    return best_params

players_list = [
    "Karl-Anthony Towns",
    "Cade Cunningham",
    "Jalen Duren",
    "Josh Giddey",
    "Nicola Vucevic",
    "Jaren Jackson Jr.",
    "Kevin Durant",
    "Devin Booker",
    "Ja Morant",
    "Desmond Bane",
]

def predict_multiple_players(player_names):
    """Make predictions for multiple players and write to CSV in batches"""
    stats_to_predict = ['PTS', 'REB', 'AST']
    
    for player_name in player_names:
        try:
            opponent, is_home_game = get_opponent_and_home_status(player_name)
            
            if not opponent:
                logger.info(f"\nNo game found today for {player_name}")
                continue
                
            rest_days = calculate_rest_days(player_name)
            logger.info(f"\nPredictions for {player_name} vs {opponent}")
            logger.info(f"Location: {'Home' if is_home_game else 'Away'}")
            logger.info(f"Rest days: {rest_days}")
            
            # Get season averages
            season_avgs = get_season_averages(player_name)
            recent_games = get_last_10_games(player_name)
            
            player_prediction = {
                'name': player_name,
                'opponent': opponent,
                'is_home_game': is_home_game,
                'rest_days': rest_days,
                'season_ppg': season_avgs.get('PTS', ''),
                'season_rpg': season_avgs.get('REB', ''),
                'season_apg': season_avgs.get('AST', ''),
            }
            
            # Calculate last 5 game averages
            if recent_games is not None and not recent_games.empty:
                last_5 = recent_games.head(5)
                player_prediction.update({
                    'l5_ppg': round(last_5['PTS'].mean(), 1),
                    'l5_rpg': round(last_5['REB'].mean(), 1),
                    'l5_apg': round(last_5['AST'].mean(), 1),
                })
            
            for stat in stats_to_predict:
                logger.info(f"Predicting {stat} for {player_name}...")
                result = predict_next_game(player_name, stat)
                if result is not None:
                    player_prediction[stat] = result['prediction']
                    logger.info(f"{stat}: {result['prediction']}")
                    logger.info(f"Season Average: {result['season_average']}")
                else:
                    player_prediction[stat] = None
                    logger.error(f"Unable to predict {stat}")
            
            # Write single player prediction to CSV
            write_predictions_to_csv([player_prediction])
            logger.info(f"Successfully wrote predictions for {player_name} to CSV")
            
        except Exception as e:
            logger.error(f"Error making predictions for {player_name}: {e}")
            logger.error("Moving on to next player...")
            continue
    
    logger.info("\nCompleted all predictions")

def write_predictions_to_csv(predictions, filename='predictions-tracking.csv'):
    import csv
    from datetime import datetime
    import os
    
    today = datetime.now().strftime("%-m/%-d/%y")
    headers = [
        'Date', 'Player', 'Opponent', 'Home/Away', 'Rest Days',
        'Season PPG', 'L5 PPG', 'Predicted PTS', 'Actual PTS',
        'Season RPG', 'L5 RPG', 'Predicted REB', 'Actual REB',
        'Season APG', 'L5 APG', 'Predicted AST', 'Actual AST'
    ]
    
    try:
        # Check if file exists and needs headers
        file_exists = os.path.exists(filename)
        need_headers = not file_exists
        
        # Check if file needs a newline
        need_newline = False
        if file_exists:
            with open(filename, 'r') as f:
                f.seek(0, 2)  # Seek to end
                if f.tell() > 0:  # If file is not empty
                    f.seek(f.tell() - 1)  # Seek to last character
                    need_newline = f.read(1) != '\n'
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            rows_written = 0
            
            # Add newline if needed
            if need_newline:
                f.write('\n')
            
            # Write headers if new file
            if need_headers:
                writer.writerow(headers)
            
            for pred in predictions:
                # Get season averages
                season_avgs = get_season_averages(pred['name'])
                if not season_avgs:
                    season_avgs = {'PTS': 0, 'REB': 0, 'AST': 0}
                
                # Get last 5 games averages
                last_10_games = get_last_10_games(pred['name'])
                if last_10_games is not None and not last_10_games.empty:
                    # Filter for 20+ minutes before calculating averages
                    last_5_games = last_10_games[last_10_games['MIN'] >= 20].head(5)
                    l5_pts = round(last_5_games['PTS'].mean(), 1)
                    l5_reb = round(last_5_games['REB'].mean(), 1)
                    l5_ast = round(last_5_games['AST'].mean(), 1)
                else:
                    l5_pts = l5_reb = l5_ast = 0
                
                # Get opponent and home/away status
                _, is_home = get_opponent_and_home_status(pred['name'])
                home_away = 'Home' if is_home else 'Away'
                
                # Get rest days
                rest_days = calculate_rest_days(pred['name'])
                
                row = [
                    today,
                    pred['name'],
                    pred['opponent'],
                    home_away,
                    rest_days,
                    season_avgs['PTS'],
                    l5_pts,
                    pred['PTS'],
                    '',  # Actual PTS
                    season_avgs['REB'],
                    l5_reb,
                    pred['REB'],
                    '',  # Actual REB
                    season_avgs['AST'],
                    l5_ast,
                    pred['AST'],
                    ''   # Actual AST
                ]
                writer.writerow(row)
                rows_written += 1
            
            print(f"\nSuccessfully wrote {rows_written} predictions to {filename}")
            
    except Exception as e:
        print(f"\nError writing to file {filename}: {e}")

def update_previous_predictions(csv_file='predictions-tracking.csv'):
    """Update any missing actual stats in the predictions CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.info("No predictions to update")
            return
            
        # Find rows with missing actual stats
        mask = (
            (df['Actual PTS'].isna()) |
            (df['Actual REB'].isna()) |
            (df['Actual AST'].isna())
        )
        missing_stats = df[mask]
        
        if missing_stats.empty:
            logger.info("No missing stats to update")
            return
            
        update_count = 0
        logger.info("\nChecking for missing stats to update:")
        
        # For each row with missing stats
        for idx, row in missing_stats.iterrows():
            player_name = row['Player']
            pred_date = pd.to_datetime(row['Date']).strftime('%-m/%-d/%y')
            
            # Get player's recent games
            recent_games = get_last_10_games(player_name)
            if recent_games is None or recent_games.empty:
                logger.warning(f"No recent games found for {player_name}")
                continue
                
            # Check each recent game for matching date
            for _, game in recent_games.iterrows():
                game_date = pd.to_datetime(game['GAME_DATE']).strftime('%-m/%-d/%y')
                
                if game_date == pred_date:
                    logger.info(f"\nFound matching game for {player_name} on {pred_date}:")
                    logger.info(f"Predicted PTS: {row['Predicted PTS']}, Actual: {game['PTS']}")
                    logger.info(f"Predicted REB: {row['Predicted REB']}, Actual: {game['REB']}")
                    logger.info(f"Predicted AST: {row['Predicted AST']}, Actual: {game['AST']}")
                    
                    # Update the actual values
                    df.loc[idx, 'Actual PTS'] = game['PTS']
                    df.loc[idx, 'Actual REB'] = game['REB']
                    df.loc[idx, 'Actual AST'] = game['AST']
                    update_count += 1
                    break  # Found the matching game, move to next player
        
        if update_count > 0:
            # Write the updated dataframe back to CSV
            df.to_csv(csv_file, index=False)
            logger.info(f"\nSuccessfully updated {update_count} missing stat lines")
        else:
            logger.info("\nNo matching games found to update stats")
            
    except FileNotFoundError:
        logger.error(f"Predictions file not found: {csv_file}")
    except Exception as e:
        logger.error(f"Error updating predictions: {e}")

def analyze_predictions(input_csv='predictions-tracking.csv', output_csv='prediction-analysis.csv'):
    """Analyze prediction accuracy and output results to CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv)
        if df.empty:
            logger.info("No predictions to analyze")
            return
            
        # Filter for rows with both predictions and actual values
        complete_preds = df.dropna(subset=['Predicted PTS', 'Actual PTS', 
                                         'Predicted REB', 'Actual REB',
                                         'Predicted AST', 'Actual AST'])
        
        if complete_preds.empty:
            logger.info("No completed predictions to analyze")
            return
            
        logger.info("\nAnalyzing predictions...")
        analysis_results = []
        
        # Overall analysis
        analysis_results.append({
            'Category': 'Overall',
            'Metric': 'Total Predictions',
            'PTS': len(complete_preds),
            'REB': len(complete_preds),
            'AST': len(complete_preds)
        })
        
        # Analyze each stat
        for stat in ['PTS', 'REB', 'AST']:
            pred_col = f'Predicted {stat}'
            actual_col = f'Actual {stat}'
            
            # Calculate differences
            complete_preds[f'{stat} Diff'] = complete_preds[actual_col] - complete_preds[pred_col]
            
            # Calculate metrics
            mae = abs(complete_preds[f'{stat} Diff']).mean()
            rmse = np.sqrt((complete_preds[f'{stat} Diff'] ** 2).mean())
            
            # Calculate accuracy within different ranges
            within_1 = (abs(complete_preds[f'{stat} Diff']) <= 1).mean() * 100
            within_2 = (abs(complete_preds[f'{stat} Diff']) <= 2).mean() * 100
            within_3 = (abs(complete_preds[f'{stat} Diff']) <= 3).mean() * 100
            
            # Add overall metrics
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Mean Absolute Error',
                stat: round(mae, 2)
            })
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Root Mean Square Error',
                stat: round(rmse, 2)
            })
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Within 1',
                stat: f"{round(within_1, 1)}%"
            })
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Within 2',
                stat: f"{round(within_2, 1)}%"
            })
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Within 3',
                stat: f"{round(within_3, 1)}%"
            })
            
            # Add biggest misses
            biggest_misses = complete_preds.nlargest(3, f'{stat} Diff')
            for i, row in biggest_misses.iterrows():
                analysis_results.append({
                    'Category': f'{stat} Overestimates',
                    'Metric': f"#{len(analysis_results) % 3 + 1}",
                    'Player': row['Player'],
                    'Date': row['Date'],
                    'Predicted': row[pred_col],
                    'Actual': row[actual_col],
                    'Difference': row[f'{stat} Diff']
                })
            
            biggest_underestimates = complete_preds.nsmallest(3, f'{stat} Diff')
            for i, row in biggest_underestimates.iterrows():
                analysis_results.append({
                    'Category': f'{stat} Underestimates',
                    'Metric': f"#{len(analysis_results) % 3 + 1}",
                    'Player': row['Player'],
                    'Date': row['Date'],
                    'Predicted': row[pred_col],
                    'Actual': row[actual_col],
                    'Difference': row[f'{stat} Diff']
                })
        
        # Player-specific analysis
        for player in complete_preds['Player'].unique():
            player_preds = complete_preds[complete_preds['Player'] == player]
            player_stats = {}
            
            for stat in ['PTS', 'REB', 'AST']:
                mae = abs(player_preds[f'{stat} Diff']).mean()
                player_stats[stat] = round(mae, 2)
            
            analysis_results.append({
                'Category': 'Player Analysis',
                'Metric': 'Mean Absolute Error',
                'Player': player,
                'PTS': player_stats['PTS'],
                'REB': player_stats['REB'],
                'AST': player_stats['AST'],
                'Sample Size': len(player_preds)
            })
        
        # Convert to DataFrame and save
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(output_csv, index=False)
        logger.info(f"Analysis saved to {output_csv}")
        
        # Also display key metrics in logs
        logger.info("\nKey Metrics Summary:")
        for stat in ['PTS', 'REB', 'AST']:
            mae = analysis_df[analysis_df['Metric'] == 'Mean Absolute Error'][stat].iloc[0]
            within_2 = analysis_df[analysis_df['Metric'] == 'Within 2'][stat].iloc[0]
            logger.info(f"{stat} - MAE: {mae}, Within 2: {within_2}")
        
    except FileNotFoundError:
        logger.error(f"Predictions file not found: {input_csv}")
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")

def main():
    """Main function to handle different command line arguments"""
    parser = argparse.ArgumentParser(description='NBA Player Prediction Tool')
    parser.add_argument('action', choices=['predict', 'update', 'analyze'],
                       help='Action to perform: predict new stats, update previous stats, or analyze results')
    parser.add_argument('--players', default="Josh Hart,Jalen Brunson,Pascal Siakam",
                       help='Comma-separated list of players to predict (for predict action)')
    
    args = parser.parse_args()
    
    if args.action == 'update':
        logger.info("Updating previous predictions...")
        update_previous_predictions()
    
    elif args.action == 'predict':
        # Convert comma-separated string to list
        players_list = [p.strip() for p in args.players.split(',')]
        logger.info(f"\nMaking predictions for: {', '.join(players_list)}")
        predict_multiple_players(players_list)
        
    elif args.action == 'analyze':
        logger.info("Analyzing prediction accuracy...")
        analyze_predictions()

if __name__ == "__main__":
    main()