import os
import json
import pickle
import time
import logging
import argparse
from datetime import datetime
import requests
from requests.exceptions import ReadTimeout
from typing import Dict, List, Any
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
from players import players as active_players

# Constants
SEASON = '2024-25'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama Configuration
OLLAMA_ENABLED = False  # Set to True after installing Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:latest"

class OllamaManager:
    """Manages LLM operations using Ollama for NBA predictions"""
    
    def __init__(self, base_url=OLLAMA_URL, model=OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.available = False
        
        # Check if Ollama is available
        if OLLAMA_ENABLED:
            try:
                # Test connection to Ollama
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    self.available = True
                    logger.info(f"Ollama initialized successfully with model: {model}")
                    available_models = response.json().get('models', [])
                    model_names = [m.get('name') for m in available_models]
                    if model not in model_names:
                        logger.warning(f"Model '{model}' not found in Ollama. Available models: {model_names}")
                        logger.warning(f"Install with: ollama pull {model}")
                        self.available = False
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                logger.error("Make sure Ollama is installed and running.")
                self.available = False
    
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        """Generate text via Ollama API"""
        if not self.available:
            return "LLM analysis not available."
            
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error generating analysis: {str(e)}"

    def analyze_matchup(self, player_name: str, 
                      opponent: str, 
                      recent_games: pd.DataFrame,
                      season_stats: Dict[str, Any],
                      is_home: bool) -> str:
        """Generate contextual analysis for a matchup"""
        if not self.available:
            return "LLM analysis not available."
            
        # Format recent games for the prompt
        recent_games_text = self._format_recent_games(recent_games)
        
        # Format home/away status
        location = "at home against" if is_home else "on the road against"
        
        # Updated to include BLK, STL, and TOV
        prompt = f"""
        You are an expert basketball analyst. Analyze the upcoming NBA matchup for {player_name} {location} {opponent}.
        
        Recent performance for {player_name}:
        {recent_games_text}
        
        Season averages:
        PTS: {season_stats.get('PTS', 'N/A')}
        REB: {season_stats.get('REB', 'N/A')}
        AST: {season_stats.get('AST', 'N/A')}
        BLK: {season_stats.get('BLK', 'N/A')}
        STL: {season_stats.get('STL', 'N/A')}
        TOV: {season_stats.get('TOV', 'N/A')}
        Games Played: {season_stats.get('Games_Played', 'N/A')}
        
        Based on this information, what factors might impact {player_name}'s performance against {opponent}? 
        What statistical trends do you notice? Are there any matchup considerations that could affect scoring, rebounding,
        assists, blocks, steals, or turnovers?
        
        Keep your analysis concise (3-4 sentences) and focused on relevant performance factors.
        """
        
        return self.generate(prompt, max_tokens=300, temperature=0.3)
    
    def explain_prediction(self, 
                         player_name: str,
                         stat: str, 
                         prediction: Dict[str, Any],
                         recent_games: pd.DataFrame) -> str:
        """Generate explanation for a specific prediction"""
        if not self.available:
            return "LLM explanation not available."
            
        pred_value = prediction['prediction']
        season_avg = prediction['season_average']
        conf_interval = prediction['confidence_interval']
        
        # Calculate if prediction is above/below season average
        diff = pred_value - season_avg
        trend = "above" if diff > 0 else "below"
        
        # Customize prompt based on stat type
        if stat == 'BLK':
            stat_context = "blocks"
            defensive_context = f"- Opponent shots at the rim per game: {prediction['opponent_stats'].get('fg_pct_allowed', 0) * 100:.1f}%"
        elif stat == 'STL':
            stat_context = "steals"
            defensive_context = f"- Opponent turnover rate: {prediction['opponent_stats'].get('def_rating', 110) / 1.1:.1f}"
        elif stat == 'TOV':
            stat_context = "turnovers"
            defensive_context = f"- Opponent defensive pressure rating: {prediction['opponent_stats'].get('def_rating', 110) / 1.1:.1f}"
        else:
            stat_context = stat.lower()
            defensive_context = f"- Opponent defensive rating: {prediction['opponent_stats']['def_rating']:.1f}"
        
        prompt = f"""
        You are an expert basketball analyst. Explain why our model predicted {player_name} will record {pred_value} {stat_context} 
        (confidence interval: {conf_interval[0]:.1f}-{conf_interval[1]:.1f}) against {prediction['opponent']}.
        
        This prediction is {abs(diff):.1f} {trend} their season average of {season_avg}.
        
        Key context:
        - Playing {'at home' if prediction['is_home_game'] else 'away'}
        - {prediction['rest_days']} days of rest
        {defensive_context}
        
        Based on this context and the prediction, provide a concise 2-3 sentence explanation that a sports fan would find insightful.
        Focus on the most important factors and their relationships relevant to {stat_context}.
        """
        
        return self.generate(prompt, max_tokens=200, temperature=0.4)
    
    def generate_report(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate an overview report for multiple predictions"""
        if not self.available:
            return "LLM report not available."
            
        # Format predictions for the prompt
        predictions_text = ""
        for pred in predictions:
            predictions_text += f"{pred['name']} vs {pred['opponent']} ({'Home' if pred['is_home_game'] else 'Away'})\n"
            # Updated to include BLK, STL, TOV
            predictions_text += f"Predicted: PTS: {pred.get('PTS', 'N/A')}, REB: {pred.get('REB', 'N/A')}, AST: {pred.get('AST', 'N/A')}, "
            predictions_text += f"BLK: {pred.get('BLK', 'N/A')}, STL: {pred.get('STL', 'N/A')}, TOV: {pred.get('TOV', 'N/A')}\n"
            predictions_text += f"Season Avg: PTS: {pred.get('season_ppg', 'N/A')}, REB: {pred.get('season_rpg', 'N/A')}, AST: {pred.get('season_apg', 'N/A')}, "
            predictions_text += f"BLK: {pred.get('season_bpg', 'N/A')}, STL: {pred.get('season_spg', 'N/A')}, TOV: {pred.get('season_topg', 'N/A')}\n\n"
        
        prompt = f"""
        You are an expert basketball analyst. Based on the following player predictions for today's games, 
        write a brief summary report highlighting the most interesting predictions and potential storylines.
        
        Player Predictions:
        {predictions_text}
        
        Write a short (3-4 paragraph) report that a sports fan would find interesting. 
        Highlight the most notable predictions, such as players projected well above/below their averages, 
        interesting matchups, or players in favorable situations. Include any notable block, steal or turnover predictions.
        """
        
        return self.generate(prompt, max_tokens=500, temperature=0.7)
    
    def _format_recent_games(self, games_df):
        """Format recent games data for the prompt"""
        if games_df is None or games_df.empty:
            return "No recent games data available."
        
        try:
            # Updated to include BLK, STL, TOV
            games = games_df.head(5)[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV', 'MIN']]
            return games.to_string(index=False)
        except Exception as e:
            logger.error(f"Error formatting recent games: {e}")
            return "Error formatting recent games data."

# Initialize the LLM manager
llm_manager = OllamaManager() if OLLAMA_ENABLED else None

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
    for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']:  # Add the new stats here
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
    try:
        name = player_name.lower()
        if name in active_players:
            player_id = active_players[name]
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
        'BLK': round(df['BLK'].mean(), 1),
        'STL': round(df['STL'].mean(), 1),
        'TOV': round(df['TOV'].mean(), 1),
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
            #TODO: Add pace and fg_pct_allowed
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
    # For BLK and STL, which tend to be low-count stats with high variance
    elif stat_to_predict in ['BLK', 'STL']:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,  # Lower depth to prevent overfitting on rare events
            learning_rate=0.03,
            min_child_weight=5,  # Higher to stabilize predictions
            subsample=0.8,
            base_score=y.mean(),
            gamma=1.0,  # Add regularization for rare events
            random_state=42
        )
    # For TOV, which has different distribution patterns
    elif stat_to_predict in ['TOV']:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=4,
            subsample=0.85,
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
        
        # Get the final result
        result = {
            'prediction': round(prediction, 1),
            'confidence_interval': confidence_interval,
            'season_average': get_season_averages(player_name)[stat_to_predict],
            'games_played': get_season_averages(player_name)['Games_Played'],
            'opponent': opponent,
            'is_home_game': is_home_game,
            'rest_days': rest_days,
            'opponent_stats': opponent_stats,
            'execution_time': time.time() - start_time
        }
        
        # Add LLM explanation if available
        if OLLAMA_ENABLED and llm_manager and llm_manager.available:
            try:
                season_stats = get_season_averages(player_name)
                
                # Generate explanation for this specific prediction
                explanation = llm_manager.explain_prediction(
                    player_name, 
                    stat_to_predict, 
                    result,
                    get_last_10_games(player_name)
                )
                
                # Add to result
                result['llm_explanation'] = explanation
                logger.info(f"LLM Explanation: {explanation}")
            except Exception as e:
                logger.error(f"Error generating LLM explanation: {e}")
                result['llm_explanation'] = f"Error generating explanation: {str(e)}"
        
        end_time = time.time()
        logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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


def predict_multiple_players(player_names):
    """Make predictions for multiple players and write to CSV in batches"""
    stats_to_predict = ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']
    all_predictions = []
    
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
                'season_bpg': season_avgs.get('BLK', ''),
                'season_spg': season_avgs.get('STL', ''),
                'season_topg': season_avgs.get('TOV', ''),
            }
            
            # Calculate last 5 game averages
            if recent_games is not None and not recent_games.empty:
                last_5 = recent_games.head(5)
                player_prediction.update({
                    'l5_ppg': round(last_5['PTS'].mean(), 1),
                    'l5_rpg': round(last_5['REB'].mean(), 1),
                    'l5_apg': round(last_5['AST'].mean(), 1),
                    'l5_bpg': round(last_5['BLK'].mean(), 1),
                    'l5_spg': round(last_5['STL'].mean(), 1),
                    'l5_topg': round(last_5['TOV'].mean(), 1),
                })
            
            # Generate LLM matchup analysis if enabled
            if OLLAMA_ENABLED and llm_manager and llm_manager.available:
                try:
                    matchup_analysis = llm_manager.analyze_matchup(
                        player_name,
                        opponent,
                        recent_games,
                        season_avgs,
                        is_home_game
                    )
                    player_prediction['matchup_analysis'] = matchup_analysis
                    logger.info(f"Matchup Analysis: {matchup_analysis}")
                except Exception as e:
                    logger.error(f"Error generating matchup analysis: {e}")
            
            # Get predictions for each stat
            for stat in stats_to_predict:
                logger.info(f"Predicting {stat} for {player_name}...")
                result = predict_next_game(player_name, stat)
                if result is not None:
                    player_prediction[stat] = result['prediction']
                    if 'llm_explanation' in result:
                        player_prediction[f'{stat}_explanation'] = result['llm_explanation']
                    logger.info(f"{stat}: {result['prediction']}")
                    logger.info(f"Season Average: {result['season_average']}")
                else:
                    player_prediction[stat] = None
                    logger.error(f"Unable to predict {stat}")
            
            # After all stat predictions are completed, calculate fantasy points
            if all(stat in player_prediction for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']):
                fantasy_pts = calculate_fantasy_points(player_prediction)
                player_prediction['Fantasy Pts'] = fantasy_pts
                logger.info(f"Fantasy Points: {fantasy_pts}")

            # Add to predictions list
            all_predictions.append(player_prediction)
            
            # Write single player prediction to CSV
            write_predictions_to_csv([player_prediction])
            logger.info(f"Successfully wrote predictions for {player_name} to CSV")
            
        except Exception as e:
            logger.error(f"Error making predictions for {player_name}: {e}")
            logger.error("Moving on to next player...")
            continue
    
    # Generate overall report if we have predictions and LLM is enabled
    if OLLAMA_ENABLED and llm_manager and llm_manager.available and all_predictions:
        try:
            overall_report = llm_manager.generate_report(all_predictions)
            logger.info("\nOverall Analysis Report:")
            logger.info(overall_report)
            
            # Save report to file
            with open('prediction-report.txt', 'w') as f:
                f.write(overall_report)
            logger.info("Saved overall report to prediction-report.txt")
        except Exception as e:
            logger.error(f"Error generating overall report: {e}")
    
    logger.info("\nCompleted all predictions")

def calculate_fantasy_points(stats):
    all_stats = [
        stats.get('PTS', 0) * 1.0,
        stats.get('REB', 0) * 1.2,
        stats.get('AST', 0) * 1.5,
        stats.get('BLK', 0) * 3.0,
        stats.get('STL', 0) * 3.0,
        stats.get('TOV', 0) * -1.0
    ]
    
    return round(sum(all_stats), 2)

def write_predictions_to_csv(predictions, filename='predictions-tracking.csv'):
    import csv
    from datetime import datetime
    import os
    
    today = datetime.now().strftime("%-m/%-d/%y")
    # Updated headers to include BLK, STL, TOV
    headers = [
        'Date', 'Player', 'Vs.', 'Loc', 'Rest',
        'Avg PPG', 'L5 PPG', 'Pred PTS', 'Actual PTS',
        'Avg RPG', 'L5 RPG', 'Pred REB', 'Actual REB',
        'Avg APG', 'L5 APG', 'Pred AST', 'Actual AST',
        'Avg BPG', 'L5 BPG', 'Pred BLK', 'Actual BLK',
        'Avg SPG', 'L5 SPG', 'Pred STL', 'Actual STL',
        'Avg TOPG', 'L5 TOPG', 'Pred TOV', 'Actual TOV',
        'Avg Fantasy Pts', 'L5 Fantasy Pts', 'Pred Fantasy Pts',
        'Actual Fantasy Pts'
    ]
    
    # If we have LLM analysis, create a separate file for it
    has_analysis = any('matchup_analysis' in pred or 
                      'PTS_explanation' in pred or 
                      'REB_explanation' in pred or 
                      'AST_explanation' in pred or
                      'BLK_explanation' in pred or
                      'STL_explanation' in pred or
                      'TOV_explanation' in pred
                      for pred in predictions)
    
    if has_analysis:
        # Updated analysis headers to include BLK, STL, TOV
        analysis_headers = ['Date', 'Player', 'Opponent', 'Matchup Analysis', 
                           'PTS Explanation', 'REB Explanation', 'AST Explanation',
                           'BLK Explanation', 'STL Explanation', 'TOV Explanation']
        analysis_rows = []
    
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
                    season_avgs = {'PTS': 0, 'REB': 0, 'AST': 0, 'BLK': 0, 'STL': 0, 'TOV': 0}
                
                # Get last 5 games averages
                last_10_games = get_last_10_games(pred['name'])
                if last_10_games is not None and not last_10_games.empty:
                    # Filter for 20+ minutes before calculating averages
                    last_5_games = last_10_games[last_10_games['MIN'] >= 20].head(5)
                    l5_pts = round(last_5_games['PTS'].mean(), 1)
                    l5_reb = round(last_5_games['REB'].mean(), 1)
                    l5_ast = round(last_5_games['AST'].mean(), 1)
                    l5_blk = round(last_5_games['BLK'].mean(), 1)
                    l5_stl = round(last_5_games['STL'].mean(), 1)
                    l5_tov = round(last_5_games['TOV'].mean(), 1)
                else:
                    l5_pts = l5_reb = l5_ast = l5_blk = l5_stl = l5_tov = 0
                
                # Get opponent and home/away status
                _, is_home = get_opponent_and_home_status(pred['name'])
                home_away = 'Home' if is_home else 'Away'
                
                # Get rest days
                rest_days = calculate_rest_days(pred['name'])

                # Calculate fantasy points
                l5_dict = {
                    'PTS': l5_pts,
                    'REB': l5_reb,
                    'AST': l5_ast,
                    'BLK': l5_blk,
                    'STL': l5_stl,
                    'TOV': l5_tov
                }
                season_fantasy_avg = calculate_fantasy_points(season_avgs)
                predicted_fantasy_pts = calculate_fantasy_points(pred)
                l5_fantasy_avg = calculate_fantasy_points(l5_dict)

                # Updated row to include BLK, STL, TOV
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
                    '',  # Actual AST
                    season_avgs['BLK'],
                    l5_blk,
                    pred['BLK'],
                    '',  # Actual BLK
                    season_avgs['STL'],
                    l5_stl,
                    pred['STL'],
                    '',  # Actual STL
                    season_avgs['TOV'],
                    l5_tov,
                    pred['TOV'],
                    '',   # Actual TOV
                    season_fantasy_avg,
                    l5_fantasy_avg,
                    predicted_fantasy_pts,
                    '',   # Actual fantasy pts
                ]
                writer.writerow(row)
                rows_written += 1
                
                # Add to analysis rows if we have LLM analysis
                if has_analysis:
                    analysis_rows.append([
                        today,
                        pred['name'],
                        pred['opponent'],
                        pred.get('matchup_analysis', ''),
                        pred.get('PTS_explanation', ''),
                        pred.get('REB_explanation', ''),
                        pred.get('AST_explanation', ''),
                        pred.get('BLK_explanation', ''),
                        pred.get('STL_explanation', ''),
                        pred.get('TOV_explanation', '')
                    ])
            
            print(f"\nSuccessfully wrote {rows_written} predictions to {filename}")
            
        # Write LLM analysis to a separate file if we have it
        if has_analysis and analysis_rows:
            analysis_file = 'predictions-analysis.csv'
            analysis_exists = os.path.exists(analysis_file)
            
            with open(analysis_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Write headers if new file
                if not analysis_exists:
                    writer.writerow(analysis_headers)
                
                # Write analysis rows
                for row in analysis_rows:
                    writer.writerow(row)
                
                print(f"Wrote LLM analysis to {analysis_file}")
            
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
            
        # Find rows with missing actual stats - updated to include BLK, STL, TOV
        mask = (
            (df['Actual PTS'].isna()) |
            (df['Actual REB'].isna()) |
            (df['Actual AST'].isna()) |
            (df['Actual BLK'].isna()) |
            (df['Actual STL'].isna()) |
            (df['Actual TOV'].isna()) |
            (df['Actual Fantasy Pts'].isna())
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
                    fantasy_pts = calculate_fantasy_points(game)
                    logger.info(f"\nFound matching game for {player_name} on {pred_date}:")
                    logger.info(f"Predicted PTS: {row['Pred PTS']}, Actual: {game['PTS']}")
                    logger.info(f"Predicted REB: {row['Pred REB']}, Actual: {game['REB']}")
                    logger.info(f"Predicted AST: {row['Pred AST']}, Actual: {game['AST']}")
                    logger.info(f"Predicted BLK: {row['Pred BLK']}, Actual: {game['BLK']}")
                    logger.info(f"Predicted STL: {row['Pred STL']}, Actual: {game['STL']}")
                    logger.info(f"Predicted TOV: {row['Pred TOV']}, Actual: {game['TOV']}")
                    logger.info(f"Predicted Fantasy Pts: {row['Pred Fantasy Pts']}, Actual: {fantasy_pts}")
                    
                    # Update the actual values - including new stats
                    df.loc[idx, 'Actual PTS'] = game['PTS']
                    df.loc[idx, 'Actual REB'] = game['REB']
                    df.loc[idx, 'Actual AST'] = game['AST']
                    df.loc[idx, 'Actual BLK'] = game['BLK']
                    df.loc[idx, 'Actual STL'] = game['STL']
                    df.loc[idx, 'Actual TOV'] = game['TOV']
                    df.loc[idx, 'Actual Fantasy Pts'] = fantasy_pts
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
            
        # Filter for rows with both predictions and actual values - updated to include Fantasy Pts
        complete_preds = df.dropna(subset=['Pred PTS', 'Actual PTS', 
                                         'Pred REB', 'Actual REB',
                                         'Pred AST', 'Actual AST',
                                         'Pred BLK', 'Actual BLK',
                                         'Pred STL', 'Actual STL',
                                         'Pred TOV', 'Actual TOV',
                                         'Pred Fantasy Pts', 'Actual Fantasy Pts'
                                         ])
        
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
            'AST': len(complete_preds),
            'BLK': len(complete_preds),
            'STL': len(complete_preds),
            'TOV': len(complete_preds),
            'Fantasy': len(complete_preds)
        })
        
        # Analyze each stat - updated to include Fantasy Pts
        for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV', 'Fantasy Pts']:
            pred_col = f'Pred {stat}'
            actual_col = f'Actual {stat}'
            
            # Calculate differences
            complete_preds[f'{stat} Diff'] = complete_preds[actual_col] - complete_preds[pred_col]
            
            # Calculate metrics
            mae = abs(complete_preds[f'{stat} Diff']).mean()
            rmse = np.sqrt((complete_preds[f'{stat} Diff'] ** 2).mean())
            
            # For Fantasy Points, use different thresholds since they have larger values
            if stat == 'Fantasy Pts':
                within_ranges = [5, 10, 15]  # 5, 10, 15 points for fantasy
            else:
                within_ranges = [1, 2, 3]    # 1, 2, 3 for regular stats
            
            # Calculate accuracy within different ranges
            within_metrics = {}
            for threshold in within_ranges:
                percentage = (abs(complete_preds[f'{stat} Diff']) <= threshold).mean() * 100
                within_metrics[threshold] = percentage
            
            # Add overall metrics
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Mean Absolute Error',
                stat.replace(' Pts', ''): round(mae, 2)
            })
            analysis_results.append({
                'Category': 'Overall',
                'Metric': 'Root Mean Square Error',
                stat.replace(' Pts', ''): round(rmse, 2)
            })
            
            # Add within range metrics
            for threshold, percentage in within_metrics.items():
                analysis_results.append({
                    'Category': 'Overall',
                    'Metric': f'Within {threshold}',
                    stat.replace(' Pts', ''): f"{round(percentage, 1)}%"
                })
            
            # Add biggest misses
            biggest_misses = complete_preds.nlargest(3, f'{stat} Diff')
            for i, row in biggest_misses.iterrows():
                analysis_results.append({
                    'Category': f'{stat} Overestimates',
                    'Metric': f"#{i+1}",
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
                    'Metric': f"#{i+1}",
                    'Player': row['Player'],
                    'Date': row['Date'],
                    'Predicted': row[pred_col],
                    'Actual': row[actual_col],
                    'Difference': row[f'{stat} Diff']
                })
        
        # Player-specific analysis - updated to include Fantasy Pts
        for player in complete_preds['Player'].unique():
            player_preds = complete_preds[complete_preds['Player'] == player]
            player_stats = {}
            
            for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV', 'Fantasy Pts']:
                mae = abs(player_preds[f'{stat} Diff']).mean()
                player_stats[stat.replace(' Pts', '')] = round(mae, 2)
            
            analysis_results.append({
                'Category': 'Player Analysis',
                'Metric': 'Mean Absolute Error',
                'Player': player,
                'PTS': player_stats['PTS'],
                'REB': player_stats['REB'],
                'AST': player_stats['AST'],
                'BLK': player_stats['BLK'],
                'STL': player_stats['STL'],
                'TOV': player_stats['TOV'],
                'Fantasy': player_stats['Fantasy'],
                'Sample Size': len(player_preds)
            })
        
        # Convert to DataFrame and save
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(output_csv, index=False)
        logger.info(f"Analysis saved to {output_csv}")
        
        # Also display key metrics in logs - updated to include Fantasy Pts
        logger.info("\nKey Metrics Summary:")
        for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']:
            mae = analysis_df[analysis_df['Metric'] == 'Mean Absolute Error'][stat].iloc[0]
            within_2 = analysis_df[analysis_df['Metric'] == 'Within 2'][stat].iloc[0]
            logger.info(f"{stat} - MAE: {mae}, Within 2: {within_2}")
        
        # Log Fantasy Points separately with different thresholds
        fantasy_mae = analysis_df[analysis_df['Metric'] == 'Mean Absolute Error']['Fantasy'].iloc[0]
        fantasy_within_10 = analysis_df[analysis_df['Metric'] == 'Within 10']['Fantasy'].iloc[0]
        logger.info(f"Fantasy Points - MAE: {fantasy_mae}, Within 10: {fantasy_within_10}")
        
    except FileNotFoundError:
        logger.error(f"Predictions file not found: {input_csv}")
    except Exception as e:
        logger.error(f"Error analyzing predictions: {e}")

def list_fantasy_predictions(date=None, csv_file='predictions-tracking.csv'):
    """
    List all players with their fantasy points predictions for a specific date,
    sorted from highest to lowest predicted fantasy points.
    
    Args:
        date (str, optional): Date in format 'm/d/yy'. Defaults to today's date.
        csv_file (str, optional): Path to the predictions CSV file.
    """
    try:
        # If no date provided, use today's date
        if date is None:
            date = datetime.now().strftime("%-m/%-d/%y")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.info("No predictions found in the CSV file")
            return
        
        # Filter for the specified date
        date_predictions = df[df['Date'] == date]
        
        if date_predictions.empty:
            logger.info(f"No predictions found for {date}")
            return
        
        # Sort by predicted fantasy points (descending)
        sorted_predictions = date_predictions.sort_values(
            by='Pred Fantasy Pts', 
            ascending=False
        )
        
        # Log the sorted predictions
        logger.info(f"\nFantasy Points Predictions for {date}:")
        logger.info(f"{'Player':<20} {'Opponent':<10} {'Home/Away':<10} {'Fantasy Pts':<12}")
        logger.info("-" * 55)
        
        for _, row in sorted_predictions.iterrows():
            player = row['Player']
            opponent = row['Vs.']
            home_away = row['Loc']
            fantasy_pts = row['Pred Fantasy Pts']
            
            logger.info(f"{player:<20} {opponent:<10} {home_away:<10} {fantasy_pts:<12.2f}")
        
        logger.info(f"\nTotal players: {len(sorted_predictions)}")
        
    except FileNotFoundError:
        logger.error(f"Predictions file not found: {csv_file}")
    except Exception as e:
        logger.error(f"Error listing fantasy predictions: {e}")

def main():
    """Main function to handle different command line arguments"""
    parser = argparse.ArgumentParser(description='NBA Player Prediction Tool')
    parser.add_argument('action', choices=['predict', 'update', 'analyze', 'fantasy'],
                       help='Action to perform: predict new stats, update previous stats, analyze results, or list fantasy predictions')
    parser.add_argument('--players', default="Josh Hart,Jalen Brunson,Pascal Siakam",
                       help='Comma-separated list of players to predict (for predict action)')
    parser.add_argument('--ollama', action='store_true',
                       help='Enable Ollama-powered analysis')
    parser.add_argument('--date', 
                       help='Date for fantasy predictions in format m/d/yy (for fantasy action)')
    
    args = parser.parse_args()
    
    # Set Ollama flag if provided
    global OLLAMA_ENABLED
    if args.ollama:
        OLLAMA_ENABLED = True
        logger.info("Enabling Ollama LLM analysis")
        global llm_manager
        llm_manager = OllamaManager()
        if not llm_manager.available:
            logger.warning("Ollama not available. Make sure it's installed and running.")
    
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
        
    elif args.action == 'fantasy':
        logger.info("Listing fantasy predictions...")
        list_fantasy_predictions(date=args.date)

if __name__ == "__main__":
    main()

