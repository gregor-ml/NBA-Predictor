import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder


# Fetches NBA game data for the 2024 season from the NBA API, filters for specific teams, and processes it into a home-away format. It corrects home/away assignments, merges home and away data, and returns a cleaned DataFrame.
def scrap_games():
    gamefinder = leaguegamefinder.LeagueGameFinder()
    df = gamefinder.get_data_frames()[0]    
    
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['min'])
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d 00:00:00')
    df = df[(df['game_date']>'2024-10-19')&(df['season_id']=='22024')]

    df['is_home'] = df['matchup'].apply(lambda x: 1 if 'vs.' in x else 0)

    team_names = ['Boston Celtics', 'Golden State Warriors', 'Detroit Pistons', 'Utah Jazz', 'Brooklyn Nets', 'Indiana Pacers', 
              'Sacramento Kings', 'Phoenix Suns', 'San Antonio Spurs', 'Minnesota Timberwolves', 'Toronto Raptors', 'Memphis Grizzlies', 
              'Miami Heat', 'Atlanta Hawks', 'Philadelphia 76ers', 'Los Angeles Lakers', 'Houston Rockets', 'Portland Trail Blazers', 'Charlotte Hornets', 
              'New York Knicks', 'Washington Wizards', 'Dallas Mavericks', 'Chicago Bulls', 'Milwaukee Bucks', 'Orlando Magic', 'Denver Nuggets', 
              'Oklahoma City Thunder', 'LA Clippers', 'New Orleans Pelicans', 'Cleveland Cavaliers']
    
    df = df[df['team_name'].isin(team_names)]
    
    # Corrects home and away assignments in the NBA game DataFrame for duplicated game entries. Identifies duplicated games based on 'game_id' and 'matchup', then randomly assigns one team as home ('is_home' = 1) and the other as away ('is_home' = 0) for each duplicated pair. Returns the corrected DataFrame.
    def fix_home_away(df):
        df = df.copy()
        duplicated = df[df.duplicated(subset=['game_id', 'matchup'], keep=False)]

        if duplicated.empty:
            return df
        for (game_id, matchup), group in duplicated.groupby(['game_id', 'matchup']):
            if len(group) == 2:
                idx_home, idx_away = np.random.choice(group.index, size=2, replace=False)
                df.at[idx_home, 'is_home'] = 1
                df.at[idx_away, 'is_home'] = 0
        return df
    
    df = fix_home_away(df)
    
    home = df[df['is_home'] == 1].set_index('game_id')
    away = df[df['is_home'] == 0].set_index('game_id')

    df = home.add_suffix('_home').merge(
        away.add_suffix('_away'),
        left_index=True,
        right_index=True
    ).reset_index()
    
    df = df.rename(columns={'season_id_home':'season_id','game_date_home':'game_date'})
    df = df.drop(columns=['season_id_away','game_date_away','is_home_home','is_home_away'])
    
    return df

# Transforms the game data into separate home and away perspectives, renaming columns for consistency and adding an is_home indicator. It maps win/loss outcomes to binary values and prepares the data for further analysis.
def transform_to_home_away_info(df):
    home_columns = [col for col in df.columns if '_home' in col or col in ['team_id_home', 'team_name_home', 'game_id', 'game_date','pts_away','season_id']]
    away_columns = [col for col in df.columns if '_away' in col or col in ['team_id_away', 'team_name_away', 'game_id', 'game_date','pts_home','season_id']]

    df_home = df[home_columns].copy()
    df_away = df[away_columns].copy()

    df_home.columns = df_home.columns.str.replace('_home', '')
    df_home.rename(columns={'team_id_home': 'team_id', 'team_name_home': 'team_name', 'pts_away':'pts_lost'}, inplace=True)
    df_away.columns = df_away.columns.str.replace('_away', '')
    df_away.rename(columns={'team_id_away': 'team_id', 'team_name_away': 'team_name','pts_home':'pts_lost'}, inplace=True)

    df_home['is_home'] = 1
    df_away['is_home'] = 0

    df_combined = pd.concat([df_home, df_away], ignore_index=True)
    df_combined['wl'] = df_combined['wl'].map({'W': 1, 'L': 0})

    df_combined['game_date'] = pd.to_datetime(df_combined['game_date'])

    return df_combined

# Computes advanced basketball statistics for each game, such as possessions, offensive/defensive ratings, effective field goal percentage, true shooting percentage, and other metrics to enhance predictive modeling.
def calculate_advanced_stats(df):
    df['possessions'] = df['fga'] - df['oreb'] + df['tov'] + 0.44 * df['fta']
    df['ORTG'] = df['pts'] / df['possessions']
    df['DRTG'] = df['pts_lost'] / df['possessions']
    df['eFG%'] = (df['fgm'] + 0.5 * df['fg3m']) / df['fga']
    df['TS%'] = df['pts'] / (2 * (df['fga'] + 0.44 * df['fta']))
    df['AST%'] = df['ast'] / df['fgm']
    df['PF%'] = df['pf'] / (df['possessions'])
    df['TOV%'] = df['tov'] / df['possessions']
    df['FTR'] = df['fta'] / df['fga']
    df['PPS'] = df['pts'] / df['fga']
    df['3PAr'] = df['fg3a'] / df['fga']
    return df

# Splits the DataFrame by season and applies a provided function to each season's data, then recombines the results. Useful for season-specific calculations while maintaining data integrity.
def season_split(df, func):    
    unique_seasons = df['season_id'].unique()
    
    modified_frames = []
    
    for season in unique_seasons:
        subframe = df[df['season_id'] == season].copy()
        modified_subframe = func(subframe)
        modified_frames.append(modified_subframe)
        
    df = pd.concat(modified_frames, ignore_index=True)
    return df

# Calculates the cumulative win/loss ratio for each team by sorting games chronologically and tracking wins and games played, providing a measure of team performance over time.
def calculate_win_loss_ratio(df):
    df = df.sort_values(by=['team_id', 'game_date'])
    df['wins'] = (df.groupby('team_id')['wl'].transform(lambda group: group.cumsum()))
    df['games'] = (df.groupby('team_id')['wl'].transform(lambda group: group.expanding().count()))
    df['win_loss_ratio'] = df['wins'] / df['games'].replace(0, np.nan)
    return df

# Computes rolling averages for key statistics over short-term (5 games) and long-term (15 games) windows, dropping raw stats to focus on averaged metrics for predictive consistency.
def avg_calc(df):
    df_copy = df.copy()
    
    stats_columns = ['wl','fga','fg3m','ftm', 'fg_pct', 'fg3a', 'fg3_pct', 'dreb', 'plus_minus',
                     'fta', 'ft_pct','oreb','fgm', 'reb', 'ast', 'stl', 'blk', 
                     'tov', 'pf', 'pts','pts_lost', 'possessions', 'ORTG', 
                     'DRTG', 'eFG%', 'TS%', 'AST%', 'PF%','TOV%', 'FTR', 'PPS', 
                     '3PAr']

    rolling_windows = {
        "ShortTerm": (5, 1),
        "LongTerm": (15, 1)
    }

    for name, (window, min_periods) in rolling_windows.items():
        avg_stats = (
            df.groupby('team_id')[stats_columns]
            .apply(lambda group: group.rolling(window=window, min_periods=min_periods).mean())
            .reset_index(level=0, drop=True)
        )
        df_copy = pd.concat([df_copy, avg_stats.add_suffix(f'_avg_{name}')], axis=1)
        
    columns_to_drop = ['fga','plus_minus','fg3m','ftm', 'fg_pct', 'fg3a', 'fg3_pct', 'dreb','plus_minus',
                    'fta', 'ft_pct','oreb','fgm', 'reb', 'ast', 'stl', 'blk', 
                    'tov', 'pf', 'pts','pts_lost', 'possessions', 'ORTG', 
                    'DRTG', 'eFG%', 'TS%', 'AST%', 'PF%','TOV%', 'FTR', 'PPS', 
                    '3PAr']
    df_copy = df_copy.drop(columns=columns_to_drop)
    return df_copy

# Creates a home-team perspective DataFrame by merging home and away game data, renaming columns to reflect opponent stats, and structuring it for home-team analysis.
def home_away(df):
    home_df = df[df['is_home'] == 1].add_suffix('_home')
    away_df = df[df['is_home'] == 0].add_suffix('_away')
    home_away = pd.merge(
        home_df, away_df, 
        left_on='game_id_home', right_on='game_id_away', 
        suffixes=('_home', '_away')
    )
    
    new_columns = {}
    for col in home_away.columns:
        if col.endswith('_home'):
            new_columns[col] = col.replace('_home', '')
        elif col.endswith('_away'):
            new_columns[col] = col.replace('_away', '_opp')
            
    home_away = home_away.rename(columns=new_columns)
    home_away = home_away.rename(columns={'is':'is_home'})
    home_away = home_away.drop(columns=['is_home_opp'])
    
    
    return home_away

# Creates an away-team perspective DataFrame by merging away and home game data, renaming columns to reflect opponent stats, and structuring it for away-team analysis.
def away_home(df):
    home_df = df[df['is_home'] == 1].add_suffix('_home')
    away_df = df[df['is_home'] == 0].add_suffix('_away')
    away_home = pd.merge(
        away_df, home_df, 
        left_on='game_id_away', right_on='game_id_home', 
        suffixes=('_away', '_home'))
    
    new_columns = {}
    for col in away_home.columns:
        if col.endswith('_away'):
            new_columns[col] = col.replace('_away', '')
        elif col.endswith('_home'):
            new_columns[col] = col.replace('_home', '_opp')
            
    away_home = away_home.rename(columns=new_columns)
    away_home = away_home.rename(columns={'is_home_opp':'is_home'})
    away_home = away_home.drop(columns=['is_opp_opp'])

    return away_home

# Calculates Elo ratings for teams based on game outcomes, updating ratings after each game using expected probabilities and a k-factor to reflect performance changes.
def elo_calc(df, initial_elo=1500, k_factor=40): 
    df = df.sort_values(by='game_date').reset_index(drop=True)
    elo_ratings = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        team = row['team_name']
        team_opp = row['team_name_opp']

        home_elo = elo_ratings.get(team, initial_elo)
        away_elo = elo_ratings.get(team_opp, initial_elo)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        outcome = row['wl']
        expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_opp = 1 - expected

        elo_ratings[team] = home_elo + k_factor * (outcome - expected)
        elo_ratings[team_opp] = away_elo + k_factor * ((1 - outcome) - expected_opp)

    df['elo'] = home_elos
    df['elo_opp'] = away_elos
    df['elo_diff'] = df['elo'] - df['elo_opp']

    return df

# Applies Elo calculations to home and away DataFrames, ensuring consistent Elo ratings across perspectives by transferring opponent Elo scores and computing Elo differences.
def add_elo_ranking(df1, df2):
    df1 = elo_calc(df1)
    df1 = df1.sort_values(by=['game_date','game_id']).reset_index(drop=True)
    df2 = df2.sort_values(by=['game_date','game_id']).reset_index(drop=True)
    df2['elo'] = df1['elo_opp']
    df2['elo_opp'] = df1['elo']
    df2['elo_diff'] = -df1['elo_diff']
    return df1, df2

# Combines home and away DataFrames, removes opponent-specific columns and differences, sorts by date, and drops missing values to produce a final, clean DataFrame for analysis.
def final_calc(df1, df2):
    df1 = df1.drop(columns = ['game_id_opp', 'game_date_opp', 'wl_opp'])
    df2 = df2.drop(columns=['game_id_opp', 'game_date_opp', 'wl_opp'])
    
    df_final = pd.concat([df1,df2])
    df_final = df_final.sort_values(by=['game_date'], ascending=False).reset_index(drop=True)
    df_final = df_final.dropna()
    
    df_final = df_final.drop([col for col in df_final.columns if col.endswith('_opp') or col.endswith('_diff')], axis=1)
    return df_final

# Renames columns in the DataFrame to user-friendly names (e.g., "Field Goal Attempts (Short Term)") for better readability in the Streamlit app's display.
def col_reaname(df):
    df = df.rename(columns={
        'team_name':'Team',
        'fga_avg_ShortTerm': 'Field Goal Attempts (Short Term)',
        'fg3m_avg_ShortTerm': 'Three-Point Makes (Short Term)',
        'ftm_avg_ShortTerm': 'Free Throws Made (Short Term)',
        'fg_pct_avg_ShortTerm': 'Field Goal % (Short Term)',
        'fg3a_avg_ShortTerm': 'Three-Point Attempts (Short Term)',
        'fg3_pct_avg_ShortTerm': 'Three-Point % (Short Term)',
        'fta_avg_ShortTerm': 'Free Throw Attempts (Short Term)',
        'ft_pct_avg_ShortTerm': 'Free Throw % (Short Term)',
        'fgm_avg_ShortTerm': 'Field Goals Made (Short Term)',
        'ast_avg_ShortTerm': 'Assists (Short Term)',
        'tov_avg_ShortTerm': 'Turnovers (Short Term)',
        'pts_avg_ShortTerm': 'Points (Short Term)',
        'possessions_avg_ShortTerm': 'Possessions (Short Term)',
        'eFG%_avg_ShortTerm': 'Effective FG% (Short Term)',
        'TS%_avg_ShortTerm': 'True Shooting% (Short Term)',
        'PPS_avg_ShortTerm': 'Points Per Shot (Short Term)',
        '3PAr_avg_ShortTerm': 'Three-Point Attempt Rate (Short Term)',
        'reb_avg_ShortTerm': 'Rebounds (Short Term)',
        'oreb_avg_ShortTerm': 'Offensive Rebounds (Short Term)',
        'dreb_avg_ShortTerm': 'Defensive Rebounds (Short Term)',
        'stl_avg_ShortTerm': 'Steals (Short Term)',
        'blk_avg_ShortTerm': 'Blocks (Short Term)',
        'pf_avg_ShortTerm': 'Personal Fouls (Short Term)',
        'pts_lost_avg_ShortTerm': 'Points Allowed (Short Term)',

        # Long Term Versions
        'fga_avg_LongTerm': 'Field Goal Attempts (Long Term)',
        'fg3m_avg_LongTerm': 'Three-Point Makes (Long Term)',
        'ftm_avg_LongTerm': 'Free Throws Made (Long Term)',
        'fg_pct_avg_LongTerm': 'Field Goal % (Long Term)',
        'fg3a_avg_LongTerm': 'Three-Point Attempts (Long Term)',
        'fg3_pct_avg_LongTerm': 'Three-Point % (Long Term)',
        'fta_avg_LongTerm': 'Free Throw Attempts (Long Term)',
        'ft_pct_avg_LongTerm': 'Free Throw % (Long Term)',
        'fgm_avg_LongTerm': 'Field Goals Made (Long Term)',
        'ast_avg_LongTerm': 'Assists (Long Term)',
        'tov_avg_LongTerm': 'Turnovers (Long Term)',
        'pts_avg_LongTerm': 'Points (Long Term)',
        'possessions_avg_LongTerm': 'Possessions (Long Term)',
        'eFG%_avg_LongTerm': 'Effective FG% (Long Term)',
        'TS%_avg_LongTerm': 'True Shooting% (Long Term)',
        'PPS_avg_LongTerm': 'Points Per Shot (Long Term)',
        '3PAr_avg_LongTerm': 'Three-Point Attempt Rate (Long Term)',
        'reb_avg_LongTerm': 'Rebounds (Long Term)',
        'oreb_avg_LongTerm': 'Offensive Rebounds (Long Term)',
        'dreb_avg_LongTerm': 'Defensive Rebounds (Long Term)',
        'stl_avg_LongTerm': 'Steals (Long Term)',
        'blk_avg_LongTerm': 'Blocks (Long Term)',
        'pf_avg_LongTerm': 'Personal Fouls (Long Term)',
        'pts_lost_avg_LongTerm': 'Points Allowed (Long Term)',

        # Other Stats
        'wl_avg_ShortTerm': 'Win/Loss Ratio (Last 5 Games)',
        'win_loss_ratio': 'Win/Loss Ratio (Whole Season)',
        'elo' : 'Elo Ranking'
    })
    return df

# Prepares a single game prediction by selecting data for the specified home and away teams, calculating stat differences between them, and formatting the data for model input.
def create_game(df, home, away):
    home = df[df['team_name'] == home]
    away = df[df['team_name'] == away]
    
    # Calculates differences between team and opponent statistics for specified prefixes and time horizons, adding new columns for each difference.
    def calculate_stat_differences(df):
        df = df.sort_values(by=['game_date'])
        stats_prefixes = [
            'wl_avg', 'plus_minus_avg', 'fg_pct_avg',
            'fg3m_avg', 'fg3a_avg', 'fg3_pct_avg', 'fta_avg', 'ft_pct_avg','reb_avg', 'ast_avg',
            'tov_avg', 'pf_avg', 'pts_avg','PTS_sum', 'pts_lost_avg', 'possessions', 'ORTG', 'DRTG', 'eFG%', 'TS%', 'AST%', 'PF%',
        'TOV%', 'FTR', 'PPS', '3PAr'
        ]
        time_horizons = ['ShortTerm', 'LongTerm']
        
        for prefix in stats_prefixes:
            for horizon in time_horizons:
                home_col = f"{prefix}_{horizon}"
                opp_col = f"{prefix}_{horizon}_opp"

                if home_col in df.columns and opp_col in df.columns:
                    diff_col_name = f"{prefix}_{horizon}_diff"
                    df[diff_col_name] = df[home_col] - df[opp_col]

        for horizon in time_horizons:
            pts_col = f"pts_avg_{horizon}"
            pts_lost_col = f"pts_lost_avg_{horizon}"
            pts_diff_col = f"pts_diff_{horizon}"
            
            if pts_col in df.columns and pts_lost_col in df.columns:
                df[pts_diff_col] = df[pts_col] - df[pts_lost_col]
        
        return df
    

    
    if len(home) != 1 or len(away) != 1:
        raise ValueError("Each team must have exactly one row in the DataFrame.")
    home['game_date'] = pd.to_datetime(home['game_date'])
    away['game_date'] = pd.to_datetime(away['game_date'])
    away = away.add_suffix('_opp')

    game = pd.concat([home.reset_index(drop=True), away.reset_index(drop=True)], axis=1)
    game['win_loss_diff'] = game['win_loss_ratio'] - game['win_loss_ratio_opp']
    game['elo_diff'] = game['elo'] - game['elo_opp']
    game = calculate_stat_differences(game)
    game = game.drop(columns=['team_abbreviation_opp', 'is_home', 'team_id', 'game_id', 'game_id_opp', 'season_id_opp', 'is_home_opp', 'team_id_opp', 
                              'season_id', 'wl', 'team_name', 'game_date_opp', 'win_loss_diff', 'matchup_opp', 'wl_opp', 'team_abbreviation', 'team_name_opp',
                              'game_date', 'matchup'])
    return game