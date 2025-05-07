# Import required libraries for the Streamlit app
import streamlit as st
import tensorflow as tf
import joblib
import json
import importlib
from pathlib import Path

# Import custom modules for calculations and plotting
import calculations
import plots

# Main function to run the Streamlit app
def main():
    # Set Streamlit page configuration to wide layout
    st.set_page_config(layout="wide")

    base_path = Path(__file__).resolve().parent

    # Cache the TensorFlow model to avoid reloading
    @st.cache_resource
    def load_model():
        model_path = base_path / "models" / "NN.keras"
        return tf.keras.models.load_model(model_path)

    # Initialize the model
    model = load_model()

    # Cache the scaler to avoid reloading
    @st.cache_resource
    def load_scaler():
        scaler_path = base_path / "data" / "scaler.pkl"
        # Load the pre-trained scaler for data normalization
        return joblib.load(scaler_path)
    
    # Initialize the scaler
    scaler = load_scaler()
    
    # Cache the PCA model to avoid reloading
    @st.cache_resource
    def load_pca():
        pca_path = base_path / "data" / "pca.pkl"
        return joblib.load(pca_path)

    # Initialize the PCA model
    pca = load_pca()

    # Cache feature names for model input
    @st.cache_resource
    def load_features():
        f_path = base_path / "data" / "features.json"
        # Load feature names from JSON file
        with open(f_path, 'r') as f:
            return json.load(f)

    # Initialize feature names
    feature_names = load_features()

    # Cache preprocessed game data to avoid redundant computation
    @st.cache_data
    def fetch_preprocessed_data():
        # Scrape game data
        df = calculations.scrap_games()
        # Transform data to home/away format
        df = calculations.transform_to_home_away_info(df)
        # Calculate advanced statistics
        df = calculations.calculate_advanced_stats(df)
        # Calculate win/loss ratios
        df = calculations.calculate_win_loss_ratio(df)
        # Calculate averages
        df = calculations.avg_calc(df)

        # Split into home and away dataframes
        h_df = calculations.home_away(df)
        a_df = calculations.away_home(df)

        # Add Elo rankings
        h_df, a_df = calculations.add_elo_ranking(h_df, a_df)
        # Perform final calculations
        return calculations.final_calc(h_df, a_df)

    # Initialize preprocessed data
    df = fetch_preprocessed_data()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    # Cache team list for dropdowns
    @st.cache_data
    def get_team_list(df):
        # Get unique team names
        return df['team_name'].unique().tolist()

    # Initialize team names
    team_names = get_team_list(df)

    # Cache prediction results
    @st.cache_data
    def predict_winner(team_home, team_away):
        # Get unique team data
        df_pred = df.drop_duplicates(subset=['team_name'], keep='first')
        
        # Create game data for prediction
        game = calculations.create_game(df_pred, team_home, team_away)
        # Scale the game data
        game = scaler.transform(game[feature_names])
        
        # Apply PCA transformation
        game = pca.transform(game)
        
        # Make prediction
        prediction_prob = model.predict(game)[0][0]
        # Calculate win probabilities
        prediction_home = round(prediction_prob * 100, 2)
        prediction_away = round(100 - prediction_home, 2)
        
        # Set colors based on prediction
        home_color = "green" if prediction_home > prediction_away else "red"
        away_color = "green" if prediction_away > prediction_home else "red"

        return prediction_home, prediction_away, home_color, away_color

    # Initialize session state for predictions
    if 'prediction_home' not in st.session_state:
        st.session_state.prediction_home = None
        st.session_state.prediction_away = None
        st.session_state.home_color = None
        st.session_state.away_color = None
        st.session_state.team_home = None
        st.session_state.team_away = None

    # Left column: Prediction interface
    with col1:
        # Display title
        st.title('🏀 NBA Match Prediction')
        # Create two columns for team selection
        col11, col12 = st.columns(2)
        with col11:
            # Home team selection dropdown
            team_home = st.selectbox("🏠 Home Team", team_names, key="team1_selectbox")
        with col12:
            # Away team selection dropdown (excludes home team)
            team_away = st.selectbox("✈️ Away Team", [t for t in team_names if t != team_home], key="team2_selectbox")

        # Predict button
        if st.button("📈 Predict", key='predict_button'):
            # Get prediction results
            prediction_home, prediction_away, home_color, away_color = predict_winner(team_home, team_away)
            
            # Store results in session state
            st.session_state.prediction_home = prediction_home
            st.session_state.prediction_away = prediction_away
            st.session_state.home_color = home_color
            st.session_state.away_color = away_color
            st.session_state.team_home = team_home
            st.session_state.team_away = team_away
            
        # Display prediction results if available
        if st.session_state.prediction_home is not None:
            st.markdown(f"""
            <div style="font-family: 'Arial', sans-serif; font-size: 24px; text-align: center;">
                <span style="font-weight: bold; color: {st.session_state.home_color};">{st.session_state.team_home}</span> 
                <span style="font-size: 24px; font-weight: bold; color: {st.session_state.home_color};">{st.session_state.prediction_home:.2f}%</span> 
                vs 
                <span style="font-size: 24px; font-weight: bold; color: {st.session_state.away_color};">{st.session_state.prediction_away:.2f}%</span>
                <span style="font-weight: bold; color: {st.session_state.away_color};"> {st.session_state.team_away}</span>
            </div>
            """, unsafe_allow_html=True)
            
    # Rename columns for display
    df = calculations.col_reaname(df)

    # Define column categories for statistics
    column_categories = {
        "Offensive Short-Term Stats": [
            'Field Goal Attempts (Short Term)', 'Three-Point Makes (Short Term)', 
            'Free Throws Made (Short Term)', 'Field Goal % (Short Term)', 
            'Three-Point Attempts (Short Term)', 'Three-Point % (Short Term)', 
            'Free Throw Attempts (Short Term)', 'Free Throw % (Short Term)', 
            'Field Goals Made (Short Term)', 'Assists (Short Term)', 
            'Turnovers (Short Term)', 'Points (Short Term)', 
            'Possessions (Short Term)', 'Effective FG% (Short Term)', 
            'True Shooting% (Short Term)', 'Points Per Shot (Short Term)', 
            'Three-Point Attempt Rate (Short Term)'
        ],
        "Defensive Short-Term Stats": [
            'Rebounds (Short Term)', 'Offensive Rebounds (Short Term)', 
            'Defensive Rebounds (Short Term)', 'Steals (Short Term)', 
            'Blocks (Short Term)', 'Personal Fouls (Short Term)', 
            'Points Allowed (Short Term)'
        ],
        "Offensive Long-Term Stats": [
            'Field Goal Attempts (Long Term)', 'Three-Point Makes (Long Term)', 
            'Free Throws Made (Long Term)', 'Field Goal % (Long Term)', 
            'Three-Point Attempts (Long Term)', 'Three-Point % (Long Term)', 
            'Free Throw Attempts (Long Term)', 'Free Throw % (Long Term)', 
            'Field Goals Made (Long Term)', 'Assists (Long Term)', 
            'Turnovers (Long Term)', 'Points (Long Term)', 
            'Possessions (Long Term)', 'Effective FG% (Long Term)', 
            'True Shooting% (Long Term)', 'Points Per Shot (Long Term)', 
            'Three-Point Attempt Rate (Long Term)'
        ],
        "Defensive Long-Term Stats": [
            'Rebounds (Long Term)', 'Offensive Rebounds (Long Term)', 
            'Defensive Rebounds (Long Term)', 'Steals (Long Term)', 
            'Blocks (Long Term)', 'Personal Fouls (Long Term)', 
            'Points Allowed (Long Term)'
        ],
        "Other Stats": [
            'Win/Loss Ratio (Last 5 Games)', 'Win/Loss Ratio (Whole Season)', 
            'Elo Ranking'
        ]
    }

    # Define chart categories
    wins_chart = {
        'Wins Progression Chart': ['Wins Progression Chart']
    }
    columns_for_chart = column_categories.copy()
    columns_for_chart.update(wins_chart)

    # Cache plot generation
    @st.cache_data
    def generate_plot(df, y_col, x):
        # Generate custom bar plot
        return plots.custom_bar_plot(df, y_col=y_col, x=x)

    # Left column: Statistics visualization
    with col1:
        # Display title
        st.title('📊 NBA Team Statistics')
        # Create two columns for category and column selection
        col21, col22 = st.columns(2)
        
        with col21:
            # Category selection dropdown
            selected_category = st.selectbox("📌 Select a statistics category:", list(columns_for_chart.keys()), index=0, key='1')       
        with col22:
            # Conditional selection based on category
            if selected_category == 'Wins Progression Chart':
                # Team selection for wins chart
                x = st.selectbox("📌 Select a team to display:", team_names, key='2')
            else:
                # Column selection for other categories
                default_columns = columns_for_chart[selected_category]
                selected_category = st.selectbox("📌 Select a column to display:", default_columns, key='2')
                x = 'team_abbreviation'
                
        # Generate and display plot
        generate_plot(df, y_col = selected_category, x = x)

    # Right column: Data table
    with col2:
        # Display title
        st.title("📋 NBA Data Overview")

        # Category selection dropdown
        selected_category = st.selectbox("📌 Select a statistics category:", list(column_categories.keys()), index=0)
        # Column selection multiselect
        default_columns = column_categories[selected_category]
        columns_to_show = st.multiselect("📌 Select columns to display:", default_columns, default=default_columns[:2])

        # Display data table if columns are selected
        if columns_to_show:
            # Create two columns for sorting options
            col21, col22 = st.columns(2)
            with col21:
                # Sort column selection
                sort_column = st.selectbox("🔽 Sort by:", columns_to_show, index=0)
            with col22:
                # Sort order selection
                sort_order = st.radio("🔄 Sort order:", ["Descending", "Ascending"], index=0)

            # Prepare and sort data
            df = df.drop_duplicates(subset=['Team'], keep='first')
            sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))

            # Display data table
            st.data_editor(
                sorted_df[['Team'] + columns_to_show], 
                use_container_width=True, 
                height=609,
                hide_index=True,
                column_config={"Team": st.column_config.TextColumn(width="medium", disabled=True)}
            )
        else:
            st.warning("⚠️ Please select at least one column to display.")

if __name__ == "__main__":
    main()
