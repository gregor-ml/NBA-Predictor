import streamlit as st
import tensorflow as tf
import joblib
import json
import importlib

import calculations
import plots


#importlib.reload(calculations)

def main():
    st.set_page_config(layout="wide")

    # Load and cache the model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('NN.keras')

    model = load_model()

    # Load and cache the scaler
    @st.cache_resource
    def load_scaler():
        return joblib.load('scaler.pkl')

    scaler = load_scaler()

    # Load and cache feature names
    @st.cache_resource
    def load_features():
        with open('features.json', 'r') as f:
            return json.load(f)

    feature_names = load_features()

    # Cache preprocessed game data
    @st.cache_data
    def fetch_preprocessed_data():
        df = calculations.scrap_games()
        df = calculations.transform_to_home_away_info(df)
        df = calculations.calculate_advanced_stats(df)
        df = calculations.calculate_win_loss_ratio(df)
        df = calculations.avg_calc(df)

        h_df = calculations.home_away(df)
        a_df = calculations.away_home(df)

        h_df, a_df = calculations.add_elo_ranking(h_df, a_df)
        return calculations.final_calc(h_df, a_df)

    df = fetch_preprocessed_data()

    col1, col2 = st.columns([1, 1])

    @st.cache_data
    def get_team_list(df):
        return df['team_name'].unique().tolist()

    team_names = get_team_list(df)

    @st.cache_data
    def predict_winner(team_home, team_away):
        df_pred = df.drop_duplicates(subset=['team_name'], keep='first')
        
        game = calculations.create_game(df_pred, team_home, team_away)
        game = scaler.transform(game[feature_names])

        prediction_prob = model.predict(game)[0][0]
        prediction_home = round(prediction_prob * 100, 2)
        prediction_away = round(100 - prediction_home, 2)
        
        home_color = "green" if prediction_home > prediction_away else "red"
        away_color = "green" if prediction_away > prediction_home else "red"

        return prediction_home, prediction_away, home_color, away_color

    if 'prediction_home' not in st.session_state:
        st.session_state.prediction_home = None
        st.session_state.prediction_away = None
        st.session_state.home_color = None
        st.session_state.away_color = None
        st.session_state.team_home = None
        st.session_state.team_away = None

    with col1:
        st.title('🏀 NBA Match Prediction')
        col11, col12 = st.columns(2)
        with col11:
            team_home = st.selectbox("🏠 Home Team", team_names, key="team1_selectbox")
        with col12:
            team_away = st.selectbox("✈️ Away Team", [t for t in team_names if t != team_home], key="team2_selectbox")

        if st.button("📈 Predict", key='predict_button'):
            prediction_home, prediction_away, home_color, away_color = predict_winner(team_home, team_away)
            
            st.session_state.prediction_home = prediction_home
            st.session_state.prediction_away = prediction_away
            st.session_state.home_color = home_color
            st.session_state.away_color = away_color
            st.session_state.team_home = team_home
            st.session_state.team_away = team_away
            
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
            
    df = calculations.col_reaname(df)

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

    wins_chart = {
        'Wins Progression Chart': ['Wins Progression Chart']
    }
    columns_for_chart = column_categories.copy()
    columns_for_chart.update(wins_chart)

    @st.cache_data
    def generate_plot(df, y_col, x):
        return plots.custom_bar_plot(df, y_col=y_col, x=x)

    with col1:
        st.title('📊 NBA Team Statistics')
        col21, col22 = st.columns(2)
        
        with col21:
            selected_category = st.selectbox("📌 Select a statistics category:", list(columns_for_chart.keys()), index=0, key='1')       
        with col22:
            if selected_category == 'Wins Progression Chart':
                x = st.selectbox("📌 Select a team to display:", team_names, key='2')
            else:
                default_columns = columns_for_chart[selected_category]
                selected_category = st.selectbox("📌 Select a column to display:", default_columns, key='2')
                x = 'team_abbreviation'
                
        generate_plot(df, y_col = selected_category, x = x)


    with col2:
        st.title("📋 NBA Data Overview")

        selected_category = st.selectbox("📌 Select a statistics category:", list(column_categories.keys()), index=0)
        default_columns = column_categories[selected_category]
        columns_to_show = st.multiselect("📌 Select columns to display:", default_columns, default=default_columns[:2])

        if columns_to_show:
            col21, col22 = st.columns(2)
            with col21:
                sort_column = st.selectbox("🔽 Sort by:", columns_to_show, index=0)
            with col22:
                sort_order = st.radio("🔄 Sort order:", ["Descending", "Ascending"], index=0)

            df = df.drop_duplicates(subset=['Team'], keep='first')
            sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))

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

