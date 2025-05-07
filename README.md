# NBA Game Prediction App

This project is a Streamlit application designed to predict the outcomes of NBA games using a pre-trained neural network model. The application provides visualizations and statistical insights into team performance, allowing users to make informed predictions.

## Project Structure

```
nba-prediction-app
├── src
│   ├── app.py                # Main entry point of the Streamlit application
│   ├── calculations.py       # Functions for data processing and calculations
│   ├── plots.py              # Functions for generating visualizations
│   ├── data
│   │   ├── features.json     # Feature names used for model input
│   │   ├── scaler.pkl        # Scaler for normalizing input data
│   │   └── pca.pkl           # PCA model for dimensionality reduction
│   └── models
│       └── NN.keras          # Pre-trained neural network model
├── notebook
│   ├── data_preprocessing_and_training_models.ipynb # Data preprocessing and model training
├── requirements.txt          # Project dependencies  
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd nba-prediction-app
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Streamlit application by running:
   ```
   streamlit run src/app.py
   ```

## Usage Guidelines

- Select the home and away teams from the dropdown menus to predict the game outcome.
- Click the "Predict" button to see the predicted win probabilities for both teams.
- Explore the various statistics and visualizations available in the application to gain insights into team performance.

## Features

- Interactive interface for selecting teams and viewing predictions.
- Visualizations of team statistics and game outcomes.
- Data processing and calculations for advanced NBA statistics.
- Pre-trained neural network model for accurate predictions.

## Data
The model was trained on the [NBA Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball) from Kaggle. To train the model on your own, download the game.csv table from the Kaggle dataset and place it in the notebook/ directory.

## Acknowledgments

This project leverages several libraries and tools, including Streamlit for the web interface and TensorFlow for machine learning. Please note that the predictions and calculations provided by this application are for informational and entertainment purposes only. They should not be used as a basis for betting or gambling decisions, as they do not guarantee outcomes and carry inherent uncertainties.
