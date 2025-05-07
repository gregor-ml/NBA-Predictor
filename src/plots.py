import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def custom_bar_plot(df, y_col, x='team_abbreviation', title='Bar Plot'):
    if y_col == 'Wins Progression Chart':
        # Filter data for the selected team and sort by game date
        selected_team = x
        sorted_df = df[df['Team'] == selected_team].sort_values(by=['game_date'], ascending=False)

        # Create a figure and axis for the line plot
        fig, ax = plt.subplots(figsize=(15, 11))

        # Plot a line chart of wins over games
        sns.lineplot(data=sorted_df, x='games', y='wins', ax=ax)

        # Get current axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Calculate the maximum range to ensure square plot
        max_range = max(x_max - x_min, y_max - y_min)
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2

        # Set adjusted axis limits for better visualization
        ax.set_xlim(0, center_x + max_range / 2)
        ax.set_ylim(-0.5, center_y + max_range / 2)
        
        # Set axis labels and title
        ax.set_xlabel("Games", fontsize=12)
        ax.set_ylabel("Wins", fontsize=12)
        ax.set_title(f"Wins Progression Chart - {x}", fontsize=14, fontweight="bold")
        
        # Add grid for better readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Set integer ticks for x and y axes
        ax.set_xticks(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 1))
        ax.set_yticks(range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1]) + 1, 1))
        # Rotate x-axis labels for clarity
        plt.xticks(rotation=45)
        # Display the plot in Streamlit
        st.pyplot(plt)
    
    # Handle standard bar plot for other statistics
    else: 
        # Remove duplicate teams, keeping the first entry
        df = df.drop_duplicates(subset=[x], keep='first')
        # Sort data by the y-column in descending order
        sorted_df = df.sort_values(by=y_col, ascending=False)

        # Create a figure and axis for the bar plot
        fig, ax = plt.subplots(figsize=(15, 11))
        # Generate a color palette for the bars
        colors = sns.color_palette("viridis", len(sorted_df))
        # Plot the bar chart
        ax.bar(sorted_df[x], sorted_df[y_col], color=colors)

        # Set axis labels and title
        ax.set_xlabel("Team", fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f"NBA Teams - {y_col}", fontsize=14, fontweight="bold")

        # Adjust y-axis limits for better visualization
        y_min, y_max = sorted_df[y_col].min(), sorted_df[y_col].max()
        ax.set_ylim(y_min * 0.93, y_max * 1.02)
        # Rotate x-axis labels and set font size
        plt.xticks(rotation=60, ha="right", fontsize=10)
        # Add horizontal grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Display the plot in Streamlit
        st.pyplot(fig)