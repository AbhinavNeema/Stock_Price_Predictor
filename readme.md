Project Title: A Multi-Sector Stock Price Predictive Model
Overview
This project is a powerful predictive model designed to analyze and forecast key financial metrics across major economic sectors. By leveraging advanced machine learning models trained on historical data, this system provides actionable insights for each designated sector. The core innovation of this project lies in its automated daily prediction engine and the integration of live market sentiment to provide a comprehensive, data-driven tool for strategic market analysis and decision-making.

Key Features
Automated Daily Predictions: The application uses a background scheduler (apscheduler) to automatically fetch data and generate next-day price predictions for all tracked stocks every day at 4:00 PM IST.

Performance-Optimized Caching: Daily predictions are cached in memory to ensure fast, low-latency responses for the main API endpoints.

Live Sentiment Analysis: The model incorporates a live sentiment score for each stock, derived from recent news articles using the VADER sentiment analysis tool, providing a crucial qualitative factor for predictions.

Top Stock Ranking: A specialized API endpoint calculates and returns the top 5 stocks based on a custom score metric, which is the ratio of predicted return to volatility, helping to identify high-potential, risk-adjusted opportunities.

On-Demand Predictions: A dedicated endpoint allows users to request real-time, single-stock predictions with historical price data for immediate analysis.

Modular Architecture: The system is built with a clear separation of concerns, with dedicated directories for models and scalers, and separate scripts for each task, making it scalable and easy to maintain.

Methodology and Implementation
Data and Preprocessing

The model uses historical data fetched from yfinance. This raw data is then processed within the application, where features like technical indicators (SMA and RSI) are generated using pandas_ta. The sentiment data is sourced from a news API, and the processing logic for this is contained within sentimentFinal.ipynb and utilized by the main application.

Model Architecture

Our system employs a multi-model architecture, where a specialized keras model is trained for each sector. The trained models are stored as serialized .keras files in the models/ directory, while their corresponding data scalers (joblib) are stored separately in the scalers/ directory.

The main application logic in app.py dynamically loads these pre-trained models and scalers. The models.py and prediction.py modules contain the core logic for loading the models and executing predictions. This modular approach allows for fine-tuned predictions and ensures the system is easily scalable to include new sectors.

A separate backtest.py script is used to evaluate the model's performance on historical data, and scheduler.py handles the logic for the automated daily prediction job.

Project Structure

.
├── .vscode/            # VS Code configuration files
├── models/             # Directory for trained Keras model files
├── scalers/            # Directory for saved data scaler files
├── venv/               # Project's Python virtual environment
├── .env                # Environment variables (e.g., API keys)
├── .gitignore          # Files to be ignored by Git
├── app.py              # Main application with API endpoints
├── backtest.py         # Script for backtesting the model
├── index.html          # Web interface for the application
├── models.py           # Model-related utility functions
├── prediction.py       # Core prediction logic
├── readme.md           # Project documentation
├── requirements.txt    # Project dependencies
├── scheduler.py        # Script for the daily prediction scheduler
├── sentimentFinal.ipynb# Jupyter Notebook for sentiment analysis exploration
└── utils.py            # Utility functions

How to Use
1. Set Up Your Environment

First, clone this repository to your local machine.

git clone [repository_url]
cd [project_folder_name]

2. Install Dependencies

Install all the necessary Python packages using the requirements.txt file. It is highly recommended to use a virtual environment.

pip install -r requirements.txt

3. Run the Application

Execute the main application script from the command line.

python app.py


Future Enhancements
We are continuously working to improve the model. Planned enhancements include:

Integration with Live Data: Incorporating real-time data APIs to provide up-to-the-minute predictions.

Web Interface: Developing a user-friendly front-end application to visualize the predictions and top stocks.

Model Optimization: Exploring more advanced architectures to further enhance performance.

Contact
For any questions or suggestions, please contact [Your Name] at [Your Email Address].

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

