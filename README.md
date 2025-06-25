# House-Price-Predictor

- [HuggingFace Space](https://huggingface.co/spaces/JulianG2002/House-Price-Estimator)
- [Kaggle DataSet](https://www.kaggle.com/datasets/kanchana1990/real-estate-data-chicago-2024?)

This  dashboard predicts house prices in Chicago based on property features using machine learning. Built with Python and Streamlit, it allows users to explore real estate trends, view model insights, and estimate property values in real-time.

---

## Features

- **Predictive Modeling**: Trained on real housing data using scikit-learn and regression algorithms.
- **Chicago-Specific Data**: Focused on local data for high-accuracy predictions within Chicago neighborhoods.
- **Feature Importance Visualization**: Understand which variables most influence house prices.
- **Data Viewer**: Browse the dataset to validate trends or explore patterns.
- **Interactive UI**: Built with Streamlit for a clean, fast, and responsive interface.

---

## How It Works

1. **Preprocessed Housing Data** is cleaned and encoded from real Chicago listings.
2. A **machine learning regression model** is trained to learn price patterns.
3. Users input property details (e.g., bedrooms, bathrooms, square footage).
4. The model returns a **predicted price** instantly, along with a breakdown of features.

---

## Resources Used

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib)
- **Streamlit** (interactive dashboard)
- **Hugging Face Spaces** (deployment)
- **CSV data pipeline** loading and transforming housing data
