# General Imports
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import os
import joblib
import plotly.express as px

# Load in our data!
data_path = os.path.join("data", "clean_real_estate_data.csv")
model_path = os.path.join("models", "house_price_model_v2.pkl")
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# Set up our nice, beautiful, flawless page
st.set_page_config(page_title = "House Price Predictor",  layout = "centered")
st.title("Chicago Housing Price Estimate :house:")
st.markdown("Enter the propty details below to estimate it's listing price.")

# Getting our user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("Square Footage", min_value = 200, max_value = 10_000, value = 1500)
        lot_sqft = st.number_input("Lot Size (in sq ft)", min_value = 500, max_value = 20_000, value = 3000)
        beds = st.slider("Bedrooms", 0, 10, 3)
        baths = st.slider("Bathrooms", 0, 10, 2)
        baths_full = st.slider("Full Baths", 0, 10, 2)
        baths_half = st.slider("Half Baths", 0, 10, 0)

    with col2:
        garage = st.slider("Garage Spaces", 0, 5, 1)
        stories = st.slider("Stories", 1, 3, 2)
        year_built = st.number_input("Year Built", min_value = 1850, max_value = 2024, value = 2000)
        home_type = st.selectbox("Home Type", ["single_family", "multi_family", "condos", "townhomes", "land", "mobile"])
        status = st.selectbox("Status", ["for_sale", "ready_to_build"])

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    current_year = pd.Timestamp.now().year
    house_age = current_year - year_built
    price_per_sqft = np.nan
    lot_ratio = lot_sqft / sqft if sqft != 0 else 0 

    # Features must be ordered in the same way that were fit into the model
    feature_order = [
        "year_built", "beds", "baths", "bath_full", "bath_half", "garage", "lot_sqft", "sqft", "stories",
        "lastSoldPrice", "house_age", "price_per_sqft", "lot_ratio", "type_condos", "type_land", "type_mobile",
        "type_multi_family", "type_single_family", "type_townhomes", "status_ready_to_build"
    ]

    # Lets build the user input data frame
    input_data = {
        "year_built": year_built,
        "beds": beds,
        "baths": baths,
        "baths_full": baths_full,
        "baths_half": baths_half,
        "garage": garage,
        "lot_sqft": lot_sqft,
        "sqft": sqft,
        "stories": stories,
        "lastSoldPrice": 0,
        "house_age": house_age,
        "price_per_sqft": 0,
        "lot_ratio": lot_ratio,
        "type_condos": 1 if home_type == "condos" else 0,
        "type_land": 1 if home_type == "land" else 0,
        "type_mobile": 1 if home_type == "mobile" else 0,
        "type_multi_family": 1 if home_type == "multi_family" else 0,
        "type_single_family": 1 if home_type == "single_family" else 0,
        "type_townhomes": 1 if home_type == "townhomes" else 0,
        "status_ready_to_build": 1 if status == "ready_to_build" else 0
    }

    input_data = pd.DataFrame([input_data])
    input_data = input_data[model.feature_names_in_]
    
    # Now we can predict and show results based on user given information
    prediction = model.predict(input_data)[0]
    st.success(f"Esimated Listing Price: ${prediction:,.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualization", "Model Information", "Model Insights", "Raw Data", "About"])

# Quick visualization comparing price vs square footage, which is our most importance feature.
with tab1:
    st.subheader("Visualization")
    fig = px.scatter(
        df,
        x = 'sqft',
        y = 'listPrice',
        color = 'type',
        title = 'Price vs Square footage by Home Type',
        labels = {'sqft': 'Square Footage', 'listPrice' : 'Listing Price' }
    )

    st.plotly_chart(fig)

# Give some information to the user about the model
with tab2:
    st.title("Model Information")

    st.markdown("### Model type: Random Forest Regressor")
    st.markdown("---")
    st.markdown("- R^2 Score 0.87")
    st.markdown("- MAE : $18,000")

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    st.download_button(
        label = "Download Trained Model (.pkl)",
        data = model_bytes, 
        file_name = "houe_price_model.pkl",
        mime = "application/octet-stream"
    )

# Our raw data tab, be able to show users raw data as well as give them the ability to download it.
with tab4:
    st.subheader("Raw Dataset Preview")
    st.markdown("Below is a preview of the cleaned real estate data used to train the model.")

    max_rows = len(df)
    num_rows = st.slider("Number of rows to preview", min_value = 10, max_value = min(2000, max_rows), value = 50, step = 10)

    st.dataframe(df.head(num_rows), use_container_width = True)

    csv = df.to_csv(index = False).encode("utf-8")
    st.download_button(
        label = "Download CSV",
        data = csv,
        file_name = "real_estate.csv",
        mime = "text/csv"
    )

# Lets show some feature importances
with tab3:
    importances = model.feature_importances_
    features = model.feature_names_in_

    # Sort the features by importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance" : importances,
    }).sort_values(by = "Importance", ascending = False)

    # Plot using plotly
    fig = px.bar(
        importance_df,
        x = "Importance",
        y = "Feature",
        orientation = "h",
        title = "Feature Importance (Sorted)",
        labels = {"Importance" : "Importance", "Feature" : "Feature"}
    )

    fig.update_layout(yaxis = dict (autorange = "reversed"))

    st.plotly_chart(fig)

    # ABOUT ME!!!!!!
    with tab5:
        st.subheader("About the Developer")
    
        st.markdown("""
        ### Julian Gutierrez  
        Aspiring Data Scientist | Machine Learning Enthusiast | Based in Chicago  
                    
        Graduated from the University of Illinois Urbana-Champaign with a B.S. in Information Sciences (May 2025).
    
        I built this project to demonstrate real-world applications of machine learning in the housing market. It combines data wrangling, feature engineering, predictive modeling, and interactive dashboard design.
    
        - Tools used: Python, Pandas, Streamlit, Plotly, scikit-learn  
        - Skills: Model deployment, data visualization, user interface design  
        - [GitHub](https://github.com/JulianGut2) | [LinkedIn](https://linkedin.com/in/juliangutierrez02)  
    
        """)
