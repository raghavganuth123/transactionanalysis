import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Configuration ---
MODEL_FILE = "fraud_rf_model.pkl"
SCALER_FILE = "scaler.pkl"

# --- Load Model and Scaler (Cached) ---
@st.cache_resource # Cache resource loading for efficiency
def load_model_and_scaler():
    """Loads the pre-trained model and scaler."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model file ('{MODEL_FILE}') or scaler file ('{SCALER_FILE}') not found.")
        st.error("Please ensure the files are in the same directory as the script or provide the correct path.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model or scaler: {e}")
        return None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="Transaction Fraud Detector", layout="centered")
st.title("💳 Transaction Fraud Detector")
st.markdown("Enter the transaction details below to predict if it's fraudulent.")

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Only proceed if model and scaler loaded successfully
if model is not None and scaler is not None:

    st.markdown("---")
    st.subheader("Enter Transaction Details:")

    # User Inputs - Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f", key="amount")
        duration = st.number_input("Transaction Duration (seconds)", min_value=0.0, format="%.1f", key="duration", help="Time taken for the transaction process.")
        attempts = st.number_input("Login Attempts (before tx)", min_value=0, step=1, key="attempts", help="Number of login attempts before this transaction.")
        time_gap = st.number_input("Time Since Last Transaction (seconds)", min_value=0, step=1, key="time_gap")

    with col2:
        balance = st.number_input("Account Balance ($ before tx)", min_value=0.0, format="%.2f", key="balance")
        # Use current time for Hour and DayOfWeek defaults, but allow user override
        now = datetime.now()
        hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, step=1, value=now.hour, key="hour")
        day_options = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        # Get day name corresponding to current day number
        current_day_name = list(day_options.keys())[now.weekday()]
        day_name = st.selectbox("Day of Week", options=day_options.keys(), index=now.weekday(), key="day")
        day_of_week = day_options[day_name] # Get numeric value

    st.markdown("---")

    # Button to trigger prediction
    if st.button("Analyze Transaction", type="primary"):
        # Create dictionary with user input - **Ensure keys match expected feature names**
        # **Important:** The order of features in the DataFrame must exactly match the order
        # the scaler and model were trained on. Assuming the order is as below:
        feature_order = [
            "TransactionAmount", "TransactionDuration", "LoginAttempts",
            "AccountBalance", "TimeGap", "Hour", "DayOfWeek"
        ]

        new_transaction = {
            "TransactionAmount": amount,
            "TransactionDuration": duration,
            "LoginAttempts": attempts,
            "AccountBalance": balance,
            "TimeGap": time_gap,
            "Hour": hour,
            "DayOfWeek": day_of_week
        }

        try:
            # Create DataFrame with the correct column order
            df = pd.DataFrame([new_transaction], columns=feature_order)

            # Scale the data
            X_scaled = scaler.transform(df)

            # Predict
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0] # Get probabilities for both classes [P(Normal), P(Fraud)]
            fraud_proba = proba[1] # Probability of class 1 (Fraud)

            # Display results
            st.subheader("🔍 Transaction Analysis Result:")
            if prediction == 1:
                st.error("Prediction: 🚨 FRAUD DETECTED")
            else:
                st.success("Prediction: ✅ Normal Transaction")

            # Show confidence score (probability of fraud)
            st.metric(label="Confidence Score (Fraud Probability)", value=f"{fraud_proba:.2%}")

            # Optional: Show probabilities for both classes
            with st.expander("See detailed probabilities"):
                 st.write(f"Probability Normal (Class 0): {proba[0]:.4f}")
                 st.write(f"Probability Fraud (Class 1): {proba[1]:.4f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the input values are valid and the model/scaler are compatible.")

else:
    st.warning("Cannot proceed without loaded model and scaler.")

st.markdown("---")
st.caption("Ensure the model (.pkl) and scaler (.pkl) files are present.")