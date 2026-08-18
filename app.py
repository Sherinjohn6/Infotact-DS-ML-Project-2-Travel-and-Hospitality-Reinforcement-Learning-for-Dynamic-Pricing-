import streamlit as st
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hotel Dynamic Pricing",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Dynamic Pricing System")
st.write(
    "Hotel demand prediction and reinforcement learning "
    "for dynamic pricing."
)

st.divider()


# ============================================================
# LOAD Q-TABLE
# ============================================================

q_table_path = "q_table.npy"

if not os.path.exists(q_table_path):

    st.error(
        "q_table.npy was not found in the project folder."
    )

    st.stop()

try:

    q_table = np.load(q_table_path)

except Exception as e:

    st.error(f"Unable to load q_table.npy: {e}")

    st.stop()


# ============================================================
# LOAD LIGHTGBM MODEL
# ============================================================

model_path = "models/lightgbm_model.pkl"

if not os.path.exists(model_path):

    st.warning(
        "LightGBM model was not found. "
        "The Q-learning pricing demonstration can still run."
    )

    model = None

else:

    try:

        model = joblib.load(model_path)

        st.success("Q-table and LightGBM model loaded successfully.")

    except Exception as e:

        st.warning(
            f"LightGBM model could not be loaded: {e}"
        )

        model = None


# ============================================================
# DEMAND INPUT
# ============================================================

st.header("📊 Demand Information")

col1, col2 = st.columns(2)

with col1:

    demand = st.slider(
        "Expected Demand",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

with col2:

    base_price = st.number_input(
        "Base Hotel Price (₹)",
        min_value=100.0,
        value=1000.0,
        step=100.0
    )


# ============================================================
# DEMAND STATE
# ============================================================

if demand < 0.40:

    state = 0
    demand_level = "Low Demand"

elif demand < 0.70:

    state = 1
    demand_level = "Medium Demand"

else:

    state = 2
    demand_level = "High Demand"


# ============================================================
# Q-LEARNING ACTION
# ============================================================

best_action = int(np.argmax(q_table[state]))

actions = {
    0: "Decrease Price",
    1: "Keep Price",
    2: "Increase Price"
}

recommended_action = actions[best_action]


# ============================================================
# RECOMMENDED PRICE
# ============================================================

if best_action == 0:

    recommended_price = base_price * 0.90

elif best_action == 1:

    recommended_price = base_price

else:

    recommended_price = base_price * 1.10


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.header("🎯 Pricing Recommendation")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Demand Level",
        demand_level
    )

with col2:

    st.metric(
        "Base Price",
        f"₹{base_price:,.2f}"
    )

with col3:

    st.metric(
        "Recommended Action",
        recommended_action
    )

with col4:

    st.metric(
        "Recommended Price",
        f"₹{recommended_price:,.2f}"
    )


# ============================================================
# Q-TABLE
# ============================================================

st.divider()

st.header("🤖 Q-Learning Policy")

q_table_display = q_table.copy()

st.dataframe(
    q_table_display,
    use_container_width=True
)


# ============================================================
# EXPLANATION
# ============================================================

st.subheader("💡 Recommendation")

if best_action == 0:

    st.info(
        "Demand is low. The system recommends decreasing "
        "the hotel price to attract more customers."
    )

elif best_action == 1:

    st.info(
        "Demand is moderate. The system recommends keeping "
        "the current hotel price."
    )

else:

    st.info(
        "Demand is high. The system recommends increasing "
        "the hotel price to maximize revenue."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hotel Demand Prediction & Reinforcement Learning "
    "for Dynamic Pricing"
)