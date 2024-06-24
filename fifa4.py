import streamlit as st  # Importing Streamlit
import time  # Importing time
import numpy as np  # Importing numpy
import pickle  # Importing pickle
import pandas as pd  # Importing pandas
import os  # Importing os
from sklearn.preprocessing import StandardScaler

# Setting the title of the app
st.title('FIFA Rating Prediction')

# Displaying a temporary welcome message
welcome_message = st.empty()
welcome_message.text("Move the sliders to match your desired ratings!")
time.sleep(1.5)  # Timer for the welcome message to terminate
welcome_message.empty()  # Code to remove the welcome message

player_name = st.text_input("Enter Player's Name")
# User inputs
feature1 = st.slider("Movement Reaction", min_value=0, max_value=100, value=50, step=1)
feature2 = st.slider("Potential", min_value=0, max_value=100, value=50, step=1)
feature3 = st.slider("Passing", min_value=0, max_value=100, value=50, step=1)
feature4 = st.slider("Wage (Euro)", min_value=0, max_value=3000000, value=50, step=1)
feature5 = st.slider("Mentality Composure", min_value=0, max_value=100, value=50, step=1)
feature6 = st.slider("Value (Euro)", min_value=0, max_value=1000000000, value=50, step=1)
feature7 = st.slider("Dribbling", min_value=0, max_value=100, value=50, step=1)
feature8 = st.slider("Attacking Short Passing", min_value=0, max_value=100, value=50, step=1)
feature9 = st.slider("Mentality Vision", min_value=0, max_value=100, value=50, step=1)
feature10 = st.slider("International Reputation", min_value=0, max_value=100, value=50, step=1)
feature11 = st.slider("Skill Long Passing", min_value=0, max_value=100, value=50, step=1)
feature12 = st.slider("Power Shot Power", min_value=0, max_value=100, value=50, step=1)
feature13 = st.slider("Physic", min_value=0, max_value=100, value=50, step=1)
feature14 = st.slider("Release Clause (Euros)", min_value=0, max_value=1000000000, value=50, step=1)
feature15 = st.slider("Age", min_value=0, max_value=100, value=50, step=1)
feature16 = st.slider("Skill Ball Control", min_value=0, max_value=100, value=50, step=1)

user_inputs_list = [
    feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8,
    feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16
]
user_inputs_list = np.array(user_inputs_list)

user_inputs = pd.DataFrame(user_inputs_list.reshape(1, -1), columns=[
    'movement_reactions', 'potential', 'passing', 'wage_eur',
    'mentality_composure', 'value_eur', 'dribbling',
    'attacking_short_passing', 'mentality_vision', 'international_reputation',
    'skill_long_passing', 'power_shot_power', 'physic', 'release_clause_eur',
    'age', 'skill_ball_control'
])

# Load the scaler and model
scaler_path = 'scaling2.pkl'
model_path = 'ranf_fifa_model.pkl'

if os.path.exists(scaler_path) and os.path.exists(model_path):
    # Scaling the user inputs
    loaded_scaler = pickle.load(open(scaler_path, 'rb'))
    scaled_user_inputs = pd.DataFrame(loaded_scaler.transform(user_inputs), columns=user_inputs.columns)

    # Load the trained model
    loaded_model = pickle.load(open(model_path, 'rb'))

    # Make predictions
    prediction = loaded_model.predict(scaled_user_inputs)

    if st.button('SUBMIT'):
        st.write(f"{player_name}'s predicted FIFA Rating is: {prediction[0]}")

else:
    st.write("Model or scaler file not found.")
