import streamlit as st 

import pandas as pd 

import pickle 

  

# Updated data for teams and cities 

teams = [ 

    'Chennai Super Kings', 

    'Sunrisers Hyderabad', 

    'Royal Challengers Bangalore', 

    'Delhi Capitals', 

    'Mumbai Indians', 

    'Kolkata Knight Riders', 

    'Rajasthan Royals', 

    'Punjab Kings' 

] 

  

cities = [ 

    'Chennai', 

    'Hyderabad', 

    'Bangalore', 

    'Delhi', 

    'Mumbai', 

    'Kolkata', 

    'Jaipur', 

    'Mohali' 

] 

  

# Mock prediction pipeline (replace this with actual prediction logic) 

# Load the model 

pipe = pickle.load(open('pipe.pkl', 'rb')) 

  

# Custom CSS for improved professional styling 

st.markdown(""" 

    <style> 

        /* Import Fjalla One and Roboto Fonts */ 

        @import url('https://fonts.googleapis.com/css2?family=Fjalla+One&family=Roboto:wght@300;400;500&display=swap'); 

  

        /* Set background color for the whole page with gradient */ 

        .main { 

            background: linear-gradient(135deg, #00A9E0 20%, #005B9A 80%); 

            color: white; 

        } 

  

        /* General text styling */ 

        h1, h2, h3, h4, h5, h6, p { 

            color: white; 

            font-family: 'Fjalla One', sans-serif; /* Bold headers */ 

        } 

  

        p { 

            font-family: 'Roboto', sans-serif; /* Regular text */ 

            font-weight: 300; 

            font-size: 16px; 

        } 

  

        /* Title font size */ 

        h1 { 

            font-size: 60px; 

            letter-spacing: 1.5px; 

            text-align: center; 

            margin-bottom: 20px; 

        } 

  

        /* Subheader font size */ 

        h2 { 

            font-size: 35px; 

            margin-bottom: 15px; 

        } 

  

        /* Section styling */ 

        .section-title { 

            color: #ADD8E6; 

            margin: 20px 0 10px; 

        } 

  

        /* Button styling */ 

        .stButton>button { 

            background-color: #005B9A;  /* Darker blue button background */ 

            color: #FFFFFF;  /* White text inside buttons */ 

            border: 2px solid #FFFFFF;  /* White border for buttons */ 

            padding: 10px; 

            font-size: 16px; 

            margin: 5px; 

            width: 200px; 

            font-family: 'Roboto', sans-serif; 

            border-radius: 10px; 

        } 

  

        /* Button hover effect */ 

        .stButton>button:hover { 

            background-color: #007BC7; 

            color: #FFFFFF; 

            border-color: #FFFFFF; 

            cursor: pointer; 

        } 

  

        /* Input fields styling */ 

        .stTextInput>div>input, .stNumberInput>div>input { 

            background-color: #EAF5FB; 

            color: black; 

            border: 2px solid #FFFFFF; 

            border-radius: 8px; 

            font-family: 'Roboto', sans-serif; 

        } 

  

        /* Divider styling */ 

        hr { 

            border-top: 2px solid #ADD8E6; 

            width: 80%; 

        } 

  

        /* Prediction result styling */ 

        .prediction-result { 

            font-size: 20px; 

            background-color: #ADD8E6; 

            color: #005B9A; 

            padding: 10px; 

            border-radius: 10px; 

            text-align: center; 

            font-family: 'Roboto', sans-serif; 

        } 

    </style> 

""", unsafe_allow_html=True) 

  

# Prediction Page Layout 

def prediction_page(): 

    st.markdown("Your match predictions, right here!") 

     

    # Input fields for batting and bowling teams 

    col1, col2 = st.columns(2) 

    with col1: 

        batting_team = st.selectbox('Select the batting team', sorted(teams)) 

    with col2: 

        bowling_team = st.selectbox('Select the bowling team', sorted(teams)) 

  

    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True) 

     

    # Input for city and target 

    selected_city = st.selectbox('Select city', sorted(cities)) 

    target = st.number_input('Target', min_value=0) 

  

    col3, col4, col5 = st.columns(3) 

    with col3: 

        score = st.number_input('Score', min_value=0) 

    with col4: 

        overs = st.number_input('Overs completed', min_value=0) 

    with col5: 

        wickets = st.number_input('Wickets lost', min_value=0) 

  

    if st.button('Predict Win Probability', key='predict_btn'): 

        runs_left = target - score 

        balls_left = 120 - (overs * 6) 

        wickets_left = 10 - wickets 

        crr = score / overs if overs > 0 else 0 

        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0 

         

        input_df = pd.DataFrame({ 

            'batting_team': [batting_team],  

            'bowling_team': [bowling_team], 

            'city': [selected_city],  

            'runs_left': [runs_left],  

            'balls_left': [balls_left], 

            'wickets': [wickets_left],  

            'total_runs_x': [target],  

            'crr': [crr],  

            'rrr': [rrr] 

        }) 

  

        result = pipe.predict_proba(input_df) 

        loss = result[0][0] 

        win = result[0][1] 

         

        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True) 

        st.markdown(f"<div class='prediction-result'>{batting_team}: {round(win * 100)}% chance of winning</div>", unsafe_allow_html=True) 

        st.markdown(f"<div class='prediction-result'>{bowling_team}: {round(loss * 100)}% chance of winning</div>", unsafe_allow_html=True) 

        st.markdown("---") 

  

# Function to display "About Cricket" page 

def about_cricket(): 

    st.subheader("About Cricket") 

    st.write(""" 

        Cricket is a bat-and-ball game played between two teams of eleven players on a field.  

        The game is played in various formats, including Test matches, One-Day Internationals (ODIs), and Twenty20 (T20).  

        In the Indian Premier League (IPL), the T20 format is followed, which is known for its fast-paced and high-energy matches. 

    """) 

  

# Function to explain how the predictor works 

def predictor_info(): 

    st.subheader("How the Predictor Works") 

    st.write(""" 

        Our prediction model uses historical match data and various match-related inputs like team performance,  

        runs scored, wickets lost, and current run rates. By processing this data, we can estimate the probability  

        of a team winning based on the ongoing match statistics. 

    """) 

  

# Main function for home screen and navigation 

def main(): 

    st.markdown("<h1 class='large-header'>IPL WIN PREDICTOR</h1>", unsafe_allow_html=True) 

  

    if "page" not in st.session_state: 

        st.session_state["page"] = "Home" 

  

    if st.session_state["page"] == "Home": 

        st.subheader("Welcome to the IPL Win Predictor!") 

        st.write(""" 

            This web app offers you insights into IPL history and predicts match outcomes based on current match statistics.  

            Navigate to various sections using the buttons below to explore IPL team history, get match predictions, learn about cricket,  

            or understand how our win predictor works. 

        """) 

  

        # Creating equal space between buttons using columns 

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1]) 

         

        with col1: 

            if st.button("Team History"): 

                st.session_state["page"] = "Team History" 

         

        with col2: 

            if st.button("Prediction"): 

                st.session_state["page"] = "Prediction" 

         

        with col3: 

            if st.button("About Cricket"): 

                st.session_state["page"] = "About Cricket" 

         

        with col4: 

            if st.button("How the Predictor Works"): 

                st.session_state["page"] = "How the Predictor Works" 

  

    elif st.session_state["page"] == "Team History": 

        st.write("Team History Page") 

        if st.button("Back to Home"): 

            st.session_state["page"] = "Home" 

  

    elif st.session_state["page"] == "Prediction": 

        prediction_page() 

        if st.button("Back to Home"): 

            st.session_state["page"] = "Home" 

  

    elif st.session_state["page"] == "About Cricket": 

        about_cricket() 

        if st.button("Back to Home"): 

            st.session_state["page"] = "Home" 

  

    elif st.session_state["page"] == "How the Predictor Works": 

        predictor_info() 

        if st.button("Back to Home"): 

            st.session_state["page"] = "Home" 

  

# Run the main function 

if __name__ == '__main__': 

    main() 

 
