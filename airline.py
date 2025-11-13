import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Airline Passenger Satisfaction", page_icon="✈️")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home",
                                             "Exploratory Data Analysis",
                                             "Model Training and Evaluation",
                                             "Make Predictions!"])

# Load dataset
df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')
#remove customer type for more honest modeling
#df.drop(columns=['customer_type'], inplace=True)

# Home Page
if page == "Home":
    st.title("✈️ Airline Dataset Explorer")

    st.subheader("Welcome to our Airline Passenger Satisfaction Explorer app!")
    st.write("""
        This app provides an interactive platform to explore the airline passenger satisfaction dataset. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://storage.googleapis.com/kaggle-datasets-images/522275/959195/b0e445e8d51cbbb098917a006378d829/dataset-cover.png?t=2020-02-20-16-56-51', caption="Kaggle Picture")
    
    st.subheader("About the Data")
    st.write("""
             The Airline Passenger Satisfaction dataset contains survey data collected from airline passengers.
             It includes features such as flight distance, passenger age, seat comfort, and more, with the target variable being the passenger's satisfaction level (satisfied or dissatisfied).
    """)
    st.link_button("Kaggle Dataset Reference", 'https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data')

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df) ### Future Work: show labels from uncleaned data
    
    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    percent_female = df['gender'].mean() * 100
    st.write(f"The average age of passengers is {df['age'].mean():.2f} years. Females comprise of {percent_female:.2f}% of passengers.")

    st.subheader("Select the type of visualization you'd like to explore:")
    #eda_type = st.multiselect("Visualization Options", ['Histograms', 'Scatterplots', 'Box Plots', 'Count Plots'])
    eda_type = st.selectbox("Visualization Options", ['Survey Boxplots', 'Box Plots', 'Scatterplots', 'Histograms', 'Count Plots'])
    #Seperate Survey Response Column from the rest of the dataset
    survey_cols = df.columns[6:20].tolist()
    #st.write("Survey Response Columns:", survey_cols)

    # Seperate Columns with Continuous Numerical data from the rest of the dataset
    cont_cols = cont_cols = ['age', 'flight_distance', 'departure_delay_in_minutes', 'arrival_delay_in_minutes']


    # obj_cols = df.select_dtypes(include='object').columns.tolist()
    # num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Distributions")
        b_selected_col = st.selectbox("Select a column for the box plot:", cont_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x=b_selected_col, title=chart_title, color='satisfaction'))

    if 'Histograms' in eda_type:
        ### Future Work: make departure and arival variables more readable
        st.subheader("Histograms - Visualizing Continuous Distributions")
        h_selected_col = st.selectbox("Select a column for the histogram:", cont_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Survey Boxplots' in eda_type:
         st.subheader("Box Plots -Survey Distributions")
         h_selected_col = st.selectbox("Select a column for the histogram:", survey_cols)
         chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
         if st.checkbox("Show by Satisfaction"):
             ### Future Work: label satisfaction
             st.plotly_chart(px.box(df, x=h_selected_col, title=chart_title, color='satisfaction'))
         else:
             st.plotly_chart(px.box(df, x=h_selected_col, title=chart_title))
            
    if 'Scatterplots' in eda_type:
        #age, flight distance, departure delay, arrival delay
        st.subheader("Scatterplots - Visualizing Relationships")
        st.plotly_chart(px.scatter(df, 
                                   x='arrival_delay_in_minutes',
                                    y='flight_distance', 
                                    title = "Flight Distance vs. Arrival Delay or Departure Delay"))
        st.plotly_chart(px.scatter(df, 
                                   x='departure_delay_in_minutes',
                                    y='flight_distance'))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Data")
        cat_col = ['customer_type', 'type_of_travel', 'class', 'satisfaction']
        selected_col = st.selectbox("Select a Category:", cat_col)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, title=chart_title))

# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']
    ### Future Work: model only trains on survey columns by default, toggle to add other factors
        #st.multiselect("Include", ["age", "customer_type", "type_of_travel", "class", "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=11)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈️ Make Predictions")

    st.subheader("Adjust the values below to make predictions on passenger satisfaction:")
    # Pick random demographic and flight data from test set
    rand_customer = df.sample(random_state=42).drop(columns=['satisfaction']).reset_index(drop=True)
    ### Future work: Add Rand_Customer info to interface
    #st.write(f"You are a {rand_customer['customer_type'].values[0]} traveling for {rand_customer['type_of_travel'].values[0]} in {rand_customer['class'].values[0]} class.")

    # User inputs for prediction
    st.write("On a scale of 1 to 5, how would you rate the following services? With 1 being the worst and 5 being the best.")
    cleanliness = st.slider("How Clean was the Plane?", min_value= 1, max_value= 5, value= 3)
    inflight_wifi_service = st.slider("How was the Inflight WiFi Service? 0 indicating no WiFi available", min_value= 0, max_value= 5, value= 0)
    departure_arrival = st.slider("How convenient was the Departure and Arrival Times?", min_value= 1, max_value= 5, value= 3)
    ### Future Work: Rest of the Survey Data
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    business = st.toggle("Business")

    # User input dataframe overwrites Rand_Customer
    user_input = pd.DataFrame({
        'cleanliness': [cleanliness],
        'inflight_wifi_service': [inflight_wifi_service],
        'departure/arrival_time_convenient': [departure_arrival],
        #Rest of survey data
        'age': [age],
        'type_of_travel': [business]
    })
    rand_customer.update(user_input)
    
    # Display the result
    if st.button("Make Prediction"):
        # Prepare the data
        X = df.drop(columns = 'satisfaction')
        y = df['satisfaction']

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Scale the user input
        user_input_scaled = scaler.transform(rand_customer)

        # Use KNN (k=11) as the model for predictions
        model = KNeighborsClassifier(n_neighbors=11)
        # Train the model on the scaled data
        model.fit(X_scaled, y)

        # Make predictions
        prediction = model.predict(user_input_scaled)[0]

        if  prediction == 1:
            st.write("I'm glad you enjoyed your flight! ✈️")
            st.balloons()
        else:
            st.write("I'm sorry to hear that your flight experience was not satisfactory. 😞")
            st.snow()