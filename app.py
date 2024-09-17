import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and layout
st.set_page_config(page_title="Comprehensive ML Platform", layout="wide")

# Title
st.title("Comprehensive ML Platform for Flower Species Prediction")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Data Upload and Preprocessing", "Model Training", "Visualization", "Prediction"])

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Data Upload and Preprocessing Page
if page == "Data Upload and Preprocessing":
    st.header("Data Upload and Preprocessing")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.write("Data Preview:")
            st.write(df.head())

            st.subheader("Data Information")
            st.write(df.describe())

            st.subheader("Select Features for Prediction")
            feature_columns = df.columns[:-1]  # Assuming the last column is the target
            length_column = st.selectbox("Select length column", feature_columns)
            width_column = st.selectbox("Select width column", feature_columns)
            target_column = df.columns[-1]

            if st.button("Preprocess Data"):
                # Create preprocessing pipeline
                preprocessor = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

                # Fit and transform the data
                X = df[[length_column, width_column]]
                y = df[target_column]

                X_preprocessed = pd.DataFrame(preprocessor.fit_transform(X), columns=[length_column, width_column])
                
                st.write("Preprocessed Data Preview:")
                st.write(X_preprocessed.head())

                # Store preprocessed data and related information in session state
                st.session_state.X_preprocessed = X_preprocessed
                st.session_state.y = y
                st.session_state.preprocessor = preprocessor
                st.session_state.length_column = length_column
                st.session_state.width_column = width_column
                st.session_state.target_column = target_column

                st.success("Data preprocessing completed successfully!")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")

    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
    else:
        algorithm = st.selectbox("Select algorithm", ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"])
        
        if st.button("Train Model"):
            try:
                X = st.session_state.X_preprocessed
                y = st.session_state.y
                
                # Encode target if it's categorical
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                
                if algorithm == "Logistic Regression":
                    model = LogisticRegression()
                elif algorithm == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif algorithm == "Random Forest":
                    model = RandomForestClassifier()
                else:  # Support Vector Machine
                    model = SVC(probability=True)
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model trained successfully. Accuracy: {accuracy:.4f}")
                
                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred, target_names=le.classes_))

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                # Store the model and label encoder in session state
                st.session_state.model = model
                st.session_state.label_encoder = le

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

# Visualization Page
elif page == "Visualization":
    st.header("Data Visualization")

    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
    else:
        viz_type = st.selectbox("Select visualization type", ["Scatter Plot", "Histogram", "Box Plot", "Pair Plot"])
        
        if st.button("Visualize"):
            try:
                if viz_type == "Scatter Plot":
                    fig = px.scatter(
                        st.session_state.data, 
                        x=st.session_state.length_column, 
                        y=st.session_state.width_column, 
                        color=st.session_state.target_column,
                        title=f"Scatter Plot: {st.session_state.length_column} vs {st.session_state.width_column}"
                    )
                elif viz_type == "Histogram":
                    fig = px.histogram(
                        st.session_state.data, 
                        x=st.session_state.length_column, 
                        color=st.session_state.target_column,
                        title=f"Histogram of {st.session_state.length_column}"
                    )
                elif viz_type == "Box Plot":
                    fig = px.box(
                        st.session_state.data, 
                        y=st.session_state.width_column, 
                        color=st.session_state.target_column,
                        title=f"Box Plot of {st.session_state.width_column}"
                    )
                elif viz_type == "Pair Plot":
                    fig = px.scatter_matrix(
                        st.session_state.data,
                        dimensions=[st.session_state.length_column, st.session_state.width_column],
                        color=st.session_state.target_column,
                        title="Pair Plot"
                    )
                
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during visualization: {str(e)}")

# Prediction Page
elif page == "Prediction":
    st.header("Model Prediction")

    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        st.write("Enter values for prediction:")
        
        length = st.number_input(f"Enter {st.session_state.length_column}", value=0.0, step=0.1)
        width = st.number_input(f"Enter {st.session_state.width_column}", value=0.0, step=0.1)
        
        if st.button("Predict"):
            try:
                input_data = pd.DataFrame([[length, width]], columns=[st.session_state.length_column, st.session_state.width_column])
                processed_input = st.session_state.preprocessor.transform(input_data)
                prediction = st.session_state.model.predict(processed_input)
                prediction_proba = st.session_state.model.predict_proba(processed_input)
                
                predicted_species = st.session_state.label_encoder.inverse_transform(prediction)
                st.write(f"Predicted species: {predicted_species[0]}")
                
                st.subheader("Prediction Probabilities")
                proba_df = pd.DataFrame(prediction_proba, columns=st.session_state.label_encoder.classes_)
                st.write(proba_df)

                # Visualize prediction probabilities
                fig = px.bar(proba_df.T, title="Prediction Probabilities")
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Comprehensive ML Platform v1.0")