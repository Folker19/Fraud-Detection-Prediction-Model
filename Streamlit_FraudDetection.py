import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from scipy.stats import boxcox
import zipfile as zp

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.subplots as sp

# Page configuration
st.set_page_config(page_title="Fraud Detection Model 🕵🏻", layout="wide", page_icon="🔎")

# Title
st.title("Fraud Detection Model 🕵🏻")

# Load the data with Streamlit's cache decorator
@st.cache_data  
def load_data(path):
    df = pd.read_csv(path)
    return df

# Load the dataset raw
data_raw_path = 'data/transactions.csv'
df_raw = load_data(data_raw_path)

# Load the dataset processed
data_processed_path = 'data/transactions_processed.csv'
df = load_data(data_processed_path)

# Load the model and the column transformer
def boxcox_transform(X, lmbda):
    return boxcox(X, lmbda=lmbda)
columns_transformer = joblib.load("src\models\column_transformer.joblib")
model = joblib.load("src\models/best_cv_random_sampling_Random Forest_pipeline.joblib")# Random Sampling


# Sidebar
st.sidebar.title('Menu')
options = st.sidebar.radio('Select an option:', ['Introduction', 'EDA', 'Model', 'Conclusion'])

# Show the dataset
if options == 'Introduction':
    st.header('Introduction')
    st.markdown('***Description:***')
    st.write('This synthetic dataset, "transactions," has been generated with Pythons Faker library to simulate transaction data from an e-commerce platform with a focus on fraud detection. It includes a range of features commonly found in transactional data, along with additional attributes specifically designed to support the development and testing of fraud detection algorithms.')
    # st.write(df_raw.shape)
    st.write('Credits for the dataset to "SHRIYASH JAGTA" :https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data)')
    st.markdown('***Objective:***')
    st.write('The objective of this project is to develop a machine learning model that can predict whether a transaction is fraudulent or not. The model has been trained on the mentioned dataset which labels transactions as fraudulent or legitimate.')
    st.write('Size of the dataset: 1,472,592 rows and 16 columns')
    st.write(df_raw.head(5))
    
    # st.write(df_raw.head())
    # st.write(df_raw.shape)
# EDA
elif options == 'EDA':
    st.header('Exploratory Data Analysis (EDA)')
    st.subheader('Dataset preprocessing')
    st.write(df.head(5))
    st.write(df.shape)
    tab1, tab2 , tab3 = st.tabs(['Univariable Analysis', 'Bivariable Analysis', 'Multivariable Analysis'])

    # Univariable Analysis
    with tab1:
        st.subheader('Univariable Analysis:')
        st.image(r'graphs/univariate_analysis/target_variable_pie_chart.png', width=600)
        @st.cache_data 
        def fig_histogram():
            numeric_variables = ['Transaction Amount', 'Customer Age', 'Transaction Hour']
            fig = sp.make_subplots(rows=1, cols=len(numeric_variables), subplot_titles=numeric_variables) # create a figure with subplots

            # add a loop that creates a histogram for each numerical variable
            for i, variable in enumerate(numeric_variables, start=1):
                fig.add_trace(go.Histogram(x=df[variable], marker=dict(line=dict(width=0.5, color='DarkSlateGrey')), opacity=0.6, name=variable), row=1, col=i)

            fig.update_layout(title_text="Histograms of numerical variables", autosize=False ,width = 600*len(numeric_variables), height = 450) # set the size of the graph
            return fig 
        fig = fig_histogram()
        st.plotly_chart(fig, use_container_width=True) 

    # Bivariable Analysis    
    with tab2:
        st.subheader('Bivariable Analysis:')  
        st.image(r'graphs\bivariate_analysis/categorical_variables_barplots.png',use_column_width=True)  
        

        # Boxplot of important variables with plotly 
        @st.cache_data
        def fig_boxplot():
            numeric_variables = ['Transaction Amount', 'Transaction Hour', 'Account Age Days']
            fig = sp.make_subplots(rows=1, cols=len(numeric_variables), subplot_titles=[f"{var} vs Is Fraudulent" for var in numeric_variables])

            # change target variable labels to 'Legitimate' and 'Fraudulent'
            df_temp = df.copy() # create a copy of the dataframe to avoid modifying the original dataframe
            df_temp['Is Fraudulent'] = df_temp['Is Fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})

            # add a loop that creates a box plot for each numerical variable
            for i, variable in enumerate(numeric_variables, start=1):
                fig.add_trace(px.box(df_temp, x='Is Fraudulent', y=variable).data[0], row=1, col=i)
            fig.update_layout(title_text="Box Plots of numerical variables vs Target variable", autosize=False ,width = 600*len(numeric_variables), height = 500) # set the size of the graph
            return fig
        fig = fig_boxplot()
        st.plotly_chart(fig, use_container_width=True) 

        # Variables discretized
        col1, col2= st.columns(2)
        with col1:
            st.image(r'graphs\bivariate_analysis/account_age_bins_barplot.png', width=500)
        with col2:
            st.image(r'graphs\bivariate_analysis/transaction_Hour_bins_barplot.png', width=500)
    # Multivariable Analysis        
    with tab3:
        st.subheader('Multivariable Analysis:')
        st.image(r'graphs/multivariate_analysis/numerical_variables_corr_matrix.png',width=900)     

# Model
elif options == 'Model':
    tab1 , tab2 = st.tabs(['Model structure', 'Predictions'])

    # Model structure
    with tab1:
        st.header('Model Structure')

        col1, col2 , col3 ,col4,= st.columns(4)
        
        with col1:
            # Pipeline structure
            st.markdown('#### Pipeline Structure of the Model:')
            st.image(r'graphs/pipeline_percentil_sampling.png',width=500)

        with col2:   
            # First balance
            st.markdown('#### Proportion After Percentil Balancing the Dataset:')
            st.image(r'graphs/ML/target_variable_after_frist_percentil_sampling_pie_chart.png', width=400)
            # jump lines
            st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            # Second balance
            st.markdown('#### Proportion Before Sampling Techniques')
            st.image(r'graphs/ML/target_variable_after_second_percentil_sampling_pie_chart.png',width=400)
        
        with col3:   
            # Pipeline structure
            st.markdown('#### Pipeline Structure of the Model:')
            st.image(r'graphs/pipeline_random_sampling.png',width=500)
        with col4:

            # First balance
            st.markdown('#### Proportion After Random Balancing the Dataset')
            st.image(r'graphs/ML/target_variable_after_frist_random_balancing_pie_chart.png',width=400)
            # jump lines
            st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            #Second balance
            st.markdown('####  Proportion Before Sampling Techniques')
            st.image(r'graphs/ML/target_variable_after_second_random_sampling_pie_chart.png',width=400)  

    # Predictions   
    with tab2:    
        st.header('Model')
        st.markdown('Introduce the values for the transaction you would like to evaluate', unsafe_allow_html=True)

        @st.cache_resource
        def prediction(Transaction_Amount, Payment_Method, Product_Category,Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range) :
            # New dataset with the input data
            input_data = pd.DataFrame([[Transaction_Amount, Payment_Method, Product_Category, Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range]], 
                                    columns=['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity','Device Used','Shipping Billing Same', 'Account Age Range','Transaction Hour Range', 'Customer Age Range'])
            # Transform the input data
            input_data = columns_transformer.transform(input_data)
            # Make the prediction
            prediction = model.predict(input_data)

            if prediction == 0:
                pred = 'Legitimate'
            else:
                pred = 'Fraudulent'
            return pred
    
        # Create the input features
            
        Transaction_Amount= st.slider('Transaction Amount', 0, 2000, disabled=False)
        Payment_Method = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'Paypal', 'Bank Transfer'], index=0)
        Product_Category = st.selectbox('Product Category', ['Home & Garden', 'Electronics', 'Toys & Games', 'Clothing', 'Health & Beauty'], index=0)
        Quantity = st.slider('Quantity', 0, 10, disabled=False)
        Device_Used = st.selectbox('Device Used', ['Desktop', 'Mobile', 'Tablet'], index=0)
        Shipping_Billing_Same = st.selectbox('Shipping Billing Same', ['No', 'Yes'], index=0)
        if Shipping_Billing_Same == 'No':
            Shipping_Billing_Same = 0
        else:
            Shipping_Billing_Same = 1
        Account_Age_Range= st.selectbox('Account Age Range', ['1-3 months', '3-6 months', '6-9 months', '6-12 months'], index=0)
        Transaction_Hour_Range = st.selectbox('Transaction Hour Range', ['Dawn', 'Morning', 'Afternoon', 'Evening'], index=0)
        Customer_Age_Range = st.selectbox('Customer Age Range', ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69'], index=0)
        result =""

        # Button to make the prediction
        if st.button("Predict"): 
            result = prediction(Transaction_Amount, Payment_Method, Product_Category,Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range) 
            if result == 'Legitimate':
                st.success('Your transaction is {}'.format(result))
            else:
                st.error('Your transaction is {}'.format(result))

# Conclusion
elif options == 'Conclusion':
    st.header('Conclusion')
    st.markdown('- Umbalanced data can lead to biased models. In this kind of situations, changing the proportion of the minority class can help to improve the model\'s performance.')
    st.markdown('- Applying sampling techniques to balance the data improved the performance of the models. In this case, the Edited Nearest Neighbour undersampling technique was the best one based on the F1 score.')
    st.markdown('- The Random Forest model has the highest cross-validation F1 score. On the other side, we have tested AutoML with Pycaret library though no improvements on the model have been obsverved. We have taken the Random Forest model for deployment.')
    st.markdown('- When evaluating on the single test set, the Gradient Boosting model achieved the highest F1 score. On the other hand, when observing the Confusion Matrix, the model has a relatively high false positive rate for the Fraudulent class.')
    st.markdown('- We have tested Neuronal Networks which are a powerful tool for classification tasks. However, due to the nature of the data, we can observe is not providing any improvements compared to the other models.')