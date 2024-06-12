import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from scipy.stats import boxcox

# Cargar el dataset
data_path = 'data/transactions_processed.csv'
df = pd.read_csv(data_path)

# Cargar el modelo y el transformador
def boxcox_transform(X, lmbda):
    return boxcox(X, lmbda=lmbda)
columns_transformer = joblib.load("src\models\column_transformer.joblib")
# model = joblib.load("src\models/best_test_random_sampling_Gradient Boosting_pipeline.joblib")# Random Sampling
# model = joblib.load("src/best_test_Gradient_Boosting_pipeline.joblib")# Gradient Boosting
model = joblib.load("src\models/final_best_model_pipeline.joblib")#pycaret


# Configuración de la página
st.set_page_config(page_title="Fraud Detection Model 🕵🏻", layout="wide", page_icon="🔎")

# Título de la aplicación
st.title("Fraud Detection Model 🕵🏻")

# Sidebar para navegación
st.sidebar.title('Navegación')
options = st.sidebar.radio('Selecciona una opción:', ['Introduction', 'EDA', 'Model', 'Gráficas del Modelo'])

# Mostrar dataset
if options == 'Introduction':
    st.header('Introduction')
    st.write('El fraude en el comercio electrónico es un problema significativo y en aumento, que ocasiona pérdidas financieras para las empresas y una experiencia negativa para los clientes.')
    st.write('Los estafadores utilizan diversas tácticas, como tarjetas de crédito robadas, secuestro de cuentas y fraude amistoso, lo que provoca devoluciones de cargos, interrupciones operativas y daños a la reputación.')
    st.write('Por eso es importante contar con un modelo que pueda predecir si una transacción es fraudulenta o legítima. En este proyecto, se han utilizado datos de transacciones de comercio electrónico para analisar los')
    st.write('patrones de fraude y crear un modelo que pueda predecir si una transacción es fraudulenta o legítima.')
    st.write('El dataset despues de ser procesado es de la siguiente manera:')
    st. write(df.head())
    st.write(df.shape)
    st.write('Creditos :https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions/data)')
    

# EDA
elif options == 'EDA':
    st.header('Exploratory Data Analysis (EDA)')
    st.subheader('Estadísticas Descriptivas')
    st.write(df.describe())
    #Definir columnas
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Distribución de las Variables')
        selected_column = st.selectbox('Selecciona una columna para ver su distribución:', df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader('Gráficas de Dispersión')
        x_column = st.selectbox('Selecciona la variable X:', df.columns)
        y_column = st.selectbox('Selecciona la variable Y:', df.columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
        st.pyplot(fig)

# Modelo
elif options == 'Model':
    st.header('Model')
    st.markdown('Introduce the values for the transaction you would like to evaluate', unsafe_allow_html=True)

    def prediction(Transaction_Amount, Payment_Method, Product_Category,Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range) :
        # Crear un DataFrame con los nombres de columna correctos
        input_data = pd.DataFrame([[Transaction_Amount, Payment_Method, Product_Category, Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range]], 
                                  columns=['Transaction Amount', 'Payment Method', 'Product Category', 'Quantity','Device Used','Shipping Billing Same', 'Account Age Range','Transaction Hour Range', 'Customer Age Range'])
        # # Transformar los datos
        st.write(input_data) # In case you want to display the input data before transformation
        input_data = columns_transformer.transform(input_data)
        st.write(input_data) # In case you want to display the input data before prediction
        # Hacer la predicción
        prediction = model.predict(input_data)

        if prediction == 0:
            pred = 'Legitimate'
        else:
            pred = 'Fraudulent'
        return pred
    
    # Create the input features
         
    Transaction_Amount= st.slider('Transaction Amount', 0, 12500, disabled=False)
    Payment_Method = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'Paypal', 'Bank Transfer'], index=0)
    Product_Category = st.selectbox('Product Category', ['Home & Garden', 'Electronics', 'Toys & Games', 'Clothing', 'Health & Beauty'], index=0)
    Quantity = st.slider('Quantity', 0, 20, disabled=False)
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

    # Hacer la predicción cuando se haga clic en 'Predict'
    if st.button("Predict"): 
        result = prediction(Transaction_Amount, Payment_Method, Product_Category,Quantity, Device_Used, Shipping_Billing_Same,  Account_Age_Range, Transaction_Hour_Range, Customer_Age_Range) 
        if result == 'Legitimate':
            st.success('Your transaction is {}'.format(result))
        else:
            st.error('Your transaction is {}'.format(result))

# Gráficas del Modelo
elif options == 'Gráficas del Modelo':
    st.header('Gráficas del Modelo')
    st.write('Aquí mostraremos las gráficas del modelo.')
    # Agregar tus gráficas del modelo aquí