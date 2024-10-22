import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np #per ricavare la root di mse

#librerie per la pca e predizione
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

st.set_page_config(page_title="Videogame Sales", page_icon="🕹️", layout="wide")

#caricamento del dataset
dataset_path = "Video_Games.csv" 
df = pd.read_csv(dataset_path)
st.title("VIDEOGAMES SALES 🕹️")
#mostra dataset
st.write("Dataset completo:")
st.write(df)

#gestione dei valori nulli sulle righe e creazione dataframe vendite
df_sales = df.iloc[:,0:11] #selezione delle feature che riguardano i dati di vendita
df_sales = df_sales.dropna().sort_values(by='Year_of_Release') #rimuove i valori nulli e riordina per anno di rilascio

#dataframe relativo all'anno di rilascio
df_year = df_sales.groupby('Year_of_Release').sum() #vendite totali per ogni anno
df_year['count'] = df_sales.groupby('Year_of_Release').count()['index'] #conta il numero di giochi ogni anno
df_year = df_year.reset_index() #trasforma l'indice precedente in una colonna
df_year['sales_to_count'] = df_year['Global_Sales']/df_year['count'] #media di vendite globali su ciascun gioco
df_year['Year_of_Release'] = df_year['Year_of_Release'].astype('int32') #datatype dell'anno da float a int

_='''---ANALISI DATI VENDITA NEL TEMPO PER REGIONE O GLOBALI---'''

st.header("Evoluzione dati di vendita nel tempo")

column_labels_mapping = {"Vendite in Nord America": "NA_Sales", "Vendite in Giappone": "JP_Sales",
                          "Vendite in Europa": "EU_Sales","Vendite Globali": "Global_Sales","Altri Mercati": "Other_Sales"}

#seleziona colonna per l'analisi interattiva
selected_column_interactive_label = st.selectbox("Seleziona una colonna per l'analisi:",
                                                 [""] + list(column_labels_mapping.keys()))  #voce vuota 

#verifica selezione una colonna
if selected_column_interactive_label:
   
    selected_column_interactive = column_labels_mapping[selected_column_interactive_label]

    #dataFrame aggregato
    df_aggregated_interactive = df_sales.groupby('Year_of_Release')[[selected_column_interactive]].sum().reset_index()

    #grafico interattivo
    fig_interactive = px.line(df_aggregated_interactive, x='Year_of_Release', y=selected_column_interactive,
                              title=f'Analisi interattiva - {selected_column_interactive}', line_shape='linear',
                              labels={'value': 'Vendite', 'Year_of_Release': 'Anno'})

    st.plotly_chart(fig_interactive)

else:
    st.warning("Seleziona almeno una colonna per l'analisi interattiva.")

_='''---VENDITE PER REGIONE E GLOBALI PER ANNO---'''
st.title('Vendite in un anno')

#filtra in base all'anno selezionato
selected_year = st.slider('Seleziona un anno:', min_value=int(df['Year_of_Release'].min()), max_value=int(df['Year_of_Release'].max()), step=1)

filtered_df = df_sales[df_sales['Year_of_Release'] == selected_year]

fig_regional = px.bar(filtered_df, x='Year_of_Release', y=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                      title=f'Vendite Regionali nel {selected_year}', labels={'value': 'Vendite (milioni di copie)',
                                                                              'Year_of_Release':'Anno'})

fig_regional.update_layout(barmode='group')

global_sales_value = filtered_df['Global_Sales'].sum()

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig_regional)

with col2:
    st.markdown(f"<h1 style='text-align:center;'>Vendite Globali nel {selected_year}</h1>"
                f"<h1 style='text-align:center; font-size: 24px;'>{global_sales_value:.2f} milioni di copie</h1>",
                unsafe_allow_html=True)


_='''---ANALISI DATI VENDITA NEL TEMPO PER PIATTAFORMA---'''
#analisi dei dati interattiva per il volume venduto nel tempo basato sulla piattaforma
st.header("Analisi con sovrapposizione dati di vendita per piattaforma di gioco")

#seleziona piattaforma
selected_elements = st.multiselect("Seleziona piattaforma di gioco:", df['Platform'].unique())

options=["", 'Vendite in Nord America', 'Vendite in Giappone', 'Vendite in Europa',
          'Vendite Globali','Altri Mercati', ]
#seleziona il tipo di analisi
selected_analysis_type = st.selectbox("Seleziona il tipo di analisi:",
                                      options,key='key1') 

if selected_elements and selected_analysis_type:
    #filtra per le piattaforme selezionate
    df_filtered = df_sales[df_sales['Platform'].isin(selected_elements)]

    #seleziona colonne in base al tipo di analisi
    if selected_analysis_type == 'Vendite in Nord America':
        selected_columns_interactive = ['NA_Sales']
    elif selected_analysis_type == 'Vendite in Giappone':
        selected_columns_interactive = ['JP_Sales']
    elif selected_analysis_type == 'Vendite in Europa':
        selected_columns_interactive = ['EU_Sales']
    elif selected_analysis_type == 'Vendite Globali':
        selected_columns_interactive = ['Global_Sales']
    elif selected_analysis_type == 'Altri Mercati':
        selected_columns_interactive = ['Other_Sales']
    else:
        selected_columns_interactive = []

    #creazione dataframe aggregato per il volume totale basato sulla piattaforma
    df_aggregated_volume = df_filtered.groupby(['Year_of_Release', 'Platform'])[selected_columns_interactive].sum().reset_index()

    fig_volume = px.line(df_aggregated_volume, x='Year_of_Release', y=selected_columns_interactive,
                                 color='Platform', title=f'{selected_analysis_type} nel tempo per piattaforma di gioco',
                                 line_shape='linear', labels={'value': 'Vendite', 'Year_of_Release': 'Anno'},
                                 color_discrete_sequence=px.colors.qualitative.Set1)
    
    df_aggregated_volume_2 = df_filtered.groupby(['Platform'])[selected_columns_interactive].sum().reset_index()

    fig_volume_2 = px.histogram(df_aggregated_volume_2, x='Platform', y=selected_columns_interactive,
                          color='Platform', title=f'{selected_analysis_type} per piattaforma di gioco',
                          labels={'value': 'Vendite', 'Platform': 'Piattaforma'},
                          color_discrete_sequence=px.colors.qualitative.Set1)

    col1, col2 = st.columns(2)

    col1.plotly_chart(fig_volume)

    col2.plotly_chart(fig_volume_2)

else:
    st.warning("Seleziona almeno una piattaforma e un tipo di analisi.")


_='''---ANALISI DATI VENDITA NEL TEMPO PER GENERE---'''
#analisi dei dati interattiva per il volume venduto nel tempo basato sul genere
st.header("Analisi con sovrapposizione dati di vendita per genere")

#seleziona genere
selected_elements = st.multiselect("Seleziona genere:", df['Genre'].unique())

#seleziona il tipo di analisi
selected_analysis_type = st.selectbox("Seleziona il tipo di analisi:",
                                      options,key='key2')  # Aggiungi una voce vuota

if selected_elements and selected_analysis_type:
    #filtra per le piattaforme selezionate
    df_filtered = df_sales[df_sales['Genre'].isin(selected_elements)]

    if selected_analysis_type == 'Vendite in Nord America':
        selected_columns_interactive = ['NA_Sales']
    elif selected_analysis_type == 'Vendite in Giappone':
        selected_columns_interactive = ['JP_Sales']
    elif selected_analysis_type == 'Vendite in Europa':
        selected_columns_interactive = ['EU_Sales']
    elif selected_analysis_type == 'Vendite Globali':
        selected_columns_interactive = ['Global_Sales']
    elif selected_analysis_type == 'Altri Mercati':
        selected_columns_interactive = ['Other_Sales']
    else:
        selected_columns_interactive = []

    df_aggregated_volume = df_filtered.groupby(['Year_of_Release', 'Genre'])[selected_columns_interactive].sum().reset_index()

    fig_volume = px.line(df_aggregated_volume, x='Year_of_Release', y=selected_columns_interactive,
                                 color='Genre', title=f'{selected_analysis_type} nel tempo per genere',
                                 line_shape='linear', labels={'value': 'Vendite', 'Year_of_Release': 'Anno'},
                                 color_discrete_sequence=px.colors.qualitative.Set1)

    df_aggregated_volume_2 = df_filtered.groupby(['Genre'])[selected_columns_interactive].sum().reset_index()

    fig_volume_2 = px.histogram(df_aggregated_volume_2, x='Genre', y=selected_columns_interactive,
                          color='Genre', title=f'{selected_analysis_type} per genere',
                          labels={'value': 'Vendite', 'Genre': 'Genere'},
                          color_discrete_sequence=px.colors.qualitative.Set1)

    col1, col2 = st.columns(2)

    col1.plotly_chart(fig_volume)

    col2.plotly_chart(fig_volume_2)

else:
    st.warning("Seleziona almeno un genere e un tipo di analisi.")



_='''
# Analisi in serie
st.write("Analisi in serie:")
st.line_chart(df.set_index('Year_of_Release')['NA_Sales'])
st.line_chart(df.set_index('Year_of_Release')['JP_Sales'])
st.line_chart(df.set_index('Year_of_Release')['EU_Sales'])
st.line_chart(df.set_index('Year_of_Release')['Global_Sales'])
st.line_chart(df.set_index('Year_of_Release')['Other_Sales'])
'''


#Modello SARIMA
st.header("Modello di previsione temporale (SARIMA)")

df_sales['Year_of_Release'] = df_sales['Year_of_Release'].astype('int32')

time_column_sarima = 'Year_of_Release' 
sales_column_sarima = 'Global_Sales'  

df_aggregated_sarima = df_sales.groupby(time_column_sarima)[sales_column_sarima].mean().reset_index()

if len(df_aggregated_sarima) < 2:
    st.error("Non ci sono abbastanza dati per generare le previsioni.")
else:
    order_sarima = (1, 1, 2)
    seasonal_order_sarima = (0, 1, 0, 52)

    model_sarima = sm.tsa.SARIMAX(df_aggregated_sarima[sales_column_sarima], order=order_sarima, seasonal_order=seasonal_order_sarima)
    result_sarima = model_sarima.fit()

    #tabelle non necessarie
    #st.subheader("Risultati del modello SARIMA:")
    #st.write(result_sarima.summary())

    forecast_period_years_sarima = st.slider("Seleziona il periodo di previsione SARIMA in anni", 1, 30, 7)

    if st.button("Esegui previsioni SARIMA"):
        last_year_sarima = df_aggregated_sarima[time_column_sarima].max()
        forecast_period_start_sarima = last_year_sarima + 1
        forecast_period_end_sarima = forecast_period_start_sarima + forecast_period_years_sarima

        forecast_index_sarima = pd.Index(range(forecast_period_start_sarima, forecast_period_end_sarima), name=time_column_sarima)

        forecast_df_sarima = pd.DataFrame({time_column_sarima: forecast_index_sarima,
                                            'Forecast': result_sarima.get_forecast(steps=len(forecast_index_sarima)).predicted_mean})

        fig_sarima = px.line(df_aggregated_sarima, x=time_column_sarima, y=sales_column_sarima, title=f'{sales_column_sarima} con Previsioni SARIMA')
        fig_sarima.add_scatter(x=forecast_df_sarima[time_column_sarima], y=forecast_df_sarima['Forecast'], mode='lines', name='Previsioni Future', line=dict(color='red'))
        
        
        fig_sarima.update_layout(xaxis_title="Anno", yaxis_title="Average Global Sales")
        fig_sarima.update_layout(title="Previsione su Average Global Sales")

        st.plotly_chart(fig_sarima)



# Modello ARIMA
st.header("Modello di previsione temporale (ARIMA)")
time_column_arima = 'Year_of_Release'
sales_column_arima = 'Global_Sales'

#df_sales['Year_of_Release'] = df_sales['Year_of_Release'].astype('int32')

df_aggregated_arima = df_sales.groupby(time_column_arima)[sales_column_arima].mean().reset_index()

if len(df_aggregated_arima) < 2:
    st.error("Non ci sono abbastanza dati per generare le previsioni.")
else:
    #addestramento del modello
    order_arima = (1, 1, 1)
    model_arima = sm.tsa.ARIMA(df_aggregated_arima[sales_column_arima], order=order_arima)
    result_arima = model_arima.fit()

    #st.subheader("Risultati del modello ARIMA:")
    #st.write(result_arima.summary())

    #interazione con l'utente per selezionare il periodo di previsione
    forecast_period_years_arima = st.slider("Seleziona il periodo di previsione ARIMA (anni)", 1, 30, 7)

    if st.button("Esegui previsioni ARIMA"):
        last_year_arima = df_aggregated_arima[time_column_arima].max()
        forecast_period_start_arima = last_year_arima + 1
        forecast_period_end_arima = forecast_period_start_arima + forecast_period_years_arima
        forecast_index_arima = pd.Index(range(forecast_period_start_arima, forecast_period_end_arima), name=time_column_arima)

        forecast_df_arima = pd.DataFrame({time_column_arima: forecast_index_arima, 
                                           'Forecast': result_arima.get_forecast(steps=len(forecast_index_arima)).predicted_mean})

        fig_arima = px.line(df_aggregated_arima, x=time_column_arima,
                             y=sales_column_arima, title=f'{sales_column_arima} con Previsioni ARIMA')
        fig_arima.add_scatter(x=forecast_df_arima[time_column_arima], y=forecast_df_arima['Forecast'],
                               mode='lines', name='Previsioni Future', line=dict(color='red'))
        st.plotly_chart(fig_arima)


#grafici che incrociano score, rating e vendite

st.title('Vendite Globali in Base a Critic Score, User Score e Ratings')

selected_x = st.selectbox('Seleziona il criterio:', ['Critic_Score','User_Score','Rating'])

fig = px.scatter(df, x=selected_x, y='Global_Sales',
                 color=selected_x, size_max=15, opacity=0.7,
                 labels={'Critic_Score': 'Critic Score', 'User_Score': 'User Score',
                          'Rating': 'Rating', 'Global_Sales': 'Global Sales (in millions)'})

fig.update_layout(title=f'Vendite Globali in Base a {selected_x}',
                  xaxis_title=selected_x, yaxis_title='Global Sales')

st.plotly_chart(fig)

st.markdown('<p style="text-align:center; font-size:small;">Realizzato da Santoro Emanuele, Morgillo Michele, Mondillo Angelica</p>',
             unsafe_allow_html=True)