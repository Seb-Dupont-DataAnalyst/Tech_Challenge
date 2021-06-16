import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsCV
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn import metrics
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import requests
import streamlit.components.v1 as components


link = "https://raw.githubusercontent.com/murpi/wilddata/master/test/history.csv"
df_sales = pd.read_csv(link)

st.set_page_config(layout='wide')

# Titre principale
#components.html("<body style='color:white;font-family:verdana; font-size:60px; border: 2px solid white; text-align: center; padding: 1px'><b>Cinéma Le Creusois</b></body>")
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)
st.markdown('<body class="title">Tech Challenge - Data Science Developer</body>', unsafe_allow_html=True)

# Création Sidebar avec les différents choix

choice = st.sidebar.radio("", ('Accueil', "1. Analyse descriptive exploratoire (EDA)", '2. Trouver la boutique correspondante',"3. Explication de l'impact de la météo",'4. Prévisions de ventes'
))

df_sales['DATE'] = pd.to_datetime(df_sales['DATE']).dt.strftime('%Y-%m-%d')
df_sales_pivot = df_sales.pivot_table(index='DATE', columns='ITEM', values='SALES')
df_sales_pivot.reset_index(inplace=True)

df_bordeaux = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/test/bordeaux2019.csv", skiprows=3)
df_lille = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/test/lille2019.csv", skiprows=3)
df_lyon = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/test/lyon2019.csv", skiprows=3)
df_marseille = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/test/marseille2019.csv", skiprows=3)

    
df_sales_bordeaux = pd.merge(df_sales_pivot,df_bordeaux,on='DATE', how='left')
df_sales_lille = pd.merge(df_sales_pivot,df_lille,on='DATE', how='left')
df_sales_lyon = pd.merge(df_sales_pivot,df_lyon,on='DATE', how='left')
df_sales_marseille = pd.merge(df_sales_pivot,df_marseille,on='DATE', how='left')

# Paramétrage suivant le choix effectué dans la Sidebar
if choice == 'Accueil':

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.image("https://github.com/Seb-Dupont-DataAnalyst/Tech_Challenge/blob/main/Image%20Tech%20Challenge.JPG?raw=true")
    




if choice == "1. Analyse descriptive exploratoire (EDA)":

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown('<body class="p">1. Analyse descriptive exploratoire (EDA)</body>', unsafe_allow_html=True)
    st.write("")

    df_sales['DATE'] = df_sales['DATE'].astype('datetime64[ns]')
    df_sales['MONTH'] = df_sales['DATE'].apply(lambda x: x.month)
    df_sales['day-of-week'] = df_sales['DATE'].dt.dayofweek
    
    
    st.write("- Aperçu des données du dataset fourni :")
    st.write(df_sales.head())

    st.write("")
    st.write("- Aucune donnée manquante, 260 jours d'activité répartis sur 2 produits soit 520 lignes dans la base de données :")
    st.image("https://github.com/Seb-Dupont-DataAnalyst/Tech_Challenge/blob/main/Image%20Tech%20Challenge%20df_info.JPG?raw=true")
    

    st.write("")
    st.write("- Affichage des différentes données statistiques issues du dataset :")
    st.write(df_sales.describe())

    st.write("")
    st.write("")
    st.write("- Passons maintenant aux représentations graphiques :")

    fig1 = px.box(df_sales, color="ITEM",y="SALES")
    fig1.update_layout(
    title={
            'text': "Répartition des données par produit",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'})
    fig1.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig1.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig1.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig1.update_layout(xaxis_title = "Produits", yaxis_title = "Ventes")
    fig1.update_layout(width=1400,height=650)
    st.write(fig1)
    st.markdown('<body class="p4">On remarque que les ventes du produit B sont plus importantes que celle du produit A. La médiane se situe autour de 115 produits vendus par jour contre 67.</body>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    fig3 = px.area(df_sales, x="DATE",y="SALES", color="ITEM")
    fig3.update_layout(
    title={
            'text': "Evolution des ventes sur l'année 2019",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'})
    fig3.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig3.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig3.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black') 
    fig3.update_layout(xaxis_title = "Mois", yaxis_title = "Ventes")
    fig3.update_layout(width=1400,height=650)
    st.write(fig3)

    st.write("")
    st.write("")

    fig2 = px.box(df_sales, x="MONTH",y="SALES", color="ITEM")
    fig2.update_layout(
        title={
            'text': "Saisonnalité des ventes par item",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'})
    fig2.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig2.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig2.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black') 
    fig2.update_layout(xaxis_title = "Mois", yaxis_title = "Ventes")
    fig2.update_layout(width=1400,height=650)
    st.write(fig2)
    
    fig10 = px.box(df_sales, x="day-of-week",y="SALES", color="ITEM")
    fig10.update_layout(
    title={
        'text': "Saisonnalité des ventes par item",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'bottom'})
    fig10.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig10.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig10.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black') 
    fig10.update_layout(xaxis_title = "Jours", yaxis_title = "Ventes")
    fig10.show()
    
    
    st.markdown("<body class='p4'>La saisonnalité des 2 produits n'est pas la même. On a un produit A qui se vend plus l'hiver (de novembre à janvier) et un produit B qui se vend beaucoup plus l'été (juin à septembre)</body>", unsafe_allow_html=True)

elif choice == '2. Trouver la boutique correspondante':
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.markdown('<body class="p">2. Trouver la boutique correspondante</body>', unsafe_allow_html=True)
    st.title('')
    st.write('')
    st.write('')
    st.write('Pour retrouver la boutique qui correspond à cet historique de ventes, nous allons nous intéresser aux coefficients de corrélation.')
    st.write('Les ventes étant liées à la météo, on devrait pouvoir isoler des valeurs plus fortes (ou plus faibles) dans une ville par rapport aux autres.')
    st.write("On s'intéresse ici à la première ligne (ou la première colonne) qui fait ressortir les variables les plus fortement corrélés avec le produit A (tableau de gauche) et avec le produit B (tableau de droite)")
    st.write('')
    st.write('')
    

    st.markdown("<p class = 'p3'>Bordeaux :</p>", unsafe_allow_html=True)
    
    col1, col2 = st.beta_columns(2)
    

    with col1:

        pmatrix = df_sales_bordeaux.corr().nlargest(6, columns="A")["A"].index

        coeffc=np.corrcoef(df_sales_bordeaux[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        sns.set(rc={'figure.facecolor':'white'})
        st.write('Produit A :')
        st.write(fig)
    
    with col2:

        pmatrix2 = df_sales_bordeaux.corr().nlargest(6, columns="B")["B"].index

        coeffc2=np.corrcoef(df_sales_bordeaux[pmatrix2].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc2, annot=True, yticklabels=pmatrix2.values, xticklabels=pmatrix2.values, vmin=0, vmax=1, cmap="YlGnBu")
        sns.set(rc={'figure.facecolor':'white'})
        st.write('Produit B :')
        st.write(fig)
        

    st.write('')
    st.write('')
    st.write('')
    st.markdown("<p class = 'p3'>Lyon :</p>", unsafe_allow_html=True)

    col1, col2 = st.beta_columns(2)
    with col1:
        pmatrix = df_sales_lyon.corr().nlargest(6, columns="A")["A"].index

        coeffc=np.corrcoef(df_sales_lyon[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write('Produit A :')
        st.write(fig)

    with col2:
        pmatrix = df_sales_lyon.corr().nlargest(6, columns="B")["B"].index

        coeffc=np.corrcoef(df_sales_lyon[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu") 
        st.write('Produit B :')
        st.write(fig)

    st.write('')
    st.write('')
    st.write('')

    st.markdown("<p class = 'p3'>Lille :</p>", unsafe_allow_html=True)

    col1, col2 = st.beta_columns(2)
    with col1:
        pmatrix = df_sales_lille.corr().nlargest(6, columns="A")["A"].index

        coeffc=np.corrcoef(df_sales_lille[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write('Produit A :')
        st.write(fig)

    with col2:
        pmatrix = df_sales_lille.corr().nlargest(6, columns="B")["B"].index

        coeffc=np.corrcoef(df_sales_lille[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write('Produit B :')
        st.write(fig)

    st.write('')
    st.write('')
    st.write('')

    st.markdown("<p class = 'p3'>Marseille :</p>", unsafe_allow_html=True)
    st.write('')
    st.write('')
    col1, col2 = st.beta_columns(2)
    with col1:
        pmatrix = df_sales_marseille.corr().nlargest(6, columns="A")["A"].index

        coeffc=np.corrcoef(df_sales_marseille[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write('Produit A :')
        st.write(fig)

    with col2:
        pmatrix = df_sales_marseille.corr().nlargest(6, columns="B")["B"].index

        coeffc=np.corrcoef(df_sales_marseille[pmatrix].values.T)

        fig, axes = plt.subplots(figsize=(12, 5))
        sns.heatmap(coeffc, annot=True, yticklabels=pmatrix.values, xticklabels=pmatrix.values, vmin=0, vmax=1, cmap="YlGnBu")
        st.write('Produit B :')
        st.write(fig)

    st.write('')
    st.write('')
    st.write('')
    st.write("A la lecture de ces tableaux, on peut en déduire que la ville concernée est Bordeaux.")
    st.write("En effet, on remarque une nette différence pour le produit A avec les corrélations des autres boutiques pour ce produit.")


elif choice == "3. Explication de l'impact de la météo":
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<body class='p'>3. Explication de l'impact de la météo</body>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("Voyons maintenant le poids de chaque critère météo dans l'évolution des ventes : :thermometer:")
    st.write("")

    st.markdown("<body class='p3'>Produit A</body>", unsafe_allow_html=True)
    st.write('')
    st.write("")
    st.write("")
    col1, col2 = st.beta_columns(2)
    with col1:

        correlation_mat = df_sales_bordeaux.corr()

        corr_pairs = correlation_mat.unstack()['A'].sort_values(ascending=False).iloc[1:10]
        st.write('Plus fortes corrélations positives :')
        st.write(corr_pairs)
        st.write("Les principales variables impactant positivement les ventes du produit A sont donc les précipitations, le code de température de l'après-midi et le taux de couverture nuageuse.")
    st.write('')

    with col2:

        correlation_mat = df_sales_bordeaux.corr()

        corr_pairs = correlation_mat.unstack()['A'].sort_values(ascending=True).iloc[1:10]
        st.write('Plus fortes corrélations négatives :')
        st.write(corr_pairs)
        st.write("Les principales variables impactant négativement les ventes du produit A sont donc la durée d'ensoleillement, l'index de chaleur maximale et la température de l'après-midi.")
        

    st.write('')
    st.write("")

    st.markdown("<body class='p3'>Produit B</body>", unsafe_allow_html=True)
    st.write('')
    st.write("")
    st.write("")
    col1, col2 = st.beta_columns(2)
    with col1:
        correlation_mat2 = df_sales_bordeaux.corr()

        corr_pairs2 = correlation_mat2.unstack()['B'].sort_values(ascending=False).iloc[1:10]
        st.write('Plus fortes corrélations positives :')
        st.write(corr_pairs2)
        st.write("Les principales variables impactant positivement les ventes du produit B sont donc la température de l'après-midi, l'index de chaleur maximale et la température maximale.")
    with col2:
        correlation_mat2 = df_sales_bordeaux.corr()

        corr_pairs2 = correlation_mat2.unstack()['B'].sort_values(ascending=True).iloc[1:10]
        st.write('Plus fortes corrélations négatives :')
        st.write(corr_pairs2)
        st.write("Les principales variables impactant négativement les ventes du produit B sont donc les précipitations, le code de température de l'après-midi et le taux de couverture nuageuse.")

    st.write("")
    st.write("")
    st.markdown("<body class='p3'>Conclusion</body>", unsafe_allow_html=True)
    st.write("")
    st.write("")    
    st.write("Contrairement à ce que pense Bernardo, le vent n'est pas la variable qui impacte le plus les ventes.")
    st.write("La vitesse du vent a bien un impact sur les ventes des produits A et B mais ce n'est pas l'élément qui joue le plus.")
    st.write("La température du vent impacte les ventes des 2 produits également mais là aussi, ce n'est pas l'élément qui joue le plus.")
   

elif choice == "4. Prévisions de ventes":
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<body class='p'>4. Prévisions de ventes</body>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")
    st.write("Pour affiner nos prévisions de ventes, nous allons nous appuyer sur un modèle de machine learning.")
    st.write("Toutes les données fournies sont importantes et participent à améliorer la précision d'un modèle.")
    st.write("Ici nous en sommes dans une situation délicate car les données fournies dans le dataset de prévision ne sont pas assez nombreuses et les variables les plus corrélées sont absentes.")
    st.write("Nous allons donc comparer plusieurs modèles pour choisir celui qui est le plus proche des données réelles lorsqu'il est appliqué sur l'année 2019")
    st.write("On retient 2 indicateurs pour cette comparaison : le Mean Absolute Error (MAE) qui est la moyenne des valeurs absolues des écarts avec le réel et le Root Mean Squared Error (RMSE) qui est la racine carrée de la moyenne des racines carrées des écarts.")
    st.write("")

    
    y = df_sales_bordeaux['A']

    X = df_sales_bordeaux[['PRECIP_TOTAL_DAY_MM',	'CLOUDCOVER_AVG_PERCENT', 'HUMIDITY_MAX_PERCENT', 'MAX_TEMPERATURE_C',	'MIN_TEMPERATURE_C',	'WINDSPEED_MAX_KMH',	'VISIBILITY_AVG_KM',	'PRESSURE_MAX_MB']]
    #X = df_sales_bordeaux.drop(['DATE', 'OPINION', 'B'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.7)


    model_LogReg = LogisticRegression().fit(X_train,y_train)


    modelKNN = KNeighborsRegressor().fit(X_train, y_train)


    modelDTR = DecisionTreeRegressor().fit(X_train, y_train)


    model = LinearRegression().fit(X_train,y_train)


    model_lasso = LassoLarsCV().fit(X_train,y_train)


    model_XGB = XGBRegressor().fit(X_train,y_train)


    model_gbr = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=.1, loss="ls").fit(X_train, y_train)


    y2 = df_sales_bordeaux['B']

    X2 = df_sales_bordeaux[['PRECIP_TOTAL_DAY_MM',	'CLOUDCOVER_AVG_PERCENT', 'HUMIDITY_MAX_PERCENT', 'MAX_TEMPERATURE_C',	'MIN_TEMPERATURE_C',	'WINDSPEED_MAX_KMH',	'VISIBILITY_AVG_KM',	'PRESSURE_MAX_MB']]
    #X = df_sales_bordeaux.drop(['DATE', 'OPINION', 'B'], axis=1)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state = 42, train_size = 0.7)

    model2 = LinearRegression().fit(X2_train,y2_train)

    modelKNN2 = KNeighborsRegressor().fit(X2_train, y2_train)

    model_LogReg2 = LogisticRegression().fit(X2_train,y2_train)

    model_lasso2 = LassoLarsCV().fit(X2_train,y2_train)

    model_XGB2 = XGBRegressor().fit(X2_train,y2_train)

    model_gbr2 = ensemble.GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=.1, loss="ls").fit(X2_train, y2_train)


    dict_MAE_A = {'KNN' : metrics.mean_absolute_error(y, modelKNN.predict(X)),'Linear Regression' : metrics.mean_absolute_error(y, model.predict(X)),
                'Logistic Regression' : metrics.mean_absolute_error(y, model_LogReg.predict(X)), 'Gradient Boosting': metrics.mean_absolute_error(y, model_gbr.predict(X)), 
                'XGBoost' : metrics.mean_absolute_error(y, model_XGB.predict(X)), 'Lasso' : metrics.mean_absolute_error(y, model_lasso.predict(X))}

    dict_MAE_B = {'KNN': metrics.mean_absolute_error(y2, modelKNN2.predict(X2)),'Linear Regression' : metrics.mean_absolute_error(y2, model2.predict(X2)), 
                'Logistic Regression' : metrics.mean_absolute_error(y2, model_LogReg2.predict(X2)), 'Gradient Boosting': metrics.mean_absolute_error(y2, model_gbr2.predict(X2)), 
                'XGBoost' : metrics.mean_absolute_error(y2, model_XGB2.predict(X2)), 'Lasso' : metrics.mean_absolute_error(y2, model_lasso2.predict(X2))}

    dict_RMSE_A = {'KNN' : np.sqrt(metrics.mean_squared_error(y, modelKNN.predict(X))), 'Linear Regression' : np.sqrt(metrics.mean_squared_error(y, model.predict(X))),
                'Logistic Regression' : np.sqrt(metrics.mean_squared_error(y, model_LogReg.predict(X))), 'Gradient Boosting': np.sqrt(metrics.mean_squared_error(y, model_gbr.predict(X))), 
                'XGBoost' : np.sqrt(metrics.mean_squared_error(y, model_XGB.predict(X))), 'Lasso' : np.sqrt(metrics.mean_squared_error(y, model_lasso.predict(X)))}

    dict_RMSE_B = {'KNN' : np.sqrt(metrics.mean_squared_error(y2, modelKNN2.predict(X2))), 'Linear Regression' : np.sqrt(metrics.mean_squared_error(y2, model2.predict(X2))),
                'Logistic Regression' : np.sqrt(metrics.mean_squared_error(y2, model_LogReg2.predict(X2))), 'Gradient Boosting': np.sqrt(metrics.mean_squared_error(y2, model_gbr2.predict(X2))), 
                'XGBoost' : np.sqrt(metrics.mean_squared_error(y2, model_XGB2.predict(X2))), 'Lasso' : np.sqrt(metrics.mean_squared_error(y2, model_lasso2.predict(X2)))}

    values_MAE_A = list(dict_MAE_A.values())
    names_MAE_A = list(dict_MAE_A.keys())

    values_MAE_B = list(dict_MAE_B.values())
    names_MAE_B = list(dict_MAE_B.keys())

    values_RMSE_A = list(dict_RMSE_A.values())
    names_RMSE_A = list(dict_RMSE_A.keys())

    values_RMSE_B = list(dict_RMSE_B.values())
    names_RMSE_B = list(dict_RMSE_B.keys())

    
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names_MAE_A, y=values_MAE_A,
                        mode='lines+markers',
                        name='Item A'))
    fig.add_trace(go.Scatter(x=names_MAE_B, y=values_MAE_B,
                        mode='lines+markers',
                        name='Item B'))
    fig.update_layout(title='<b>Mean Absolute Error</b>' , title_x=0.5)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_layout(xaxis_title = "Modèles", yaxis_title = "Valeurs d'erreur")
    fig.update_layout(width=1300,height=550)
    st.write(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=names_RMSE_A, y=values_RMSE_A,
                        mode='lines+markers',
                        name='Item A'))
    fig.add_trace(go.Scatter(x=names_RMSE_B, y=values_RMSE_B,
                        mode='lines+markers',
                        name='Item B'))
    fig.update_layout(title='<b>Root Mean Squared Error</b>', title_x=0.5)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_layout(xaxis_title = "Modèles", yaxis_title = "Valeurs d'erreur")
    fig.update_layout(width=1300,height=550)
    st.write(fig)

    st.write("On retient le modèle GradientBoosting qui apparait comme l'un des meilleurs modèles pour cet exercice")
    st.write('')
    st.write('')
    st.write('Voyons maintenant les prévisions de ventes calculées :')


    df_predict = pd.read_csv("https://raw.githubusercontent.com/murpi/wilddata/master/test/forecast.csv")

    X = df_predict[['PRECIP_TOTAL_DAY_MM',	'CLOUDCOVER_AVG_PERCENT', 'HUMIDITY_MAX_PERCENT', 'MAX_TEMPERATURE_C',	'MIN_TEMPERATURE_C',	'WINDSPEED_MAX_KMH',	'VISIBILITY_AVG_KM',	'PRESSURE_MAX_MB']]
    df_predict['predictions A'] = model_gbr.predict(X)
    df_predict['predictions B'] = model_gbr2.predict(X)

    df_predict['predictions A'] = df_predict['predictions A'].round(decimals=0)
    df_predict['predictions B'] = df_predict['predictions B'].round(decimals=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_predict['DATE'], y=df_predict['predictions A'], name = 'Produit A', text=df_predict['predictions A'], textposition='auto', textfont=dict(color="white")))
    fig.add_trace(go.Bar(x=df_predict['DATE'], y=df_predict['predictions B'], name = 'Produit B', text=df_predict['predictions B'], textposition='auto', textfont=dict(color="white")))
    fig.update_layout(title='<b>Prédictions des ventes pour la semaine à venir</b>', title_x=0.5)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)',
                       'paper_bgcolor': 'rgba(0,0,0,0)', })
    fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='black')
    fig.update_layout(xaxis_title = "Jours", yaxis_title = "Ventes")
    fig.update_layout(width=1400,height=650)
    st.write(fig)

    st.write('Total Ventes A :' , df_predict['predictions A'].sum())
    st.write('Total Ventes B :', df_predict['predictions B'].sum())
