#### To run the app: ####

# pip install streamlit
# to run file locally, cd into folder where file and data are located and execute 'streamlit run app_dbscan.py' in command line
# make sure data file is in the same folder as this file

#########################

import streamlit as st
import numpy as np
import pandas as pd
# Sklearn imports
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'
# Plots
import plotly.graph_objects as go
import matplotlib.cm as cm


# Main sections for app
header = st.container()
background = st.container()
data_model = st.container()

# Cache data load
@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(columns=data.columns[0], axis=1)
    return data


@st.cache_data
def dbscan_model(epsilon, min_sample,df):

    # Fit DBSCAN with eps that resulted in max silhouette score
    dbscan = DBSCAN(eps = epsilon, min_samples = min_sample)
    dbscan.fit(df)
    df['Cluster']=dbscan.labels_
    

    # Calculate metrics
    labels = dbscan.labels_

    #Calculating "the number of clusters"
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    

    #Computing "the Silhouette Score"
    sil_score = metrics.silhouette_score(df, labels)

    return df, n_clusters_, sil_score


@st.cache_data
def radar_chart(df):

    result = result_df.groupby(['Cluster']).mean()

    categories = result.columns
    fig = go.Figure()

    for g in result.index:
        fig.add_trace(go.Scatterpolar(
            r = result.loc[g].values,
            theta = categories,
            fill = 'toself',
            name = f'cluster #{g}'
        ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[-5, 5] # here we can define the range
        )),
    showlegend=True,
        title="Radar Plot",
        title_x=0.5
    )

    fig.update_layout(
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",
    )

    return fig

with header:
    st.title('Multivariate Clustering with GHG CO2 Emissions and Socioeconomic Data')

with data_model:
    # Load data
    data = load_data('sample_df_2018.csv')

    st.header('Preview of data')
    st.dataframe(data.head(5))

    st.header('DBSCAN Model')
    st.write('Select input below to run model')

    
    # User Input
    col_one, col_two = st.columns(2) 
    eps = col_one.selectbox('Epsilon Value', options=[0.25, 0.30, 0.45, 0.5])
    min_sample = col_two.selectbox('Min Sample Value', options=[5, 6, 7, 8, 9, 10, 11, 12])

    if st.button('Run model'):
    # Run DBSCAN model
        result_df, result_n_cluster, result_sil_score = dbscan_model(eps, min_sample, data)

        st.write('Preview of resulting dataframe')
        st.dataframe(result_df.head(5))
        st.write("Resulting number of clusters: ")
        st.write(result_n_cluster)
        st.write("Resulting silhouette score: ")
        st.write(result_sil_score)

        
        # Radar chart

        st.plotly_chart(radar_chart(result_df), use_container_width=True)

    else: 
        st.write('Select epsilon and min_sample value to get started')


    