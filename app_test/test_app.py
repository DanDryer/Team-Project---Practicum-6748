#### To run the app: ####

# pip install streamlit
# to run file locally, cd into folder where file and data are located and execute 'streamlit run test_app.py' in command line
# make sure data file is in the same folder as this file

#########################

import streamlit as st
import pandas as pd
import numpy as np

# Plots
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import chart_studio.plotly as pyo


# Main sections for app
header = st.container()
background = st.container()
kmeans_plots = st.container()
fdr_plots = st.container()


# Cache data load
@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    data = data.drop(columns=data.columns[0], axis=1)
    return data

@st.cache_data
def kmeans_heatmap(df, value_vars):

    variables = value_vars

    values = df[variables].values

    # Define the minimum and maximum values for the colorscale
    zmin = np.nanmin(values)
    zmax = np.nanmax(values)

    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=variables,
        y=df['Cluster'],
        colorscale='Viridis',
        zmin=zmin,
        zmax=zmax,
        text=np.around(values, decimals=4),
        hovertemplate='Variable: %{x}<br>Cluster: %{y}<br>Value: %{text}'
    ))

    # Set the text font size and color for the heatmap squares
    fig.update_traces(textfont=dict(size=12, color='black'))

    # Set the heatmap layout
    fig.update_layout(
        xaxis=dict(
            title='Variables',
            tickangle=270,
            tickfont=dict(size=10),
            automargin=True
        ),
        yaxis=dict(title='Cluster')
    )

    return fig

@st.cache_data
def kmeans_scatter(df,dropdown_values):
    """
    :param df: dataframe resulting from k-means analysis on all vars, all years, and 6yr pct change var, and cluster labels column
    :param dropdown_values: list of column names to be used in the dropdown menu
    :return: plotly figure object that:
                plots all vars vs avg_co2
                has dropdown menu for y column
                slider bar for year filter
                shades points by cluster
    """
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Define colors for each cluster
    cluster_colors = ['#fde725', '#7ad151', "#22a884", "#2a788e", "#414487", "#440154"]

    # Create the scatter plot figure
    fig = make_subplots(rows=1, cols=1)

    # Define initial y column
    y_column = 'avg_co2'
    cluster_column = 'Cluster'  # Assuming the column name for cluster labels is 'Cluster'

    # Create an empty list to store the data for each cluster
    cluster_data_list = []

    # Add initial scatter plot with fixed x as 'avg_co2' and default y column
    traces = []
    for cluster in range(6):
        cluster_data = df[df[cluster_column] == cluster]
        cluster_data_list.append(cluster_data)  # Append cluster data to the list

        trace = go.Scatter(
            x=cluster_data['avg_co2'],
            y=cluster_data[y_column],
            mode='markers',
            marker=dict(
                size=10,
                color=cluster_colors[cluster],  # Assign a different color for each cluster
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            text=cluster_data['LOCATION'],
            name=f'Cluster {cluster}'
        )
        traces.append(trace)

    fig.add_traces(traces)

    # Define the dropdown menu options for y column
    dropdown_options_y = []
    for col in dropdown_columns:
        dropdown_options_y.append(
            {
                'label': col,
                'method': 'update',
                'args': [
                    {'y': [df[col]]},
                    {'yaxis': {'title': col}},
                    {'marker.color': [cluster_colors[cluster] for cluster in df[cluster_column]]}
                ]
            }
        )

    # Add dropdown menu for y column
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_options_y,
                direction='down',
                active=0,
                x=0.8,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ]
    )

    # Add slider bar for year filter
    slider_steps = [
        {
            'args': [
                {'y': [cluster_data.loc[cluster_data['year'] == year, y_column].tolist() for cluster_data in cluster_data_list]},
                {'xaxis.title.text': 'avg_co2', 'yaxis.title.text': y_column,
                 'marker.color': [[cluster_colors[cluster] for cluster in cluster_data[cluster_column]] if cluster_data[cluster_column].nunique() > 1 else cluster_colors[0] for cluster_data in cluster_data_list]}
            ],
            'label': str(year),
            'method': 'update'
        }
        for year in sorted(df['year'].unique())  # Sort the years in ascending order
    ]

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Year: "},
                pad={"t": 50},
                steps=slider_steps
            )
        ]
    )

    # Set plot layout with color scheme and legend
    fig.update_layout(
        xaxis=dict(title='avg_co2'),
        yaxis=dict(title=y_column),
        autosize=True,
        showlegend=True,  # Show the legend
        legend=dict(
            title='Cluster',  # Set the legend title
            itemsizing='constant',  # Fix the legend item size
            bgcolor='rgba(0,0,0,0)',  # Set the legend background color
            traceorder='normal'  # Keep the traces in the legend order
        ),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),

    )
    return fig

@st.cache_data
def kmeans_map_by_value(df, dropdown_values):
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Create the scatter point US map
    # Create the scatter point US map
    fig = make_subplots(rows=1, cols=1)

    # Set initial variable for the scatter point US map
    variable = dropdown_columns[0]

    # Define color scaling range for 'avg_co2' variable
    avg_co2_min = df['avg_co2'].min()
    avg_co2_max = df['avg_co2'].max()

    # Add scatter trace to the figure
    scatter = go.Scattergeo(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            sizemode='diameter',
            sizeref=0.1,
            sizemin=1,
            color=df[variable],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=variable)
        ),
        hovertemplate='<b>Location:</b> %{text}<br>' +
                      f'<b>{variable}:</b> %{{marker.color}}',
        text=df['LOCATION']
    )

    fig.add_trace(scatter)

    # Define the dropdown menu options for the variable
    dropdown_options = []
    for col in dropdown_columns:
        if col == 'avg_co2':
            dropdown_options.append(
                {
                    'label': col,
                    'method': 'update',
                    'args': [
                        {'marker': {'colorscale': 'Viridis', 'color': df[col],
                                    'cmin': avg_co2_min, 'cmax': avg_co2_max,
                                    'colorbar': {'title': col, 'cmin': avg_co2_min, 'cmax': avg_co2_max}}},
                        {'marker': {'colorscale': 'Viridis', 'color': df[col],
                                    'cmin': avg_co2_min, 'cmax': avg_co2_max,
                                    'colorbar': {'title': col, 'cmin': avg_co2_min, 'cmax': avg_co2_max}}}
                    ]
                }
            )
        else:
            cmin = df[col].min()
            cmax = df[col].max()

            dropdown_options.append(
                {
                    'label': col,
                    'method': 'update',
                    'args': [
                        {'marker': {'colorscale': 'Viridis', 'color': df[col], 'colorbar': {'title': col, 'cmin': cmin, 'cmax': cmax}}},
                        {'marker': {'colorscale': 'Viridis', 'color': df[col], 'colorbar': {'title': col, 'cmin': cmin, 'cmax': cmax}}}
                    ]
                }
            )

    # Add dropdown menu for the variable
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_options,
                direction='down',
                active=0,
                x=0.8,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ]
    )

    # Set plot layout with dark color scheme
    fig.update_layout(
        geo=dict(
            scope='usa',
            showland=True,
            landcolor='lightgray',
            showlakes=True,
            lakecolor='white',
            showocean=True,
            oceancolor='aliceblue',
            showcountries=True,
            countrycolor='gray',
            projection_type='albers usa'
        )
    )
    # Add slider bar for year filter
    slider = dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=[]
    )

    avg_co2_min = 394
    avg_co2_max = 414

    for year in sorted(df['year'].unique()):
        step = {
            "label": str(year),
            "method": "update",
            "args": [
                {
                    "marker": {
                        "size": 15,
                        "sizemode": "diameter",
                        "sizeref": 0.1,
                        "sizemin": 1,
                        "color": [df[df['year'] == year][variable].tolist()[0]],
                        "colorscale": "Viridis",
                        "colorbar": {"title": variable, "cmin": avg_co2_min, "cmax": avg_co2_max},
                        "cmin": avg_co2_min,
                        "cmax": avg_co2_max
                    }
                },
                {"marker.colorbar.title": variable},
            ]
        }
        if variable == 'avg_co2':
            step['args'][0]['marker']['color'] = df[df['year'] == year][variable].tolist()
        slider['steps'].append(step)

    fig.update_layout(sliders=[slider])

    return fig

@st.cache_data
def fdr_scatter(df, dropdown_values):
    """
    :param df: dataframe resulting from k-means analysis on all vars, all years, and 6yr pct change var, and cluster labels column
    :param dropdown_values: list of column names to be used in the dropdown menu
    :return: plotly figure object that:
                plots all vars vs avg_co2
                has dropdown menu for y column
                shades points by cluster
    """
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Define colors for each cluster
    cluster_colors = ['#fde725', '#7ad151', "#22a884", "#2a788e", "#414487", "#440154"]

    # Create the scatter plot figure
    fig = make_subplots(rows=1, cols=1)

    # Define initial y column
    y_column = 'Mean_avg_co2'
    cluster_column = 'Cluster'  # Assuming the column name for cluster labels is 'Cluster'

    # Create an empty list to store the data for each cluster
    cluster_data_list = []

    # Add initial scatter plot with fixed x as 'avg_co2' and default y column
    traces = []
    for cluster in range(6):
        cluster_data = df[df[cluster_column] == cluster]
        cluster_data_list.append(cluster_data)  # Append cluster data to the list

        trace = go.Scatter(
            x=cluster_data['Mean_avg_co2'],
            y=cluster_data[y_column],
            mode='markers',
            marker=dict(
                size=10,
                color=cluster_colors[cluster],  # Assign a different color for each cluster
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            text=cluster_data['LOCATION'],
            name=f'Cluster {cluster}'
        )
        traces.append(trace)

    fig.add_traces(traces)

    # Define the dropdown menu options for y column
    dropdown_options_y = []
    for col in dropdown_columns:
        dropdown_options_y.append(
            {
                'label': col,
                'method': 'update',
                'args': [
                    {'y': [df[col]]},
                    {'yaxis': {'title': col}},
                    {'marker.color': [cluster_colors[cluster] for cluster in df[cluster_column]]}
                ]
            }
        )

    # Add dropdown menu for y column
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_options_y,
                direction='down',

                active=0,
                x=0.8,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ]
    )

    # Set plot layout with color scheme and legend
    fig.update_layout(
        xaxis=dict(title='avg_co2'),
        yaxis=dict(title=y_column),
        autosize=True,
        showlegend=True,  # Show the legend
        legend=dict(
            title='Cluster',  # Set the legend title
            itemsizing='constant',  # Fix the legend item size
            bgcolor='rgba(0,0,0,0)',  # Set the legend background color
            traceorder='normal'  # Keep the traces in the legend order
        ),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),

    )
    return fig

@st.cache_data
def fdr_map_by_value(df, dropdown_values):
    """
    :param df: dataframe resulting from k-means analysis on all vars, all years, and 6yr pct change var, and cluster labels column
    :param dropdown_values: list of column names to be used in the dropdown menu
    :return: plotly figure object that:
                plots all vars vs avg_co2
                has dropdown menu for variable
                shades points by values
    """
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Create the scatter point US map
    fig = make_subplots(rows=1, cols=1)

    # Set initial variable for the scatter point US map
    variable = dropdown_columns[0]

    # Add scatter trace to the figure
    scatter = go.Scattergeo(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            sizemode='diameter',
            sizeref=0.1,
            sizemin=1,
            color=df[variable],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=variable)
        ),
        hovertemplate='<b>Location:</b> %{text}<br>' +
                      f'<b>{variable}:</b> %{{marker.color}}',
        text=df['LOCATION']
    )

    fig.add_trace(scatter)

    # Define the dropdown menu options for the variable
    dropdown_options = []
    for col in dropdown_columns:
        cmin = df[col].min()
        cmax = df[col].max()

        dropdown_options.append(
            {
                'label': col,
                'method': 'update',
                'args': [
                    {'marker': {'colorscale': 'Viridis', 'color': df[col], 'colorbar': {'title': col, 'cmin': cmin, 'cmax': cmax}}},
                    {'marker': {'colorscale': 'Viridis', 'color': df[col], 'colorbar': {'title': col, 'cmin': cmin, 'cmax': cmax}}}
                ]
            }
        )

    # Add dropdown menu for the variable
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_options,
                direction='down',
                active=0,
                x=0.8,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )
        ]
    )

    # Set plot layout with dark color scheme
    fig.update_layout(
        geo=dict(
            scope='usa',
            showland=True,
            landcolor='lightgray',
            showlakes=True,
            lakecolor='white',
            showocean=True,
            oceancolor='aliceblue',
            showcountries=True,
            countrycolor='gray',
            projection_type='albers usa'
        )
    )
    return fig

@st.cache_data
def facet_plot_maps_by_cluster(df):
    cluster_colors = {
        0: '#fde725',
        1: '#7ad151',
        2: '#22a884',
        3: '#2a788e',
        4: '#414487',
        5: '#440154'
    }

    # Get unique values in the "Cluster" column
    clusters = df['Cluster'].unique()

    # Create subplots for each cluster
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Cluster {cluster}' for cluster in clusters], specs=[[{'type': 'scattergeo'}]*3]*2)

    # Set the center coordinates for the map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # Iterate over clusters and create scatter plots
    for i, cluster in enumerate(clusters, 1):
        scatter_geo = go.Scattergeo(
            lat=df[df['Cluster'] == cluster]['latitude'],
            lon=df[df['Cluster'] == cluster]['longitude'],
            mode='markers',
            marker=dict(
                size=3,
                color=cluster_colors[cluster],
                colorscale='Viridis',
                showscale=False
            ),
            name=f'Cluster {cluster}'

        )

        fig.add_trace(scatter_geo, row=(i-1) // 3 + 1, col=(i-1) % 3 + 1)

    fig.update_geos(
        projection_type="albers usa",
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='aliceblue',
        showlakes=True,
        lakecolor='white',
        showcountries=True,
        countrycolor='gray'
    )

    fig.update_layout(
        height=600,
        width=800,
        showlegend=True,
        legend=dict(
            title='Cluster',
            itemsizing='constant',
            bgcolor='rgba(0,0,0,0)',
            traceorder='normal'
        )
    )

    return fig

with header:
    st.title('Multivariate Cluster Analysis of US Atmospheric CO2 Concentrations and Socioeconomic Vulnerability Variables 2014-2020')
    st.markdown('GE Vernova (Team 2): Dan Dryer Dryer, Daniel C <ddryer3@gatech.edu>, Joanna Rashid ,<joannarashid@gatech.edu>, Nhu Y Pham <npham48@gatech.edu>')
    st.markdown('July 12, 2023')
    st.divider()

with kmeans_plots:
    data = load_data('kmeans_sample.csv')
    data_centroids = load_data('kmeans_tot_pct_centroids.csv')
    kmeans_vars = ['avg_co2', 'total_population', 'housing_units', 'num_households', 'unemployment',
                           'socioeconomic', 'household_comp', 'minority_status', 'housing_type',
                           'overall_svi', 'xco2_std', 'co2_6yr_pct_change']

    st.header('K-Means Clustering on All Variables')
    st.markdown('K-Means clustering on all variables and one total percent change over six-year period. This model performed best on these variables with a silhouette score of 0.2398.')

    st.subheader('Description of Data (original unscaled values for interpretability)')
    st.dataframe(data[kmeans_vars].describe())

    st.subheader('Heatmap of Cluster Centroids')
    st.markdown('K-Means clustering k=6')
    st.markdown('To illustrate the cluster characteristics, this heatmap illustrates the mean scaled value for each variable within a cluster. ')
    st.plotly_chart(kmeans_heatmap(data_centroids, kmeans_vars), use_container_width=True)

    st.subheader('All Variables vs CO2 by cluster')
    st.markdown('K-Means Clustering k=6')
    st.plotly_chart(kmeans_scatter(data, kmeans_vars), use_container_width=True)

    st.subheader('Map by Value')
    st.markdown('K-Means Clustering k=6')
    st.plotly_chart(kmeans_map_by_value(data, kmeans_vars), use_container_width=True)

    st.subheader('Map by Cluster')
    st.markdown('K-Means Clustering k=6')
    st.plotly_chart(facet_plot_maps_by_cluster(data), use_container_width=True)
    st.divider()

with fdr_plots:
    data_fdr_centroids = load_data('kmeans_fdr_centroids.csv')
    data_fdr = load_data('fdr_sample.csv')
    fdr_vars = ['Mean_avg_co2', 'Mean_total_population', 'Mean_housing_units',
                        'Mean_num_households', 'Mean_unemployment', 'Mean_socioeconomic',
                        'Mean_household_comp', 'Mean_minority_status', 'Mean_housing_type',
                        'Mean_overall_svi', 'Mean_xco2_std', 'Slope_avg_co2',
                        'Slope_total_population', 'Slope_housing_units', 'Slope_num_households',
                        'Slope_unemployment', 'Slope_socioeconomic', 'Slope_household_comp',
                        'Slope_minority_status', 'Slope_housing_type', 'Slope_overall_svi',
                        'Slope_xco2_std']

    st.header('K-Means Clustering on Functionally Reduced Variables')
    st.markdown('K-Means clustering on variables reduced to slopes and means by linear regression. This model performed best on these variables with a silhouette score of 0.3142')

    st.subheader('Description of Data (slope and mean values)')
    st.dataframe(data_fdr[fdr_vars].describe())

    st.subheader('Heatmap of Cluster Centroids')
    st.markdown('K-Means Clustering of functionally reduced variables, k=6')
    st.markdown('To illustrate the cluster characteristics, this heatmap illustrates the mean scaled value for each variable within a cluster. ')
    st.plotly_chart(kmeans_heatmap(data_fdr_centroids, fdr_vars), use_container_width=True)

    st.subheader('Slopes and Means for all Variables vs. Slope of CO2  by Cluster')
    st.markdown('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(fdr_scatter(data_fdr, fdr_vars), use_container_width=True)

    st.subheader('Map of Slopes and Means for all Variables by Value')
    st.markdown('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(fdr_map_by_value(data_fdr, fdr_vars), use_container_width=True)

    st.subheader('Map of Slopes and Means for all Variables by Cluster')
    st.markdown('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(facet_plot_maps_by_cluster(data_fdr), use_container_width=True)
    st.divider()
