#### To run the app: ####

# pip install streamlit
# to run file locally, cd into folder where file and data are located and execute 'streamlit run test_app.py' in command line
# make sure data file is in the same folder as this file

#########################

import streamlit as st
import pandas as pd
import numpy as np
# Sklearn imports

# Plots
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.colors


# Main sections for app
header = st.container()
header2 = st.container()
header3 = st.countainer()
background = st.container()
kmeans_heat_map = st.container()
kmeans_plots = st.container()
fdr_heat_map = st.container()
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
    cluster_colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange']

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
def kmeans_map_by_cluster(df, dropdown_values):
    """
    :param df: dataframe resulting from k-means analysis on all vars, all years, and 6yr pct change var, and cluster labels column
    :param dropdown_values: list of column names to be used in the dropdown menu
    :return: plotly figure object that:
                plots all vars vs avg_co2
                has dropdown menu for variable
                shades points by cluster
    """
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Define cluster colors
    cluster_colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange']

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
            color=df['Cluster'],  # Shades points by cluster value
            colorscale=[(i, cluster_colors[i]) for i in range(len(cluster_colors))],  # Use the specified cluster colors
            showscale=False,  # Disable the colorscale legend
        ),
        hovertemplate='<b>Location:</b> %{text}<br>' +
                      '<b>Cluster:</b> %{marker.color}<br>',
        text=df['LOCATION']
    )

    fig.add_trace(scatter)

    # Add custom legend for clusters
    legend_items = [
        go.Scattergeo(
            lat=[],
            lon=[],
            mode='markers',
            marker=dict(
                size=10,
                color=cluster_colors[i],
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hoverinfo='skip',
            showlegend=True,
            name=f'Cluster {i}',
        )
        for i in range(len(cluster_colors))
    ]

    fig.add_traces(legend_items)

    # Define the dropdown menu options for the variable
    dropdown_options = []
    for col in dropdown_columns:
        if col == 'avg_co2':
            dropdown_options.append(
                {
                    'label': col,
                    'method': 'update',
                    'args': [
                        {'marker': {'colorscale': [(i, cluster_colors[i]) for i in range(len(cluster_colors))], 'color': df['Cluster']}},
                        {'marker': {'colorscale': [(i, cluster_colors[i]) for i in range(len(cluster_colors))], 'color': df['Cluster']}}
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

    # Set plot layout with legend
    fig.update_layout(
        title='US Socioeconomic Vulnerability Variables and Atmospheric CO2 Concentration 2014-2020',
        template='plotly_dark',  # Set the dark color scheme
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
        ),
        showlegend=True,
        legend=dict(
            title='Clusters',
            itemsizing='constant',
            bgcolor='rgba(0,0,0,0)',
            xanchor='right',
            yanchor='top',
            x=0.98,
            y=0.98,
            bordercolor='black',
            borderwidth=1
        )
    )

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
    cluster_colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange']

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
def fdr_map_by_cluster(df, dropdown_values):
    """
    :param df: dataframe resulting from k-means analysis on all vars, all years, and 6yr pct change var, and cluster labels column
    :param dropdown_values: list of column names to be used in the dropdown menu
    :return: plotly figure object that:
                plots all vars vs avg_co2
                has dropdown menu for variable
                shades points by cluster
    """
    # Specify the columns for the dropdown menu
    dropdown_columns = dropdown_values

    # Define cluster colors
    cluster_colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange']
    cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

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
            color=df['Cluster'],  # Shades points by cluster value
            colorscale=cluster_colors,  # Use the specified cluster colors
            showscale=False,  # Disable the colorscale legend
        ),
        hovertemplate='<b>Location:</b> %{text}<br>' +
                      '<b>Cluster:</b> %{marker.color}<br>',
        text=df['LOCATION']
    )

    fig.add_trace(scatter)

    # Add custom legend for clusters
    legend_items = [
        go.Scattergeo(
            lat=[],
            lon=[],
            mode='markers',
            marker=dict(
                size=10,
                color=cluster_colors[i],
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hoverinfo='skip',
            showlegend=True,
            name=cluster_labels[i],
        )
        for i in range(len(cluster_colors))
    ]

    fig.add_traces(legend_items)

    # Define the dropdown menu options for the variable
    dropdown_options = []
    for col in dropdown_columns:
        if col == 'avg_co2':
            dropdown_options.append(
                {
                    'label': col,
                    'method': 'update',
                    'args': [
                        {'marker': {'colorscale': cluster_colors, 'color': df['Cluster']}},
                        {'marker': {'colorscale': cluster_colors, 'color': df['Cluster']}}
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

    # Set plot layout with legend
    fig.update_layout(  # Set the dark color scheme
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
        ),
        showlegend=True,
        legend=dict(
            title='Clusters',
            itemsizing='constant',
            bgcolor='rgba(0,0,0,0)',
            xanchor='right',
            yanchor='top',
            x=0.98,
            y=0.98,
            bordercolor='black',
            borderwidth=1
        )
    )

    return fig



with header:
    st.title('Multivariate Clustering with US GHG CO2 Emissions Socioeconomic Vulnerability Variables 2014-2020')
    
with header2:
    st.title('K-Means Clustering on All Variables')
    st.subtitle('K-Means clustering on all variables and one total percent change over six-year period. This model performed best on these variables with a silhouette score of 0.2398')

with kmeans_heat_map:
    data = load_data('kmeans_tot_pct_centroids.csv')
    kmeans_heatmap_vars = ['avg_co2', 'total_population', 'housing_units', 'num_households', 'unemployment',
                           'socioeconomic', 'household_comp', 'minority_status', 'housing_type',
                           'overall_svi', 'xco2_std', 'co2_6yr_pct_change']
    st.header('Heatmap of Cluster Centroids')
    st.subheader('K-Means clustering k=6')
    st.plotly_chart(kmeans_heatmap(data, kmeans_heatmap_vars), use_container_width=True)

with kmeans_plots:
    data = load_data('kmeans_sample.csv')
    kmeans_vars = ['xco2_std', 'total_population', 'housing_units', 'num_households',
                   'unemployment', 'socioeconomic', 'household_comp',
                   'minority_status', 'housing_type', 'overall_svi', 'co2_6yr_pct_change']
    st.header('All Variables vs CO2 by cluster')
    st.subheader('K-Means Clustering k=6')
    st.plotly_chart(kmeans_scatter(data, kmeans_vars), use_container_width=True)

    st.header('Map by Value')
    st.subheader('K-Means Clustering k=6')
    st.plotly_chart(kmeans_map_by_value(data, kmeans_vars), use_container_width=True)

    #st.header('Map by Cluster')
    #st.subheader('K-Means Clustering k=6')
    #st.plotly_chart(kmeans_map_by_cluster(data, kmeans_vars), use_container_width=True)

with header3:
    st.title('K-Means Clustering on Functionally Reduced Variables')
    st.subtitle('K-Means clustering on variables reduced to slopes and means by linear regression. This model performed best on these variables with a silhouette score of 0.3142')
    
with fdr_heat_map:
    data_fdr = load_data('kmeans_fdr_centroids.csv')
    fdr_heatmap_vars = ['Mean_avg_co2', 'Mean_total_population', 'Mean_housing_units',
                        'Mean_num_households', 'Mean_unemployment', 'Mean_socioeconomic',
                        'Mean_household_comp', 'Mean_minority_status', 'Mean_housing_type',
                        'Mean_overall_svi', 'Mean_xco2_std', 'Slope_avg_co2',
                        'Slope_total_population', 'Slope_housing_units', 'Slope_num_households',
                        'Slope_unemployment', 'Slope_socioeconomic', 'Slope_household_comp',
                        'Slope_minority_status', 'Slope_housing_type', 'Slope_overall_svi',
                        'Slope_xco2_std']
    st.header('Heatmap of Cluster Centroids')
    st.subheader('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(kmeans_heatmap(data_fdr, fdr_heatmap_vars), use_container_width=True)

with fdr_plots:
    data_fdr = load_data('fdr_sample.csv')
    fdr_map_vars = ['Mean_avg_co2', 'Mean_total_population', 'Mean_housing_units',
                    'Mean_num_households', 'Mean_unemployment', 'Mean_socioeconomic',
                    'Mean_household_comp', 'Mean_minority_status', 'Mean_housing_type',
                    'Mean_overall_svi', 'Mean_xco2_std', 'Slope_avg_co2',
                    'Slope_total_population', 'Slope_housing_units', 'Slope_num_households',
                    'Slope_unemployment', 'Slope_socioeconomic', 'Slope_household_comp',
                    'Slope_minority_status', 'Slope_housing_type', 'Slope_overall_svi',
                    'Slope_xco2_std']
    st.header('Slopes and Means for all Variables vs. Slope of CO2  by Cluster')
    st.subheader('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(fdr_scatter(data_fdr, fdr_map_vars), use_container_width=True)

    st.header('Map of Slopes and Means for all Variables by Value')
    st.subheader('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(fdr_map_by_value(data_fdr, fdr_map_vars), use_container_width=True)

    st.header('Map of Slopes and Means for all Variables by Cluster')
    st.subheader('K-Means Clustering of functionally reduced variables, k=6')
    st.plotly_chart(fdr_map_by_cluster(data_fdr, fdr_map_vars), use_container_width=True)




