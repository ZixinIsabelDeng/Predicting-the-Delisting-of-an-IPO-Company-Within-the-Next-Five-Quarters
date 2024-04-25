import dash
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_minmax = pd.read_csv('DT-minmax_scaling.csv')
df_standard = pd.read_csv('DT-standard_scaling.csv')
df_robust = pd.read_csv('DT-robust_scaling.csv')
df_scaled = pd.read_csv('principal_component.csv')
df_normalized = pd.read_csv('DT-normalized.csv')

pca = PCA()
pca.fit(df_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = explained_var.cumsum()
principalComponents = pca.fit_transform(df_scaled)
components_df = pd.DataFrame(principalComponents, columns=[f'PCA{i+1}' for i in range(20)])


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='scale-selector',
        options=[
            {'label': 'Min-Max Scaled Data', 'value': 'minmax'},
            {'label': 'Standard Scaled Data', 'value': 'standard'},
            {'label': 'Robust Scaled Data', 'value': 'robust'},
            {'label': 'Normalized Data', 'value': 'normalized'},
        ],
        value='standard',
        style={'width': '48%', 'display': 'inline-block'}
    ),
    
    html.P("Filter by company status:"),
    dcc.RadioItems(
        id='status-filter',
        options=[
            {'label': 'All Companies', 'value': 'all'},
            {'label': 'Listed Companies', 'value': 1},
            {'label': 'Delisted Companies', 'value': 0}
        ],
        value='all', 
        style={'width': '100%', 'margin-bottom': '20px'}
    ),

    html.Div([
        dcc.Graph(id='correlation-heatmap'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        html.Div([
            html.P("Select Metric for Time Series Analysis:"),
            dcc.Dropdown(
                id='metric-time-series',
                options=[
                    {'label': 'Total Assets', 'value': 'Assets'},
                    {'label': 'Total Liabilities', 'value': 'Liabilities'},
                ],
                value=['Assets'],
                multi=True,
                style={'width': '90%'}
            ),
            dcc.Graph(id='time-series-chart'),
        ], style={'display': 'inline-block', 'width': '49%'}),
        html.Div([
            dcc.Graph(id='net-cashflow-trend-graph'),
        ], style={'display': 'inline-block', 'width': '49%'}),

    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Div([
        html.H2("Select Cash Flow Metric for Comparison:"),
        dcc.Dropdown(
            id='cashflow-metric-selector-bar',
            options=[
                {'label': 'Operating Cash Flow', 'value': 'opCashflow'},
                {'label': 'Net Cash Flow', 'value': 'netCashflow'},
                {'label': 'Net Income', 'value': 'netIncome'},
                {'label': 'Sales', 'value': 'sales'},
            ],
            value='opCashflow',
            style={'width': '48%'}
        ),
        dcc.Graph(id='cashflow-comparison-bar-chart'),
    ]),

#----------------------------------------------------------------------------------------------
    html.Div([
        html.H1("PCA Visualization Dashboard"),
        dcc.Graph(id='scree-plot',
                figure=px.bar(x=[f'PCA{i+1}' for i in range(20)], y=explained_var, labels={'x': 'Component', 'y': 'Explained Variance'}, title="PCA Variance Distribition")),
        dcc.Graph(id='cumulative-variance-plot',
                figure=px.line(x=[f'PCA{i+1}' for i in range(20)], y=cumulative_var, labels={'x': 'Component', 'y': 'Cumulative Explained Variance'}, title="Cumulative Variance Plot")),
        dcc.Graph(id='biplot',
                figure=px.scatter(components_df, x='PCA1', y='PCA2', title="PCA Biplot (PCA1 vs PCA2)")),
        dcc.Graph(id='3d-biplot',
                figure=px.scatter_3d(components_df, x='PCA1', y='PCA2', z='PCA3', title="3D PCA Biplot (PCA1 vs PCA2 vs PCA3)"))
    ])
])

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('scale-selector', 'value'), 
     Input('status-filter', 'value')],
)
def update_correlation_heatmap(selected_scale, status_filter):
    df = {
        'minmax': df_minmax,
        'standard': df_standard,
        'robust': df_robust,
        'normalized': df_normalized,
    }.get(selected_scale, df_standard)

    if status_filter != 'all': 
        df = df[df['status'] == status_filter]

    fig = create_correlation_heatmap(df)
    return fig
    

def create_correlation_heatmap(df):
    corr_matrix = df.corr()

    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        showscale=True,
        colorscale='RdBu'
    )

    fig.update_layout(
        title='Correlation Matrix',
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'},
        margin={'l': 140, 'b': 140, 't': 50, 'r': 10},
        hovermode='closest',
        width=1200, 
        height=800,  
        autosize=False, 
    )

    return fig

@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('scale-selector', 'value'),
     Input('metric-time-series', 'value')]
)
def update_time_series(selected_scale, selected_metrics):
    if isinstance(selected_metrics, str):
        selected_metrics = [selected_metrics] 
    elif not isinstance(selected_metrics, list):
        selected_metrics = list(selected_metrics)

    df = {
        'minmax': df_minmax,
        'standard': df_standard,
        'robust': df_robust,
        'normalized': df_normalized,
    }.get(selected_scale, df_standard)

    traces = []

    metric_to_column_prefix = {
        'Assets': 'totalAsset',
        'Liabilities': 'totalLiabilities'
    }

    df_listed = df[df['status'] == 1]
    df_delisted = df[df['status'] == 0]

    for metric in selected_metrics:
        column_prefix = metric_to_column_prefix[metric]

        for df_subset, label in zip([df_listed, df_delisted], ['Listed', 'Delisted']):
            y_values = [df_subset[f'{column_prefix}Q{i}'].mean() for i in range(1, 6)]

            traces.append(go.Scatter(
                x=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                y=y_values,
                mode='lines+markers',
                name=f'{metric} ({label})'
            ))

    layout = go.Layout(
        title='Assets/Liabilities Over Quarters for Listed vs Delisted Companies',
        xaxis={'title': 'Quarter'},
        yaxis={'title': 'Value'},
        autosize=True
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig



@app.callback(
    Output('cashflow-comparison-bar-chart', 'figure'),
    [Input('scale-selector', 'value'),
     Input('cashflow-metric-selector-bar', 'value')]
)
def update_cashflow_comparison_bar_chart(selected_scale, selected_cashflow_metric):
    quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    df = {
        'minmax': df_minmax,
        'standard': df_standard,
        'robust': df_robust,
        'normalized': df_normalized,
    }[selected_scale]

    cashflow_columns = [f'{selected_cashflow_metric}Q{i}' for i in range(1, 6)]
    
    df_listed = df[df['status'] == 1][cashflow_columns].mean()
    df_delisted = df[df['status'] == 0][cashflow_columns].mean()

    fig = go.Figure(data=[
        go.Bar(
            name='Listed',
            x=quarters,
            y=df_listed,
            marker_color='green'
        ),
        go.Bar(
            name='Delisted',
            x=quarters,
            y=df_delisted,
            marker_color='red'
        )
    ])
    
    fig.update_layout(
        barmode='group',
        title=f'Average {selected_cashflow_metric.replace("Cashflow", " Cash Flow")} by Listing Status Across Quarters',
        xaxis_title='Quarter',
        yaxis_title=f'Average {selected_cashflow_metric.replace("Cashflow", " Cash Flow")}',
        legend_title='Listing Status'
    )
    
    return fig

@app.callback(
    Output('net-cashflow-trend-graph', 'figure'),
    [Input('scale-selector', 'value')]
)
def update_cashflow_trend_graph(selected_scale):
    df = {
        'minmax': df_minmax,
        'standard': df_standard,
        'robust': df_robust,
        'normalized': df_normalized,
    }[selected_scale]

    quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    listed_averages = [df[df['status'] == 1][f'netCashflow{i}'].mean() for i in quarters]
    delisted_averages = [df[df['status'] == 0][f'netCashflow{i}'].mean() for i in quarters]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=quarters,
        y=listed_averages,
        mode='lines+markers',
        name='Listed',
        line=dict(color='green'),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=quarters,
        y=delisted_averages,
        mode='lines+markers',
        name='Delisted',
        line=dict(color='red'),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='Average Net Cash Flow Trends Over Time by Listing Status',
        xaxis_title='Quarter',
        yaxis_title='Average Net Cash Flow',
        legend_title='Company Status',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)


app.layout.children.append(
    dcc.Graph(
        id='pca1-distribution',
        figure=px.histogram(
            components_df,
            x='PCA1',
            nbins=40,
            title="Distribution of PCA1 Values"
        )
    )
)
