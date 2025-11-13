
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.io import loadmat

#读取.mat文件
'''
to process the data in the same way as trainmodel.py.
'''
arr = loadmat("./datasets/Guangzhou-data-set/tensor.mat")["tensor"]
imputed = loadmat("./imputer/Guangzhou-data-setImputed/Imputed_tensor.mat")["tensor"]
sparse = loadmat("./datasets/Guangzhou-data-setMitMissingValue/sparse_tensor.mat")["tensor"]
print("original data shape:", arr.shape)
print("sparse data shape:", imputed.shape)

num_segments, num_days, time_steps = arr.shape

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Traffic Speed Imputation Visualization (.mat data)"),

    html.Label("Select Road Segment:"),
    dcc.Dropdown(
        id='segment_dropdown',
        options=[{'label': f'Road Segment {i}', 'value': i} for i in range(num_segments)],
        value=0,
        clearable=False
    ),

    html.Br(),

    html.Label("Select Day:"),
    dcc.Dropdown(
        id='day_dropdown',
        options=[{'label': f'Day {d}', 'value': d} for d in range(num_days)],
        value=0,
        clearable=False
    ),

    html.Br(),

    dcc.Graph(id='imputation_graph')
])

@app.callback(
    Output('imputation_graph', 'figure'),
    Input('segment_dropdown', 'value'),
    Input('day_dropdown', 'value')
)

def update_graph(segment_index, day_index):
    fig = go.Figure()

    # 当前路段、当前日期的一天 144 个时间点
    y_orig = arr[segment_index, day_index, :]
    y_incomp = sparse[segment_index, day_index, :]
    y_imp = imputed[segment_index, day_index, :]

    # 原始数据
    fig.add_trace(go.Scatter(
        y=y_orig,
        mode='lines',
        name='Original',
        line=dict(color="#2EAB71", width=3)
    ))

    # 随机缺失数据
    fig.add_trace(go.Scatter(
        y=y_incomp,
        mode='markers+lines',
        name='With Missing',
        line=dict(color='#8B0000', width=2, dash='dot'),
        marker=dict(color='#8B0000', size=6, symbol='circle-open')
    ))

    fig.add_trace(go.Scatter(
        y=y_imp,
        mode='lines',
        name='Imputed (BGCP)',
        line=dict(color="#FFD700", width=3)
    ))

    fig.update_layout(
        title=f"Road Segment {segment_index} - Day {day_index}",
        xaxis_title="Time of Day (index 0 ~ 143)",
        yaxis_title="Speed",
        template="plotly_white",
        plot_bgcolor='rgba(248,249,250,1)',
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear",
            gridcolor='rgba(230,230,230,0.8)'
        ),
        yaxis=dict(
            gridcolor='rgba(230,230,230,0.8)'
        )
    )

    return fig

# ===================== 4. 启动服务 =====================

if __name__ == '__main__':
    app.run(debug=True)

