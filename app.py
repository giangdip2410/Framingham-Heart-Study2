"""
Project: Ung Dung Machine Learning Chuan Doan Kha Nang Mac Benh Tim trong 10 nam toi
Class : Python
Teacher : Ts.Vu Tien Dung
Group 5, Ms Data Science K3
"""

# Imports
import dash
import ast
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from whitenoise import WhiteNoise
import joblib
import glob
import os
from datasets import Datasets

from styles import *
import dash_daq as daq

# preprocesing for models
def get_transform(nparray):
    data_file = "framingham.csv"
    numeric_var = [
        "age",
        "cigsPerDay",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    ]
    level_var = ["education"]
    category_var = [
        "male",
        "currentSmoker",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
    ]
    target = ["TenYearCHD"]

    # Create Data object
    data = Datasets(
        data_file=data_file,
        cat_cols=category_var,
        num_cols=numeric_var,
        level_cols=level_var,
        label_col=target,
        train=True,
    )
    return data.preprocess_newdata(nparray)


# predict model
def get_predict(data):
    data = get_transform(np.array(data))
    list_file = glob.glob(os.path.join("./log/best_model", "*"))
    list_model = []
    for l in list_file:
        if l.split("/")[-1].split(".")[-1] == "pkl":
            list_model.append(l)
    model_path = list_model[0]
    print(f"Loading..model {model_path}")
    clf = joblib.load(model_path)
    pred = clf.predict_proba(data)[:, 1][0]
    return pred


# Set stylesheets and app.
# ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"
external_stylesheets = [dbc.themes.LUX, FA]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Group5 - HUS - DSK3"
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root="static/")


# Defining app layout
margin_bottom = "30px"

# Banner

banner = dbc.Row(
    children=[
        dbc.Col(
            html.Img(
                src=app.get_asset_url("heart.png"),
                id="logo",
                style={"border-radius": "50%"},
            ),
            width=2,
            align="left",
        ),
        dbc.Col(
            html.H1("Prediction risk for cardiovascular disease"),
            align="center",
            width=10,
        ),
    ],
    style={"margin-bottom": "50px", "margin-top": "-30px"},
    align="center",
)

markdown_text = """
>
>                           Fill out the below information to predict the risk for future 10-year cardiovascular disease
>
"""

gender = dcc.RadioItems(
    id="gender",
    options=[
        {"label": "Male", "value": 1},
        {"label": "Female", "value": 0},
    ],
    value=1,
    labelStyle={"width": "42%", "display": "inline-block"},
)

age = dcc.Input(id="age", type="number", min=10, max=100, placeholder=40)

education = dcc.Slider(
    id="education",
    min=1,
    max=4,
    marks={
        1: "HS",
        2: "HS Diploma/GED",
        3: "College",
        4: "Degree",
    },
    value=1,
    included=False,
)

cursmoke = dcc.RadioItems(
    id="cursmoke",
    options=[
        {"label": "Yes", "value": 1},
        {"label": "No", "value": 0},
    ],
    value=0,
    labelStyle={"width": "46%", "display": "inline-block"},
)

cigaperday = dcc.Input(id="cigaperday", type="number", min=0, max=100, placeholder=0)

bpmeds = dcc.Input(id="bpmeds", type="number", min=0, max=5, placeholder=0)

prevalentstroke = dcc.RadioItems(
    id="prevalentstroke",
    options=[
        {"label": "Yes", "value": 1},
        {"label": "No", "value": 0},
    ],
    value=0,
    labelStyle={"width": "46%", "display": "inline-block"},
)

prevalenthyp = dcc.RadioItems(
    id="prevalenthyp",
    options=[
        {"label": "Yes", "value": 1},
        {"label": "No", "value": 0},
    ],
    value=0,
    labelStyle={"width": "46%", "display": "inline-block"},
)

diabetes = dcc.RadioItems(
    id="diabetes",
    options=[
        {"label": "Yes", "value": 1},
        {"label": "No", "value": 0},
    ],
    value=0,
    labelStyle={"width": "46%", "display": "inline-block"},
)

totchol = dcc.Input(id="totchol", type="number", min=100, max=800, placeholder=400)

sysbp = dcc.Input(id="sysbp", type="number", min=70, max=310, placeholder=190.5)

diabp = dcc.Input(id="diabp", type="number", min=30, max=160, placeholder=85.5)

bmi = dcc.Input(id="bmi", type="number", min=10, max=60, placeholder=24.5)

heartrate = dcc.Input(id="heartrate", type="number", min=40, max=160, placeholder=75)

glucose = dcc.Input(id="glucose", type="number", min=30, max=330, placeholder=80)

white_button_style = {
    "background-color": "white",
    "color": "black",
    "height": "50px",
    "width": "250px",
    "margin-top": "50px",
    "margin-left": "450px",
}

blue_button_style = {
    "background-color": "CornflowerBlue",
    "color": "white",
    "height": "50px",
    "width": "250px",
    "margin-top": "50px",
    "margin-left": "450px",
}


app.layout = dbc.Jumbotron(
    style={"background-color": "#ebebeb"},  # ADD SETTINGS HERE
    children=[
        # Banner
        # Main Layout
        dbc.Row(  # ADD SETTINGS HERE
            children=[
                # PARAMETER SETTINGS COLUMN
                dbc.Col(
                    children=[
                        banner,
                        # dcc.Markdown(children=markdown_text),
                        html.Div(
                            children="Fill out the below information to predict the risk for future 10-year cardiovascular disease.",
                            style={
                                "textAlign": "center",
                                "marginTop": "-50px",
                                "margin-bottom": "25px",
                                "font-weight": "bold",
                                "color": "#111111",
                                "font-size": "18px",
                            },
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            dbc.Row(
                [
                    html.Div(
                        dbc.Col(
                            html.Div("Gender"),
                            # width={"size": 2, "order": 1, "offset": 3},
                        ),
                        style={"margin-left": "390px"},
                    ),
                    html.Div(
                        dbc.Col(
                            html.Div(gender),
                            # width={"size": 2, "order": 12, "offset": 0},
                        ),
                        style={"margin-left": "10px"},
                    ),
                ]
            ),
            style={"margin-bottom": "5px"},
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Age"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "410px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(
                            age,
                        ),
                        # width={"size": 100, "order": 12, "offset": 2},
                    ),
                    style={
                        "margin-left": "10px",
                        "margin-top": "-5px",
                    },
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Education"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "370px", "margin-top": "20px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(education),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={
                        "margin-left": "-10px",
                        "margin-top": "22px",
                        "width": "57%",
                        "color": "blue",
                    },
                ),
            ]
        ),
        html.Div(
            dbc.Row(
                [
                    html.Div(
                        dbc.Col(
                            html.Div("Current Smoke"),
                            # width={"size": 2, "order": 1, "offset": 3},
                        ),
                        style={"margin-left": "335px", "margin-top": "10px"},
                    ),
                    html.Div(
                        dbc.Col(
                            html.Div(cursmoke),
                            # width={"size": 2, "order": 12, "offset": 0},
                        ),
                        style={"margin-left": "10px", "margin-top": "11px"},
                    ),
                ]
            ),
            style={"margin-bottom": "5px", "margin-top": "5px"},
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Cigarettes smoked per day"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "250px", "margin-top": "-5px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(cigaperday),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "10px", "margin-top": "-5px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("A mount of BP medication is on"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "215px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(bpmeds),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={
                        "margin-left": "10px",
                        "margin-top": "15px",
                    },
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Prevalent Stroke"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "321px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(prevalentstroke),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "13px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Prevalence of hypertension"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "243px", "margin-top": "5px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(prevalenthyp),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "14px", "margin-top": "5px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Diabetes"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "375px", "margin-top": "5px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(diabetes),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "5px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Total cholesterol"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "320px", "margin-top": "3px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(totchol),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "3px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Systolic blood pressure"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "272px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(sysbp),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Diastolic blood pressure"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "265px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(diabp),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Body Mass Index"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "317px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(bmi),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Heart rate in bpm"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "312px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(heartrate),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Div("Glucose level"),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                    style={"margin-left": "343px", "margin-top": "15px"},
                ),
                html.Div(
                    dbc.Col(
                        html.Div(glucose),
                        # width={"size": 2, "order": 12, "offset": 0},
                    ),
                    style={"margin-left": "15px", "margin-top": "15px"},
                ),
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    dbc.Col(
                        html.Button(
                            id="button",
                            children=["Prediction"],
                            n_clicks=0,
                            style=white_button_style,
                        ),
                        # width={"size": 2, "order": 1, "offset": 3},
                    ),
                ),
            ]
        ),
        html.Div(
            id="result",
            style={
                "margin-left": "350px",
                "margin-top": "40px",
                "color": "blue",
                "font-size": "18px",
            },
        ),
    ],
)


@app.callback(
    Output(component_id="result", component_property="children"),
    [
        Input("gender", "value"),
        Input("age", "value"),
        Input("education", "value"),
        Input("cursmoke", "value"),
        Input("cigaperday", "value"),
        Input("bpmeds", "value"),
        Input("prevalentstroke", "value"),
        Input("prevalenthyp", "value"),
        Input("diabetes", "value"),
        Input("totchol", "value"),
        Input("sysbp", "value"),
        Input("diabp", "value"),
        Input("bmi", "value"),
        Input("heartrate", "value"),
        Input("glucose", "value"),
        Input(component_id="button", component_property="n_clicks"),
    ],
)
def update_output(
    gender,
    age,
    education,
    cursmoke,
    cigaperday,
    bpmeds,
    prevalentstroke,
    prevalenthyp,
    diabetes,
    totchol,
    sysbp,
    diabp,
    bmi,
    heartrate,
    glucose,
    n_clicks,
):
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks > 0:
        arr = np.array(
            [
                gender,
                age,
                education,
                cursmoke,
                cigaperday,
                bpmeds,
                prevalentstroke,
                prevalenthyp,
                diabetes,
                totchol,
                sysbp,
                diabp,
                bmi,
                heartrate,
                glucose,
            ]
        )
        for v in arr:
            if v is None:
                return "Please input all information for risk prediction"
        # predict
        init_features = [float(x) for x in arr]
        X_new = np.array([init_features])
        y_pred = get_predict(X_new)
        if y_pred > 0.5:
            prediction_text = "Your Risk score is {} . High risk, please care more about your health".format(
                str(round(y_pred, 2))
            )
        else:
            prediction_text = "Your Risk score is {} . Congras, let's keep your healthy hobbies!".format(
                str(round(y_pred, 2))
            )
        return prediction_text


@app.callback(
    Output("button", "style"),
    [Input("button", "n_clicks")],
)
def change_button_style(n_clicks):

    if n_clicks > 0:

        return blue_button_style

    else:

        return white_button_style


# Statring the dash app
if __name__ == "__main__":
    app.run_server(debug=True)
