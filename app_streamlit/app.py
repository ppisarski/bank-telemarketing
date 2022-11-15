"""
Copyright © 2022 Pawel Pisarski
"""

import os
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff

pio.templates.default = "plotly_white"

PLOTLY_CONFIG = dict(displayModeBar=True, displaylogo=False)
PLOTLY_LAYOUT = dict(margin=dict(r=0, t=0, l=0, b=0), plot_bgcolor="rgba(0,0,0,0)")

STREAMLIT_STYLE = """
    <style>
        footer {visibility: hidden;}
        footer:after {
            content:'Copyright © 2022 Pawel Pisarski';
            visibility: visible;
            display: block;
            position: relative;
            padding: 5px;
            top: 2px;
        }
    </style>
"""


@st.cache(persist=False, allow_output_mutation=True)
def get_data() -> pd.DataFrame:
    return pd.read_parquet(os.path.join("data", "train.parquet"))


def explore(data: pd.DataFrame):
    """
    Exploratory data analysis
    """
    col1, col2, col3 = st.columns(3)
    attr = col1.selectbox("Attribute", data.columns)
    attr2 = col2.selectbox("Second Attribute", [None, *data.columns])
    color = col3.selectbox("Color by", [None, *data.columns])

    print(data[attr].unique())
    if attr2 is None:
        fig = px.histogram(data, y=attr, color=color)
    else:
        fig = px.strip(data, x=attr2, y=attr, color=color, stripmode="group")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("Dataset"):
        st.dataframe(data)

    with st.expander("Dataset Summary"):
        st.dataframe(data.describe())


def model(data: pd.DataFrame):
    """
    Model
    """
    col1, col2 = st.columns(2)
    fold = col1.select_slider("Fold", range(5))
    threshold = col2.slider("Threshold", 0., 1., 0.5, 0.01)

    st.header("Confusion matrix")

    z = [[0.1, 0.3, 0.5, 0.2],
         [1.0, 0.8, 0.6, 0.1],
         [0.1, 0.3, 0.6, 0.9],
         [0.6, 0.4, 0.2, 0.2]]

    x = ['healthy', 'multiple diseases', 'rust', 'scab']
    y = ['healthy', 'multiple diseases', 'rust', 'scab']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.header("ROC AUC")

    st.header("Feature Importance")

    with st.expander("Model"):
        with open(os.path.join("data", "estimator.html"), "r") as f:
            st.components.v1.html(f.read(), scrolling=True)


def predict():
    """
    Predict
    """
    col1, col2 = st.columns(2)
    fold = col1.select_slider("Fold", range(5))
    threshold = col2.slider("Threshold", 0., 1., 0.5, 0.01)

    st.header("Inputs")

    st.header("Output")


def main():
    st.set_page_config(page_title="Bank Telemerketing", layout="wide")
    st.sidebar.title("Analysis")
    layout = st.sidebar.selectbox("Type", [
        "Predict",
        "Explore",
        "Model",
    ])
    st.title(f"Bank Telemarketing - {layout}")
    df = get_data()

    if layout in ["Predict"]:
        predict()
    if layout in ["Explore"]:
        explore(df)
    if layout in ["Model"]:
        model(df)
    st.markdown(__doc__)
    # st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
