import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

np.random.seed(1)

st.title('Monte Carlo Integration')
st.header('1D')

st.latex('\int_{0}^{1} 2x \,dx = 1')


def f(x):
    return 2 * x


n_max = 200
X_all = np.random.uniform(0, 1, (n_max,))
Y_all = f(X_all)

n = st.slider('Number of samples',
              min_value=1,
              max_value=n_max,
              step=1)
X = X_all[:n]
Y = f(X)

source = pd.DataFrame({'x': X, 'y': Y})
chart = alt.Chart(source)
samples = chart.mark_point().encode(
    x=alt.X('x', scale=alt.Scale(domain=[0, 1])),
    y=alt.Y('y', scale=alt.Scale(domain=[0, 2])),
    tooltip=['x', 'y']
)
samples_mean = chart.mark_rule(
    color='violet',
    tooltip='arithmetic mean'
).encode(
    y=alt.Y('average(y)')
)
st.altair_chart(samples + samples_mean)

N = np.arange(n_max) + 1
A = np.zeros_like(X_all)
for i in range(n_max):
    A[i] = Y_all[:i + 1].mean()

source = pd.DataFrame({'Number of samples': N, 'Average of y': A})
chart = alt.Chart(source).mark_line(color='violet').encode(
    x='Number of samples',
    y=alt.Y('Average of y', scale=alt.Scale(domain=[0.5, 1.5])),
    tooltip=['Number of samples', 'Average of y']
)
st.altair_chart(chart)
