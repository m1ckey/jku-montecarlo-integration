import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import streamlit as st

np.random.seed(1)


st.title('Basic concept of Monte Carlo Integration')
st.write('The idea is to estimate a Integral of a function.')
st.write('As simple example we use a univariate function in form:')
st.latex('F = \int_{a}^{b} f(x) \,dx')

st.write('so let:')
st.latex('F = \int_{0}^{1} 2x \,dx = 1')

def f(x):
    return 2 * x

def montecarlo(a,b,X,f):
    '''
    :param a: lower limit
    :param b: upper limit
    :param X: points
    :param f: function
    :return: estimate of the integral
    '''
    y = f(X)
    return (b-a)*y.sum()/len(X)


# integral limits
a, b = 0, 1
x = np.linspace(a, b)
y = f(x)

# plot function
fig, ax = plt.subplots(figsize= (10,6))
ax.set_ylabel('f(x)')
ax.set_xlabel('x')
ax.grid(True)
ax.axhline(y=0, color='k', linewidth = 0.5)

ax.plot(x, y, 'r', linewidth=1)     # plot function line

# Make the shaded region
ix = np.linspace(a, b)
iy = f(ix)
verts = [(a, 0), *zip(ix, iy), (b, 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
ax.add_patch(poly)
ax.text(0.6 * (a + b), 0.3, r"$\int_0^1 f(x)\mathrm{d}x$",
        horizontalalignment='center', fontsize=20)

st.pyplot(fig)  # plot figure

st.write('We can take random points $x_i$ between $a=0$ and $b=1$. \
We can get a rectangle area of each point by multiply $f(x_i)\cdot (b-a)$.')
st.write('The idea of the Monte Carlo integration is to approximate the integrated value by sum up the rectangle areas and take the average of it.')

# create subplots for each point
x_rand = np.random.uniform(a, b, 9)
fig, ax = plt.subplots(3,3, figsize=(10,6))
for axs, x_i in zip(ax.reshape(-1), x_rand):
    axs.plot(x, y, 'r', linewidth=1)  # plot function line
    verts = [(a, 0), (a, f(x_i)),(b, f(x_i)), (b, 0)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    axs.add_patch(poly)
    axs.plot(x_i,f(x_i),marker='o',color='b')
st.pyplot(fig)  # plot figure

st.write('Then the Monte Carlo estimator has the following form with $n \in \mathbb{N}$ for the random draws of $X$')
st.latex(r'\left\langle F^n \right\rangle  = (b-a) \dfrac{1}{n-1} \sum_{i=0}^{n} f(X_i) ')
st.write(f'for the upper example with 9 points we would get a estimate of $F={montecarlo(a,b,x_rand,f):.3f}$')
st.write('')
st.write('Not a good estimate because of few random numbers. But as we increase the numbers we will get a better result. \
The law of large numbers gives us that as $n$ approaches infinity, the the Monte Carlo estimator converges in probability to $F$')
st.latex(r' Pr\left\lbrace \lim\limits_{n\to \infty} \left\langle F^n\right\rangle  = F \right\rbrace = 1')

x_rand = np.random.uniform(a, b, 1000)
st.write(f'If we increase the random numbers, for example 1000 points we get a estimate of $F={montecarlo(a,b,x_rand,f):.3f}$ , a much better result')


# start of Interactive Code
st.header('Interactive code')

n_max = st.number_input('Maximum Random Numbers', min_value=1, step=1, value=1000)
a = st.number_input('Lower Bound (a)', value=0.0, step=0.1)
b = st.number_input('Upper Bound (b)', value=1.1, step=0.1)

X_all = np.random.uniform(a, b, n_max)

n = st.slider('Number of samples (n)',
              min_value=1,
              max_value=n_max,
              step=1,
              value=10)
X = X_all[:n]
Y = f(X)

# result montecarlo
st.write(f'$F^n={montecarlo(a,b,X,f):.3f}$')


source = pd.DataFrame({'x': X, 'y': Y})
chart = alt.Chart(source)
samples = chart.mark_point().encode(
    x=alt.X('x', scale=alt.Scale(domain=[a, b])),
    y=alt.Y('y', scale=alt.Scale(domain=[f(a), f(b)])),
    tooltip=['x', 'y']
)
samples_mean = chart.mark_rule(
    color='violet',
    tooltip='arithmetic mean'
).encode(
    y=alt.Y('average(y)')
)
st.altair_chart(samples + samples_mean)

N = np.arange(n) + 1
A = np.zeros_like(X)
for i in range(n):
    A[i] = montecarlo(a,b,X[:i + 1],f)

source = pd.DataFrame({'Number of samples': N, 'F(x)': A})
chart = alt.Chart(source).mark_line(color='violet').encode(
    x='Number of samples',
    y=alt.Y('F(x)', scale=alt.Scale(domain=[source.min()[1], source.max()[1]])),
    tooltip=['Number of samples', 'F(x)']
)
st.altair_chart(chart)
