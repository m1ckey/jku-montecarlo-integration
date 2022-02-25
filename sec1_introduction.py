import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon

from shared import montecarlo


def introduction():
    st.title('Introduction to Monte Carlo Integration')
    st.write('The basic idea is to empirically estimate the definite integral of a function.')
    st.write('')
    st.write('We will use a univariate function as a simple example:')
    st.latex('f(x) = 2x')
    st.latex(r'F = \int_{a}^{b} f(x) \,dx')

    st.write('Geometric interpretation for $a=0$, $b=1$:')

    def f(x):
        return 2 * x


    # integral limits
    a, b = 0, 1
    x = np.linspace(a, b)
    y = f(x)

    # plot function
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')
    ax.grid(True)
    ax.axhline(y=0, color='k', linewidth=0.5)

    ax.plot(x, y, 'r', linewidth=1)  # plot function line

    # Make the shaded region
    ix = np.linspace(a, b)
    iy = f(ix)
    verts = [(a, 0), *zip(ix, iy), (b, 0)]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax.add_patch(poly)
    ax.text(0.6 * (a + b), 0.3, r"$\int_0^1 f(x)\mathrm{d}x = 1$",
            horizontalalignment='center', fontsize=20)

    st.pyplot(fig)  # plot figure

    st.write('If we take a random point $x_i$ between $a=0$ and $b=1$. '
             r'We can construct the area of a rectangle by multiplying $f(x_i)\cdot (b-a)$.')

    # create subplots for each point
    x_rand = np.random.uniform(a, b, 9)
    fig, ax = plt.subplots(3, 3, figsize=(10, 6))
    for axs, x_i in zip(ax.reshape(-1), x_rand):
        axs.plot(x, y, 'r', linewidth=1)  # plot function line
        verts = [(a, 0), (a, f(x_i)), (b, f(x_i)), (b, 0)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        axs.add_patch(poly)
        axs.plot(x_i, f(x_i), marker='o', color='b')
    st.pyplot(fig)  # plot figure

    st.write('The idea of Monte Carlo integration is to approximate the definite integral by drawing many $x_i$ from a '
             '(uniform) distribution $X$ and taking the average of the areas.')
    st.write('')
    st.write('The Monte Carlo estimator has the following form with $n$ random draws from $X$.')
    st.latex(r'I  \approx \dfrac{(b-a)}{n} \sum_{i=1}^{n} f(X_i)')
    st.write(f'for the upper example with $n=9$ we get an estimate of $F={montecarlo(a, b, x_rand, f):.3f}$')
    st.write('')
    st.write('Not a good estimate because $n$ is small. But as we increase $n$ we will get a '
             'better result. The law of large numbers says that the the Monte Carlo estimator '
             'converges in probability to $F$')
    st.latex(r' Pr\left\lbrace \lim\limits_{n\to \infty} I  = F \right\rbrace = 1')

    x_rand = np.random.uniform(a, b, 1000)
    st.write(f'If we set $n=1000$ we get an estimate of $F={montecarlo(a, b, x_rand, f):.3f}$ , a much better result')
