import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from shared import montecarlo


def interactive_1d_example():
    st.header('1D Interactive Example')

    f_user = st.text_input('f(x): function as python code!'
                           ' [Addition: + , Subtraction: -, Multiplication: *, Division: /, Power: **]', value='2*x')

    st.write("Some example functions")
    st.write("x**2")
    st.write("np.where(x < 0.5, 0, 1)")
    st.write("np.sin(x * 2 * np.pi)")

    a = st.number_input('Lower Bound (a)', value=0., step=0.1)
    b = st.number_input('Upper Bound (b)', value=1., step=0.1, min_value=a)
    n_max = 2000
    n = int(st.slider('Number of samples (n)', min_value=1, max_value=n_max, step=1, value=10))


    # create function from string
    # f_u = "f=lambda x :"


    # not actually used, will be overwritten by user input. Helps editors with error finding + type checking.
    def user_function(x):
        return x

    user_function_code = f"def user_function(x):\n    return {f_user}"
    # exec does not change local function scope, therefore we have to manage the
    # overwriting of user_function ourselves.
    ldic = locals()
    exec(user_function_code, globals(), ldic)
    user_function = ldic["user_function"]

    X_all = np.random.uniform(a, b, n_max)
    X = X_all[:n]
    Y = user_function(X)

    # result montecarlo
    st.subheader('1D Crude Monte Carlo method')
    st.write(f'$I={montecarlo(a, b, X, user_function):.3f}$')

    source = pd.DataFrame({'x': X, 'f(x)': Y})
    chart = alt.Chart(source)
    samples = chart.mark_point().encode(
        x=alt.X('x', scale=alt.Scale(domain=[a, b])),
        y=alt.Y('f(x)', scale=alt.Scale(domain=[min(Y), max(Y)]), title='f(x)'),
    ).properties(
        title='Samples'
    )

    samples_mean = chart.mark_rule(
        color='violet',
        tooltip='arithmetic mean of f(x)'
    ).encode(
        y=alt.Y('average(f(x))', title='mean f(x)')
    )

    st.altair_chart(samples + samples_mean)

    N = np.arange(n) + 1
    A = np.zeros_like(X)
    for i in range(n):
        A[i] = montecarlo(a, b, X[:i + 1], user_function)

    source = pd.DataFrame({'n': N, 'I': A})
    chart = alt.Chart(source).mark_line(color='violet').encode(
        x='n',
        y=alt.Y('I', scale=alt.Scale(domain=[source.min()[1], source.max()[1]]), title='Approximation of F(x)'),
    ).properties(
        title='Montecarlo'
    )
    st.altair_chart(chart)
