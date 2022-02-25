import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

KEY_OFFSET = 2000


def interactive_2d_example():
    st.header('2D Interactive Example')

    circle_function = 'np.where((np.abs(x1)**2 + np.abs(x2)**2) < 1, 1, 0)'
    normal_distribution_function = 'stats.multivariate_normal((np.abs(x1)**2 + np.abs(x2)**2) < 1, 1, 0)'

    f_user_input_text = 'f(x1, x2): function as python code!'
    f_user_input_key = '2d_example_user_function'
    f_user = st.text_input(f_user_input_text, value=circle_function, key=f_user_input_key)






    user_function_code = f"def user_function(x1: np.array, x2: np.array):\n    return {f_user}"

    # exec does not change local function scope, therefore we have to manage the
    # overwriting of user_function ourselves.
    ldic = locals()
    exec(user_function_code, globals(), ldic)
    user_function = ldic["user_function"]

    col1, col2 = st.columns(2)

    with col1:
        a1 = st.number_input('x1 lower Bound (a1)', value=-1.5, step=0.1)
        a2 = st.number_input('x2 lower Bound (a2)', value=-1.5, step=0.1)
    with col2:
        b1 = st.number_input('x1 upper Bound (b1)', value=1.5, step=0.1, min_value=a1)
        b2 = st.number_input('x2 upper Bound (b2)', value=1.5, step=0.1, min_value=a2)

    IMPORTANCE_SAMPLING = 'Importance sampling with normal distribution'
    CRUDE_SAMPLING = 'Crude sampling with uniform distribution'

    sampling_method = st.selectbox('Sampling method', (CRUDE_SAMPLING, IMPORTANCE_SAMPLING))

    if sampling_method == IMPORTANCE_SAMPLING:
        col1, col2 = st.columns(2)
        with col1:
            normal_mean_x1 = st.number_input('Normal mean x1', value=0.0, step=0.1)
            normal_mean_x2 = st.number_input('Normal mean x2', value=0.0, step=0.1)
        with col2:
            std_mean_x1 = st.number_input('Normal sdt x1', value=1.0, step=0.1)
            std_mean_x2 = st.number_input('Normal sdt x2', value=1.0, step=0.1)

    n_max = 2000
    n = n_max
    #n = int(st.slider('Number of samples (n)', min_value=1, max_value=n_max, step=1, value=300, key=KEY_OFFSET + 5))

    if sampling_method == IMPORTANCE_SAMPLING:
        X1_all = np.random.normal(normal_mean_x1, std_mean_x1, size=n)
        X2_all = np.random.normal(normal_mean_x2, std_mean_x2, size=n)
    else:
        X1_all = np.random.uniform(a1, b1, n_max)
        X2_all = np.random.uniform(a2, b2, n_max)

    X1 = X1_all[:n]
    X2 = X2_all[:n]
    Index_all = np.arange(0, n_max)
    Index = Index_all[:n]
    Y = user_function(X1, X2)

    if sampling_method == IMPORTANCE_SAMPLING:
        # Importance sampling might have some bugs
        X1_g = stats.norm.pdf(X1, normal_mean_x1, std_mean_x1)
        X2_g = stats.norm.pdf(X1, normal_mean_x2, std_mean_x2)
        X_g = X1_g * X2_g

        Y_cumulative_sum = np.cumsum(Y / X_g)
        Y_cumulative_approximation = Y_cumulative_sum / (Index_all + 1) # * (b1 - a1) * (b2 - a2)
        st.write(f'After $N={n}$ iterations we get the approximation from the ' +
                 'sequence of normaly distributed random numbers $(x_n)_{n \in \mathbb{N}^2}$')

        st.latex(f'\\int_{{{a1}}}^{{{b1}}} \\int_{{{a2}}}^{{{b2}}} f(x_1,x_2) dx_2 dx_1 ' +
                 '\\approx \\frac{1}{N} \sum_{n=1}^{N} \\frac{f(x_n)}{norm(x_n)} ' +
                 f'\\approx {Y_cumulative_approximation[-1]:.2f}')
    else:
        Y_cumulative_sum = np.cumsum(Y)
        Y_cumulative_approximation = Y_cumulative_sum / (Index_all+1) * (b1-a1) * (b2-a2)

        st.write(f'After $N={n}$ iterations we get the approximation from the ' +
                 'sequence of uniform random numbers $(x_n)_{n \in \mathbb{N}^2}$')

        st.latex(f'\\int_{{{a1}}}^{{{b1}}} \\int_{{{a2}}}^{{{b2}}} f(x_1,x_2) dx_2 dx_1 ' +
                 '\\approx \\frac{(b_1-a_1)(b_2 - a_2)}{N} \sum_{n=1}^{N} f(x_n) ' +
                 f'\\approx {Y_cumulative_approximation[-1]:.2f}')

    slider = alt.binding_range(min=1,
                               max=n,
                               step=1)
    select_index = alt.selection_single(name='index_filter',
                                        fields=['index_filter'],
                                        bind=slider)

    select_condition_string = "datum.index < index_filter_index_filter"

    source = pd.DataFrame({
        'x1': X1,
        'x2': X2,
        'f(x)': Y,
        'index': Index,
        'approximation': Y_cumulative_approximation,
    })
    base_chart = alt.Chart(source, width=500, height=470)
    samples = base_chart.mark_point().transform_filter(select_condition_string).encode(
        x=alt.X('x1', scale=alt.Scale(domain=[a1, b1]), title='x1'),
        y=alt.Y('x2', scale=alt.Scale(domain=[a2, b2]), title='x2'),
        color='f(x)',
        tooltip=['x1', 'x2', 'f(x)']
    ).properties(
        title='Samples'
    ).interactive()

    history_chart = base_chart.properties(
        height=100,
    )

    samples_mean = history_chart.mark_line().transform_filter(select_condition_string).encode(
        x='index',
        y='approximation',
        tooltip=['index', 'approximation'],
    )

    full_chart = (samples & samples_mean).add_selection(select_index)

    st.altair_chart(full_chart)
