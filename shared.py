def montecarlo(a, b, X, f):
    '''
    :param a: lower limit
    :param b: upper limit
    :param X: points
    :param f: function
    :return: estimate of the integral
    '''
    Y = f(X)
    return (b - a) * Y.sum() / len(X)