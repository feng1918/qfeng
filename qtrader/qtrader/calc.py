#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.api as sm
from statsmodels import regression


def linear_regression(x, y, zoom=False):
    """
    Use statsmodels.regression.linear_model.OLS for x, y
    :x:  Analyzing data
    :y:  Analyzing data
    :zoom: Using zoom?
    :return (model, zoom_factor)
    """
    zoom_factor = 1
    if zoom:
        zoom_factor = x.max() / y.max()
        y = zoom_factor * y
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    return model, zoom_factor


def linear_regression_y(y, zoom=False):
    """
    Use a liner_regreess model: y = kx + b to calculate degree
    :y:  Analyzing data
    :return (model, zoom_factor)
    """
    x = np.arange(0, len(y))
    return linear_regression(x, y, zoom)


def linear_regression_degree(y):
    """
    Use a liner_regreess model: y = kx + b to calculate degree
    :y: Analyzing data
    :return: degree
    """
    model, zoom_factor = linear_regression_y(y, zoom=True)
    rad = model.params[1]
    return np.rad2deg(rad)


def linear_regression_kb(y):
    """
    get (k, b) of a liner_regreess model: y = kx + b
    :y: Analyzing data
    :return: (k, b)
    """
    model, _ = linear_regression_y(y)

    k = model.params[1]
    b = model.params[0]
    return (k, b)
