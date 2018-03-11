#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sb
from draw import plot_bokeh_candle_dochl, candlestick_vol


def getChangeRatio(quotes, draw=False):
    """docstring for getChangeRatio"""
    changeRatio = quotes['close'].pct_change() * 100
    if draw:
        sb.distplot(changeRatio[1:])
    return changeRatio


def getAmplitude(quotes, draw=False):
    amplitude = (quotes['high']-quotes['low']) / quotes['close'].shift(1) * 100
    if draw:
        sb.distplot(amplitude[1:])
    return amplitude


def getJumpPower(quotes, draw=False):
    DAYS = 400
    if quotes.shape[0] < DAYS:
        jump_threshold = quotes['close'].median() * 0.03
    else:
        jump_threshold = quotes['close'][-1 * DAYS:].median() * 0.03
    dump_power = np.where(np.abs(quotes['low'] - quotes['close'].shift(1)) > jump_threshold,
                          (quotes['low'] - quotes['close'].shift(1)) / jump_threshold,
                          np.nan)
    if draw:
        # plot_bokeh_candle_dochl(quotes.index, quotes['open'].values, quotes['close'].values,
        #                         quotes['high'].values, quotes['low'].values, symbol='')
        candlestick_vol(quotes)
    return dump_power
