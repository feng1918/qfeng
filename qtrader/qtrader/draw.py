#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.dates as mdates
# from matplotlib.mpl_finance import candlestick_ochl
from matplotlib.finance import candlestick_ochl, candlestick2_ochl, volume_overlay
import matplotlib.pyplot as plt
import bokeh.plotting as bp
from math import pi


def draw_candlestick(quotes):
    """docstring for draw_candlestick"""
    COLORUP = 'red'
    COLORDOWN = 'green'
    WIDTH = 0.6

    fig, axes = plt.subplots(figsize=(18, 7))
    candlestick_ochl(axes, zip(mdates.date2num(quotes.index.to_pydatetime()),
                               quotes['open'], quotes['high'],
                               quotes['low'], quotes['close']),
                     width=WIDTH, colorup=COLORUP, colordown=COLORDOWN)
    axes.autoscale_view()


def plot_bokeh_candle_dochl(date, p_open, close, high, low, symbol):
    COLORUP = 'red'
    COLORDOWN = 'green'
    bp.output_notebook()

    p = bp.figure(x_axis_type="datetime", plot_width=840, title=symbol)
    p.xaxis.major_label_orientation = pi / 4
    p.grid.grid_line_alpha = 0.3

    w = 24 * 60 * 60 * 1000
    mids = (p_open + close) / 2
    spans = abs(close - p_open)

    inc = close > p_open
    dec = p_open > close

    p.segment(date.to_datetime(), high, date.to_datetime(), low, color="black")
    p.rect(date.to_datetime()[inc], mids[inc], w, spans[inc], fill_color=COLORUP, line_color=COLORUP)
    p.rect(date.to_datetime()[dec], mids[dec], w, spans[dec], fill_color=COLORDOWN, line_color=COLORDOWN)
    bp.show(p)


def candlestick_vol(quotes):
    SCALE = 5
    COLORUP = 'red'
    COLORDOWN = 'green'

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(17, 10))

    candlestick2_ochl(ax, quotes['open'], quotes['close'], quotes['high'], quotes['low'],
                      width=0.5, colorup=COLORUP, colordown=COLORDOWN, alpha=0.6)
    ax.set_xticks(range(0, len(quotes.index), SCALE))
    ax.set_ylabel('Quote')
    ax.grid(True)

    bc = volume_overlay(ax2, quotes['open'], quotes['close'], quotes['volume'], colorup=COLORUP, colordown=COLORDOWN, width=0.5, alpha=0.6)
    ax2.add_collection(bc)
    ax2.set_xticks(range(0, len(quotes.index), SCALE))
    ax2.set_xticklabels(quotes.index[::SCALE], rotation=30)
    ax2.grid(True)
    plt.subplots_adjust(hspace=0.01)
    plt.show()


