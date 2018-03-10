#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class Stock(object):
    """docstring for Stock
    Stock is a pandas.dataframe.
    Source must have these columns :
        - date: trade_date
        - paper: stock id
        - open: open price
        - close: close price
        - high: highest price
        - low: lowest price
        - volume: # of volume
    Some can be defined in source or calculated
        - changeRatio
        - amplitude
    """
    def __c_date(self, x):
        return [pd.datetime.strptime(str(i), '%Y%m%d') for i in x]

    def __init__(self, fname, delimiter=',', head=0, encoding='utf-8'):
        super(Stock, self).__init__()
        data = pd.read_csv(fname, delimiter=delimiter, header=0, encoding=encoding)
        data[['date']] = data[['quote_date']].apply(self.__c_date)
        data = data.set_index('date')
        data = data.sort_index()
        if 'changeRatio' not in data.columns:
            data = data.assign(changeRatio=data['close'].pct_change() * 100)
        if 'amplitude' not in data.columns:
            data = data.assign(amplitude=(data['high'] - data['low']) / data['close'].shift(1) * 100)
        self.data = data

    def get_data(self):
        """docstring for get_data"""
        return self.data
