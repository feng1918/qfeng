#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sb
from draw import candlestick_vol


def gen_change_ratio(quotes, charts=True):
    """
    Generate ChangeRatio for quotes
    """
    changeRatio = quotes['close'].pct_change() * 100
    if charts:
        sb.distplot(changeRatio[1:])
    return changeRatio


def gen_amplitude(quotes, charts=True):
    """
    Generate Amplitude for quotes
    """
    amplitude = (quotes['high']-quotes['low']) / quotes['close'].shift(1) * 100
    if charts:
        sb.distplot(amplitude[1:])
    return amplitude


def gen_jump_powers(quotes, charts=True):
    """
    Generate JumpPowers for quotes
    """
    DAYS = 400
    if quotes.shape[0] < DAYS:
        jump_threshold = quotes['close'].median() * 0.03
    else:
        jump_threshold = quotes['close'][-1 * DAYS:].median() * 0.03
    dump_power = np.where(np.abs(quotes['low'] - quotes['close'].shift(1)) > jump_threshold,
                          (quotes['low'] - quotes['close'].shift(1)) / jump_threshold,
                          np.nan)
    if charts:
        # plot_bokeh_candle_dochl(quotes.index, quotes['open'].values, quotes['close'].values,
        #                         quotes['high'].values, quotes['low'].values, symbol='')
        candlestick_vol(quotes)
    return dump_power


def gen_smoothed(serie, win_len=20, charts=True):
    """
    Generate smoothed value for a serie
    """
    from scipy.signal import savgol_filter as smooth
    POLYORDER = 3
    smoothed_cls = pd.Series(smooth(serie, (win_len + 1), POLYORDER), serie.index)
    if charts:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 7))
        plt.plot(smoothed_cls, label='smoothed close', c='r')
        plt.plot(serie, label='close', c='b')
        plt.grid(True)
        plt.legend(loc='best')

    return smoothed_cls


def gen_cluster_1d(data_list, number_class):
    """
    Cluster one dimension data with Jenks Natural Breaks.
    https://stackoverflow.com/questions/28416408/scikit-learn-how-to-run-kmeans-on-a-one-dimensional-array
    """
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass


def gen_stational_points(origin, num_class=3, delta=0.003, charts=True):
    df = pd.DataFrame(origin.values, columns=['origin'])
    df['smoothed'] = gen_smoothed(df['origin'], charts=False)
    df['pre'] = df.smoothed.shift(1)
    df['aft'] = df.smoothed.shift(-1)
    df['der1'] = (df['aft'] - df['pre']) / df['smoothed']
    df['der2'] = (df['aft'] + df['pre'] - 2 * df['smoothed'])
    minimum = df[(df.der1.abs() < delta) & (df.der2 > 0)]
    maximum = df[(df.der1.abs() < delta) & (df.der2 < 0)]
    support = gen_cluster_1d(minimum['origin'].values, num_class)
    resistance = gen_cluster_1d(maximum['origin'].values, num_class)
    if charts:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 7))
        plt.plot(df['origin'], label='origin')
        plt.plot(df['smoothed'], label='smoothed')
        for i in support:
            plt.axhline(i, c='r', label=f'support {i}')
        for i in resistance:
            plt.axhline(i, c='y', label=f'resistance {i}')
        plt.grid(True)
        plt.legend(loc='best')
    return (support, resistance)


def gen_supres(serie, win_len=11, delta=1.1, charts=True):
    """
    This function takes a numpy array of close price
    and returns a list of support and resistance levels 
    respectively. n is the number of entries to be scanned.
    """
    ltp = serie
    # converting win_len to a nearest even number
    if win_len % 2 != 0:
        win_len += 1

    ltp_s = gen_smoothed(ltp, win_len, charts=False)
    n_ltp = ltp.shape[0]
    # taking a simple derivative
    ltp_d = np.zeros(n_ltp)
    ltp_d[1:] = np.subtract(ltp_s[1:], ltp_s[:-1])

    resistance = []
    support = []

    for i in range(n_ltp - win_len):
        arr_sl = ltp_d[i:(i + win_len)]
        first = arr_sl[:(win_len // 2)]  # first half
        last = arr_sl[(win_len // 2):]  # second half

        r_1 = np.sum(first > 0)
        r_2 = np.sum(last < 0)

        s_1 = np.sum(first < 0)
        s_2 = np.sum(last > 0)

        # local maxima detection
        if (r_1 == (win_len // 2)) and (r_2 == (win_len // 2)):
            resistance.append(ltp[i + ((win_len // 2) - 1)])

        # local minima detection
        if (s_1 == (win_len // 2)) and (s_2 == (win_len // 2)):
            support.append(ltp[i + ((win_len // 2) - 1)])

    # remove closed lines
    # support = support.sort()
    # cl_sup = []
    # n_sup = len(support)
    # i = 0
    # while i < n_sup:
    #     cl_sup.append(support[i])
    #     for x in range(i, n_sup):
    #         if support[i] * delta < support[x]:
    #             break
    #     i = x

    # resistance = resistance.sort()
    if charts is True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 7))
        plt.plot(ltp, label='close')
        plt.plot(ltp_s, label='smoothed close')
        for i in support:
            plt.axhline(i, c='r', label=f'support {i}')
        for i in resistance:
            plt.axhline(i, c='g', label=f'resistance {i}')
        plt.grid(True)
        plt.legend(loc='best')

    return support, resistance


def gen_linedata(x0, y0, x1, y1):
    """
    Generate slope and y-intercept for two points
    """
    slope = (y1 - y0) / (x1 - x0)
    intercept = (x1 * y0 - x0 * y1) / (x1 - x0)
    return (slope, intercept)


def gen_trends(x, window=1/3.0, charts=True):
    """
    Returns a Pandas dataframe with support and resistance lines.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    x = np.array(x)

    if window < 1:
        window = int(window * len(x))

    max1 = np.where(x == max(x))[0][0]  # find the index of the abs max
    min1 = np.where(x == min(x))[0][0]  # find the index of the abs min

    # First the max
    if max1 + window > len(x):
        max2 = max(x[0:(max1 - window)])
    else:
        max2 = max(x[(max1 + window):])

    # Now the min
    if min1 - window < 0:
        min2 = min(x[(min1 + window):])
    else:
        min2 = min(x[0:(min1 - window)])

    # Now find the indices of the secondary extrema
    max2 = np.where(x == max2)[0][0]  # find the index of the 2nd max
    min2 = np.where(x == min2)[0][0]  # find the index of the 2nd min

    # Create & extend the lines
    (max_slope, max_intercept) = gen_linedata(max2, x[max2], max1, x[max1])
    y_max = max_slope * len(x) + max_intercept  # y = a * x + b
    maxline = np.linspace(max_intercept, y_max, len(x))

    (min_slope, min_intercept) = gen_linedata(min2, x[min2], min1, x[min1])
    y_min = min_slope * len(x) + min_intercept
    minline = np.linspace(min_intercept, y_min, len(x)) 

    # OUTPUT
    trends = np.transpose(np.array((x, maxline, minline)))
    trends = pd.DataFrame(trends, index=np.arange(0, len(x)),
                          columns=['Data', 'Max Line', 'Min Line'])

    if charts is True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 7))
        plt.plot(trends)
        plt.scatter(max1, x[max1],  marker='o', c='g')
        plt.scatter(max2, x[max2],  marker='o', c='b')
        plt.scatter(min1, x[min1],  marker='x', c='r')
        plt.scatter(min2, x[min2],  marker='x', c='black')
        plt.axvline(window, linestyle='--')
        plt.grid(True)
        plt.show()

    return max_slope, min_slope


def gen_segtrends(x, segments=2, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np
    y = np.array(x)

    # Implement trendlines
    segments = int(segments)
    maxima = np.ones(segments)
    minima = np.ones(segments)
    segsize = int(len(y)/segments)
    for i in range(1, segments+1):
        ind2 = i*segsize
        ind1 = ind2 - segsize
        maxima[i-1] = max(y[ind1:ind2])
        minima[i-1] = min(y[ind1:ind2])

    # Find the indexes of these maxima in the data
    x_maxima = np.ones(segments)
    x_minima = np.ones(segments)
    for i in range(0, segments):
        x_maxima[i] = np.where(y == maxima[i])[0][0]
        x_minima[i] = np.where(y == minima[i])[0][0]

    if charts:
        import matplotlib.pyplot as plt
        plt.plot(y)
        plt.grid(True)

    for i in range(0, segments-1):
        maxslope = (maxima[i+1] - maxima[i]) / (x_maxima[i+1] - x_maxima[i])
        a_max = maxima[i] - (maxslope * x_maxima[i])
        b_max = maxima[i] + (maxslope * (len(y) - x_maxima[i]))
        maxline = np.linspace(a_max, b_max, len(y))

        minslope = (minima[i+1] - minima[i]) / (x_minima[i+1] - x_minima[i])
        a_min = minima[i] - (minslope * x_minima[i])
        b_min = minima[i] + (minslope * (len(y) - x_minima[i]))
        minline = np.linspace(a_min, b_min, len(y))

        if charts:
            plt.plot(maxline, 'g')
            plt.plot(minline, 'r')

    if charts:
        plt.show()

    # OUTPUT
    return x_maxima, maxima, x_minima, minima


def miniTrends(x, window=20, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np

    y = np.array(x)
    if window < 1:  # if window is given as fraction of data length
        window = float(window)
        window = int(window * len(y))
    x = np.arange(0, len(y))
    dy = y[window:] - y[:-window]
    crit = dy[:-1] * dy[1:] < 0

    # Find whether max's or min's
    maxi = (y[x[crit]] - y[x[crit] + window] > 0) & \
           (y[x[crit]] - y[x[crit] - window] > 0) * 1
    mini = (y[x[crit]] - y[x[crit] + window] < 0) & \
           (y[x[crit]] - y[x[crit] - window] < 0) * 1
    maxi = maxi.astype(float)
    mini = mini.astype(float)
    maxi[maxi == 0] = np.nan
    mini[mini == 0] = np.nan
    xmax = x[crit] * maxi
    xmax = xmax[~np.isnan(xmax)]
    xmax = xmax.astype(int)
    xmin = x[crit] * mini
    xmin = xmin[~np.isnan(xmin)]
    xmin = xmin.astype(int)

    # See if better max or min in region
    yMax = np.array([])
    xMax = np.array([])
    for i in xmax:
        indx = np.where(xmax == i)[0][0] + 1
        try:
            Y = y[i:xmax[indx]]
            yMax = np.append(yMax, Y.max())
            xMax = np.append(xMax, np.where(y == yMax[-1])[0][0])
        except Exception:
            pass
    yMin = np.array([])
    xMin = np.array([])
    for i in xmin:
        indx = np.where(xmin == i)[0][0] + 1
        try:
            Y = y[i:xmin[indx]]
            yMin = np.append(yMin, Y.min())
            xMin = np.append(xMin, np.where(y == yMin[-1])[0][0])
        except Exception:
            pass
    if y[-1] > yMax[-1]:
        yMax = np.append(yMax, y[-1])
        xMax = np.append(xMax, x[-1])
    if y[0] not in yMax:
        yMax = np.insert(yMax, 0, y[0])
        xMax = np.insert(xMax, 0, x[0])
    if y[-1] < yMin[-1]:
        yMin = np.append(yMin, y[-1])
        xMin = np.append(xMin, x[-1])
    if y[0] not in yMin:
        yMin = np.insert(yMin, 0, y[0])
        xMin = np.insert(xMin, 0, x[0])

    # Plot results if desired
    if charts is True:
        from matplotlib.pyplot import plot, show, grid
        plot(x, y)
        plot(xMax, yMax, '-o')
        plot(xMin, yMin, '-o')
        grid(True)
        show()
    # Return arrays of critical points
    return xMax, yMax, xMin, yMin


def iterlines(x, window=30, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.

    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np

    x = np.array(x)
    n = len(x)
    if window < 1:
        window = int(window * n)
    sigs = np.zeros(n, dtype=float)

    i = window
    while i != n:
        if x[i] > max(x[i-window:i]):
            sigs[i] = 1
        elif x[i] < min(x[i-window:i]):
            sigs[i] = -1
        i += 1

    xmin = np.where(sigs == -1.0)[0]
    xmax = np.where(sigs == 1.0)[0]
    ymin = x[xmin]
    ymax = x[xmax]
    if charts is True:
        from matplotlib.pyplot import plot, grid, show
        plot.figure(figsize=(18, 7))
        plot(x)
        plot(xmin, ymin, 'ro')
        plot(xmax, ymax, 'go')
        grid(True)
        show()

    return sigs
