import tushare as ts
import pandas as pd


class SHSZData(object):
    """docstring for SHSZData"""
    def __init__(self, data_folder):
        super(SHSZData, self).__init__()
        stocks = ts.get_stock_basics()
        self.stocks = stocks[stocks['timeToMarket'] != 0]
        self.DATA_FOLDER = data_folder

    def download_d_all(self):
        """Download all data"""
        for code, row in self.stocks.iterrows():
            time_to_market = str(row['timeToMarket'])
            start = "{}-{}-{}".format(time_to_market[:4], time_to_market[4:6], time_to_market[6:8])
            self.download_d(code, start=start)

    def download_d(self, code, start='2000-01-01', end='2023-01-01'):
        """docstring for download"""
        print(f"Downloading {code}...")
        df = ts.get_k_data(code, start=start, end=end)
        df.to_csv(f"{self.DATA_FOLDER}/{code}.csv", index=False)

    def retry_d(self):
        from pathlib import Path
        for code, row in self.stocks.iterrows():
            csv_f = Path(f"{self.DATA_FOLDER}/{code}.csv")
            if not csv_f.exists():
                time_to_market = str(row['timeToMarket'])
                start = "{}-{}-{}".format(time_to_market[:4], time_to_market[4:6], time_to_market[6:8])
                self.download_d(code, start=start)

    def update_d_all(self):
        """docstring for update_d_all"""
        for code, row in self.stocks.iterrows():
            self.update_d(code)

    def update_d(self, code):
        """docstring for update_d"""
        print(f"Updating {code}...")
        old_df = self.get_d(code)
        str_latest_date = old_df.iloc[-1]['date']
        start = pd.to_datetime(str_latest_date, format='%Y-%m-%d') + pd.DateOffset(1)
        new_df = ts.get_k_data(code, start.strftime('%Y-%m-%d'))
        new_df = new_df.sort_values(by='date')
        df = old_df.append(new_df, ignore_index=True)
        df.to_csv(f"{self.DATA_FOLDER}/{code}.csv", index=False)

    def get_d(self, code):
        """docstring for read_d"""
        df = pd.read_csv(f'{self.DATA_FOLDER}/{code}.csv', delimiter=',', header=0)
        df = df.sort_values(by='date')
        return df

    def get_basic(self, code):
        """docstring for get_name"""
        return self.stocks.loc[code]


class SHSZSelection(object):
    """docstring for SHSZSelection"""
    def __init__(self, data_folder, selection_func, equities=None):
        super(SHSZSelection, self).__init__()
        self.selection_func = selection_func
        self.DATA_FOLDER = data_folder
        stocks = ts.get_stock_basics()
        stocks = stocks[stocks['timeToMarket'] != 0]
        self.equities = equities if equities else stocks.index.values
    
    def run(self):
        results = None
        for e in self.equities:
            df = pd.read_csv(f'{self.DATA_FOLDER}/{e}.csv', delimiter=',', header=0)
            df = df.sort_values(by='date')
            rs = self.selection_func(e, df)
            results = pd.concat([results, rs])
        return results
