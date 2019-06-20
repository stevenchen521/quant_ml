import pandas as pd
import tushare as ts

def pickle_to_excel(pickle_path, excel_path = None):
    if excel_path == None:
        excel_path = pickle_path.replace('.pkl', '.xlsx')
    else:
        pass
    writer = pd.ExcelWriter(excel_path)
    p1 = pd.read_pickle(pickle_path)
    p1['benchmark_portfolio'].to_excel(writer, sheet_name="benchmark_portfolio")
    if 'plots' in p1.keys():
        p1['plots'].to_excel(writer, sheet_name="plots")
    else:
        pass
    p1['portfolio'].to_excel(writer, sheet_name="portfolio")
    p1['stock_account'].to_excel(writer, sheet_name="stock_account")
    p1['stock_positions'].to_excel(writer, sheet_name="stock_positions")
    # .to_excel(writer, sheet_name="QgrC_top")
    # for key, val in p1.summary():
    #     writer.writerow([key, val])
    pd.DataFrame.from_dict(p1['summary'], orient='index').to_excel(writer, sheet_name="performance")
    p1['trades'].to_excel(writer, sheet_name="trades")
    writer.save()


import six
import tushare as ts
from datetime import date
from dateutil.relativedelta import relativedelta
from rqalpha.data.base_data_source import BaseDataSource



class TushareKDataSource(BaseDataSource):
    def __init__(self, path):
        super(TushareKDataSource, self).__init__(path)

    @staticmethod
    def get_tushare_k_data(instrument, start_dt, end_dt):
        order_book_id = instrument.order_book_id
        code = order_book_id.split(".")[0]

        if instrument.type == 'CS':
            index = False
        elif instrument.type == 'INDX':
            index = True
        else:
            return None

        return ts.get_k_data(code, index=index, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'))

    def get_bar(self, instrument, dt, frequency):
        if frequency != '1d':
            return super(TushareKDataSource, self).get_bar(instrument, dt, frequency)

        bar_data = self.get_tushare_k_data(instrument, dt, dt)

        if bar_data is None or bar_data.empty:
            return super(TushareKDataSource, self).get_bar(instrument, dt, frequency)
        else:
            return bar_data.iloc[0].to_dict()

    def history_bars(self, instrument, bar_count, frequency, fields, dt, skip_suspended=True):
        if frequency != '1d' or not skip_suspended:
            return super(TushareKDataSource, self).history_bars(instrument, bar_count, frequency, fields, dt, skip_suspended)

        start_dt_loc = self.get_trading_calendar().get_loc(dt.replace(hour=0, minute=0, second=0, microsecond=0)) - bar_count + 1
        start_dt = self.get_trading_calendar()[start_dt_loc]

        bar_data = self.get_tushare_k_data(instrument, start_dt, dt)

        if bar_data is None or bar_data.empty:
            return super(TushareKDataSource, self).get_bar(instrument, dt, frequency)
        else:
            if isinstance(fields, six.string_types):
                fields = [fields]
            fields = [field for field in fields if field in bar_data.columns]

            return bar_data[fields].as_matrix()

    def available_data_range(self, frequency):
        return date(2005, 1, 1), date.today() - relativedelta(days=1)

