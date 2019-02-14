from abc import ABCMeta, abstractmethod
from importlib import import_module
import inspect
import pandas as pd
from base.model.document import Stock, Future
from helper.args_parser import file_path
from helper.util import catch_exception
from helper.util import get_attribute
from helper.util import get_logger


import numpy as np

LOGGER = get_logger(__name__)


class ProcessTmpl(metaclass=ABCMeta):

    def __init__(self, data_source, state_codes, start_date, end_date, indicators, scaler):
        self._data_source = data_source
        self._state_codes = state_codes
        self._start_date = start_date
        self._end_date = end_date
        self._indicators = indicators

        self._scaler = scaler
        self._dates = list()
        self._origin_frame = dict()
        self._post_frames = dict()    # with analysis added
        self._scaled_frames = dict()

    @classmethod
    def get_instance(cls, data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs):
        cls = get_attribute(inspect.getmodule(cls).__name__ + '.Process' + data_source)
        return cls(data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs)

    def process(self):
        self._data_fetch()
        self._add_analysis()
        self._data_scale()
        self._post_process()

        return self._dates, self._scaled_frames

    @abstractmethod
    def _data_fetch(self):
        pass

    @abstractmethod
    def _data_scale(self):
        pass

    def _post_process(self):
        pass

    # def _ta_calculate(self):
    #     return IndicatorAnalysis(self._origin_frame, self._indicators).analyze()

    def _add_analysis(self):
        for state_code in self._state_codes:
            post_frame = IndicatorAnalysis(self._origin_frame[state_code], self._indicators).add_analysis()
            # dates = post_frame.iloc[:, 0]
            # self._dates = dates
            # instruments = post_frame.iloc[:, 1:post_frame.shape[1]]

            self._post_frames[state_code] = pd.DataFrame(data=post_frame.get_values(), index=self._dates.get_values().flatten(), columns= post_frame.columns)


class ProcessCSV(ProcessTmpl):

    # **kwargs: {filePath}
    def __init__(self, data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)  # set class attributes for CSV Processor

        self.file_path = file_path
        self._state_code = str(self.file_path.split('/')[-1:][0]).split('.')[0]
        super(ProcessCSV, self).__init__(data_source, state_codes, start_date, end_date, indicators, scaler)

    def _data_fetch(self):
        data_frame = pd.DataFrame(pd.read_csv(self.file_path))

        dates = data_frame.iloc[:, 0].get_values()

        data = data_frame.iloc[:, 1:data_frame.shape[1]]
        dates = pd.DataFrame(dates, index=dates, columns=['date'])

        df_ori = pd.DataFrame(data.get_values(), index=dates.get_values().flatten(), columns=data.columns)
        self._origin_frame[self._state_code] = df_ori.loc[self._start_date:self._end_date]
        self._dates = dates.loc[self._start_date:self._end_date]


    def _data_scale(self):
        scales = self._scaler
        post_frame = self._post_frames[self._state_code].copy()
        # instruments = post_frame.iloc[:, 1:post_frame.shape[1]]
        scales.fit(post_frame)
        instruments_scaled = scales.transform(post_frame)
        post_frame = instruments_scaled
        self._scaled_frames[self._state_code] = pd.DataFrame(data=post_frame, index=self._dates.get_values().flatten(), columns=self._post_frames[self._state_code].columns)

    def _post_process(self):

        # data_frame = self._origin_frame
        # data = data_frame.iloc[:, 1:data_frame.shape[1]]
        # df_ori = pd.DataFrame(data.get_values(), index=self._dates, columns=data.columns)

        state_code = self._state_code

        self._origin_frame[state_code] = self._origin_frame[state_code].dropna(axis=0)
        self._post_frames[state_code] = self._post_frames[state_code].dropna(axis=0)
        self._scaled_frames[state_code] = self._scaled_frames[state_code].dropna(axis=0)
        # df_dates = self._dates.loc[self._start_date:self._end_date]
        self._dates = list(self._scaled_frames[state_code].index)

        # self._scaled_frames[state_code] = self._scaled_frames[state_code].drop(['open', 'high', 'low'], axis=1)


class IndicatorAnalysis:

    def __init__(self, origin_frame, indicators):
        self._origin_frame = origin_frame
        self._indicators = indicators

    # def rsi(self, *args):
    #     para = args[0]
    #     result = talib.RSI(self._origin_frame[para[0]], int(para[1]))
    #     return pd.DataFrame(result, columns=['rsi_{}'.format(para[1])])
    #
    # def macd(self, *args):
    #     para = args[0]
    #     result = talib.MACD(self._origin_frame[para[0]], int(para[1]), int(para[2]), int(para[3]))
    #     return pd.DataFrame(result[0], columns=['macd'])

    @catch_exception(LOGGER)
    def analyze(self):
        df_indicators = pd.DataFrame()

        # instance = self.get_instance()
        for indicator in self._indicators:
            indicator = indicator.lower()
            meta_info = indicator.split('_')

            method_name = meta_info[1]

            method = getattr(self, method_name, None)
            if method is not None:
                del meta_info[1]
                if df_indicators.empty:
                    df_indicators = (method(meta_info))
                else:
                    df_indicators = df_indicators.join(method(meta_info))

            else:
                # try to get the method from talib
                method = get_attribute('.'.join(['talib', method_name.upper()]))
                args = list(map(int, meta_info[2: len(meta_info)]))  # convert from string to int
                result = method(self._origin_frame[meta_info[0]], *args)

                if isinstance(result, pd.core.series.Series):
                    df_result = pd.DataFrame(result, columns=[method_name])
                else:
                    for idx, res in enumerate(result):
                        if idx == 0:
                            df_result = pd.DataFrame(res, columns=[method_name+'_'+str(idx)])
                        else:
                            df_result = df_result.join(pd.DataFrame(res, columns=[method_name+'_'+str(idx)]))
                    # df_result = pd.DataFrame(result[0], columns=[method_name])

                if df_indicators.empty:
                    df_indicators = df_result
                else:
                    df_indicators = df_indicators.join(df_result)

                df_result = None # clean the data frame


        return df_indicators

    def add_analysis(self):
        return self._origin_frame.join(self.analyze())

    def get_instance(self):
        """ initialize a instance """
        ds_cls = get_attribute(
            inspect.__package__ + inspect.getmodulename(__file__) + '.{}'.format(self.__class__.__name__))
        return ds_cls

#
# class Util:
#     def get_attribute(kls):
#         """ Python version of Class.forName() in Java """
#         parts = kls.split('.')
#         module = ".".join(parts[:-1])
#
#         m = import_module(module)
#
#         return getattr(m, parts[len(parts) - 1])


class ProcessMongoDB(ProcessTmpl):

    def __init__(self, data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)  # set class attributes for MongoDB Processor

        super(ProcessMongoDB, self).__init__(data_source, state_codes, start_date, end_date, indicators, scaler)

    def _validate_codes(self):
        if not self.state_code_count:
            raise ValueError("Codes cannot be empty.")
        for code in self.state_codes:
            if not self.doc_class.exist_in_db(code):
                raise ValueError("Code: {} not exists in database.".format(code))

    def _data_fetch(self):
        # Remove invalid codes first.
        self._validate_codes()
        # Init columns and data set.
        columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
        # Load data.
        for index, code in enumerate(self.state_codes):
            # Load instrument docs by code.
            instrument_docs = Stock.get_k_data(code, self._start_date, self._end_date)
            # Init instrument dicts.
            instrument_dicts = [instrument.to_dic() for instrument in instrument_docs]
            # Split dates.
            dates = [instrument[1] for instrument in instrument_dicts]
            # Split instruments.
            instruments = [instrument[2:] for instrument in instrument_dicts]
            # Update dates set.
            dates_set = dates_set.union(dates)
            # Build origin and scaled frames.
            scaler = self._scaler
            scaler.fit(instruments)
            instruments_scaled = scaler.transform(instruments)
            origin_frame = pd.DataFrame(data=instruments, index=dates, columns=columns)
            scaled_frame = pd.DataFrame(data=instruments_scaled, index=dates, columns=columns)
            # Build code - frame map.
            self._origin_frame = origin_frame
            self._scaled_frames = scaled_frame

        # Init date iter.
        self.dates = sorted(list(dates_set))
        # Rebuild index.
        for code in self.state_codes:
            origin_frame = self.origin_frames[code]
            scaled_frame = self.scaled_frames[code]
            self.origin_frames[code] = origin_frame.reindex(self.dates, method='bfill')
            self.scaled_frames[code] = scaled_frame.reindex(self.dates, method='bfill')

    def _data_scale(self):
        pass


if __name__ == '__main__':
    # df = pd.DataFrame(np.random.random(100), columns=['close'])
    #
    # ta = IndicatorAnalysis(df, ['close_rsi_14', 'close_macd_12_26_9'])
    # print(ta.analyze())

    # kwargs = {"state_code_count": "2"}
    # posted_frame, scaled_frame = ProcessTmpl.get_instance('CSV', ['nasdaq'], None, None, ['close_rsi_14', 'close_macd_12_26_9'],
    #                                 'MongoDB', **kwargs).process()
    # print(posted_frame)
    # print(scaled_frame)

    m = import_module('talib')
    rsi = getattr(m, 'RSi')
    print(rsi)

    # pd.to_datetime()
