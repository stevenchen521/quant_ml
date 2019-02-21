import inspect
import pandas as pd
from abc import ABCMeta, abstractmethod
# from base.model.document import Stock, Future
from base.env.pre_process_conf import file_path
# from base.env.pre_process_setting import analysis
from base.env.pre_process_conf import active_stragery
from helper.util import catch_exception
from helper.util import get_attribute
from helper.util import get_logger

import talib

# from sklearn.preprocessing import MinMaxScaler

LOGGER = get_logger(__name__)


class Action(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def fire(self, data_frame):
        raise NotImplementedError


class Fetch(Action):

    @staticmethod
    @abstractmethod
    def fire(self, data_frame=None):
        raise NotImplementedError


class PreAnalyze(Action):

    @staticmethod
    @abstractmethod
    def fire(self, origin_frame):
        print('this is pre analyze')
        # raise NotImplementedError


class Analyze(Action):

    @staticmethod
    @abstractmethod
    def fire(self, pre_frame):
        for state_code in self._state_codes:
            analyze_frames = IndicatorAnalysis(pre_frame[state_code], self._indicators).add_analysis()
            # dates = post_frame.iloc[:, 0]
            # self._dates = dates
            # instruments = post_frame.iloc[:, 1:post_frame.shape[1]]
            self._analyze_frames[state_code] = pd.DataFrame(data=analyze_frames.get_values(),
                                                            index=self._dates.get_values().flatten(),
                                                            columns=analyze_frames.columns)


class PostAnalyze(Action):

    @staticmethod
    @abstractmethod
    def fire(self, data_frame):
        raise NotImplementedError
        # print("this is PostAnalyze")
        # for state_code in self._state_codes:
        #     self._analyze_frames[state_code].to_csv("../../back_testing/data/{}.csv".format(state_code))



class FetchCSVSingle(Fetch):

    @staticmethod
    def fire(self):
        state_code = str(file_path.split('/')[-1:][0]).split('.')[0]

        self._state_code = state_code

        data_frame = pd.DataFrame(pd.read_csv(file_path))

        dates = data_frame.iloc[:, 0].get_values()

        data = data_frame.iloc[:, 1:data_frame.shape[1]]
        dates = pd.DataFrame(dates, index=dates, columns=['date'])

        df_ori = pd.DataFrame(data.get_values(), index=dates.get_values().flatten(), columns=data.columns)
        self._origin_frames[state_code] = df_ori.loc[self._start_date:self._end_date]
        self._dates = dates.loc[self._start_date:self._end_date]


class PreAnalyzeDefault(PreAnalyze):

    @staticmethod
    def fire(self, origin_frames):
        self._pre_frames = origin_frames


class PostAnalyzeDefault(PostAnalyze):

    @staticmethod
    def fire(self, analyze_frames):
        # print("this is PostAnalyzeDefault")
        scales = self._scaler
        for state_code in self._state_codes:
            self._analyze_frames[state_code].to_csv("../../back_testing/data/{}.csv".format(state_code))
        post_frame = analyze_frames[self._state_code].copy()
        scales.fit(post_frame)
        instruments_scaled = scales.transform(post_frame)
        post_frame = instruments_scaled
        self._post_frames[self._state_code] = pd.DataFrame(data=post_frame, index=self._dates.get_values().flatten(),
                                                           columns=analyze_frames[self._state_code].columns)

        state_code = self._state_code
        self._origin_frames[state_code] = self._origin_frames[state_code].dropna(axis=0)
        self._post_frames[state_code] = self._post_frames[state_code].dropna(axis=0)
        # self._scaled_frames[state_code] = self._scaled_frames[state_code].dropna(axis=0)
        # df_dates = self._dates.loc[self._start_date:self._end_date]
        self._dates = list(self._post_frames[state_code].index)


def get_active_strategy():
    module = active_stragery.get('module')
    fetch = active_stragery.get('fetch')
    pre_analyze = active_stragery.get('pre_analyze')
    analyze = active_stragery.get('analyze')
    post_analyze = active_stragery.get('post_analyze')

    action_fetch = get_attribute('.'.join([module, fetch]))
    action_pre_analyze = get_attribute('.'.join([module, pre_analyze]))
    # analyze = get_attribute('.'.join([module, active_stragery.get('analyze')]))
    indicators = analyze
    action_post_analyze = get_attribute('.'.join([module, post_analyze]))
    return action_fetch, action_pre_analyze, indicators, action_post_analyze


class ProcessStrategy(object):

    def __init__(self, fetch, pre_analyze, indicators, post_analyze,
                 state_codes, start_date, end_date, scaler):
        self._fetch = fetch
        self._pre_analyze = pre_analyze
        self._analyze = Analyze # the analyze function in this module
        self._post_analyze = post_analyze

        # self._data_source = data_source
        self._state_codes = state_codes
        self._start_date = start_date
        self._end_date = end_date
        self._indicators = indicators

        self._scaler = scaler
        self._dates = list()
        self._origin_frames = dict()
        self._pre_frames = dict()
        self._analyze_frames = dict()   # with analysis added
        self._post_frames = dict()
        # self._scaled_frames = dict()

    @classmethod
    def get_instance(cls, data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs):
        cls = get_attribute(inspect.getmodule(cls).__name__ + '.Process' + data_source)
        return cls(data_source, state_codes, start_date, end_date, indicators, scaler, **kwargs)

    def process(self):
        self._fetch.fire(self)
        self._pre_analyze.fire(self, self._origin_frames)
        self._analyze.fire(self, self._pre_frames)
        self._post_analyze.fire(self, self._analyze_frames)

        return self._dates, self._post_frames


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

    def trend(self, *args):
        """  If closing price value leads its MA 15 and MA 15 is rising for last 5 days then trend is Uptrend
        i.e. trend signal is 1.

        If closing price value lags its MA 15 and MA 15 is falling for last 5 days then trend is Downtrend
        i.e. trend signal is 0.

        For up trend:
        Tr_i = [(cp_i - min cp)/(max cp - min cp)] * 0.5 + 0.5

        For down trend:
        Tr_i = [(cp_i - min cp)/(max cp - min cp)] * 0.5

        min cp = min(cp_i, cp_i+1, cp_i+2)
        max cp = max(cp_i, cp_i+1, cp_i+2)

        """

        TREND_DOWN = -1
        TREND_NO = 0
        TREND_UP = 1

        def determine_trend_ma(targets, trend_bars_idx, current_val):
            # determine the trend based on the move average.
            # e.x. if the target falling in last 5(trend_bars_idx) days and current value lower than mv, trend is down
            latest_trend = None
            for idx in range(trend_bars_idx):

                # if trend_bars_idx - idx - 2 == 0:break

                # if the current target is larger than the previous one
                if targets[trend_bars_idx - idx] >= targets[trend_bars_idx - idx - 1]:
                    trend = TREND_UP
                    if latest_trend == TREND_DOWN:
                        return TREND_NO
                    latest_trend = trend
                if targets[trend_bars_idx - idx] < targets[trend_bars_idx - idx - 1]:
                    trend = TREND_DOWN
                    if latest_trend == TREND_UP:
                        return TREND_NO
                    latest_trend = trend

            if trend == TREND_UP and current_val < targets[trend_bars_idx]:
                return TREND_NO
            elif trend == TREND_DOWN and current_val > targets[trend_bars_idx]:
                return TREND_NO

            return trend

        def calculate_up_trend(current_val, target_future_bars):
            if max(target_future_bars) == min(target_future_bars):
                breakpoint()
            return ((current_val - min(target_future_bars)) / (max(target_future_bars) - min(target_future_bars))) * 0.5 + 0.5

        def calculate_down_trend(current_val, target_future_bars):
            return ((current_val - min(target_future_bars)) / (max(target_future_bars) - min(target_future_bars))) * 0.5
        
        # we calculate the trend
        para = args[0]
        src_column = para[0]
        ma_bars = int(para[1])
        trend_bars = int(para[2])
        future_bars = int(para[3])
        target = self._origin_frame[src_column].copy()
        result = target.copy()

        # get moving average
        ma = talib.MA(target, timeperiod=ma_bars)

        last_trend = None
        for curr_idx, val in enumerate(target):
            result[curr_idx] = None
            if curr_idx >= ma_bars + trend_bars-1 and curr_idx < (len(target)-future_bars):
                target_trend_bars = ma[curr_idx-trend_bars: curr_idx+1]

                # determine the trend based on the move average.
                # e.x. if falling in last 5 days, trend is down
                ma_trend = determine_trend_ma(target_trend_bars, trend_bars, val)

                # if trend is down and price is lower than the ma we calculate trend with down formula
                if ma_trend == TREND_DOWN:
                    last_trend = TREND_DOWN
                    result[curr_idx] = calculate_down_trend(val, target[curr_idx: curr_idx+future_bars])

                # if trend is up and price is higher than the ma we calculate trend with up formula
                elif ma_trend == TREND_UP:
                    last_trend = TREND_UP
                    result[curr_idx] = calculate_up_trend(val, target[curr_idx: curr_idx+future_bars])
                elif ma_trend == TREND_NO:
                    # if have no trend, we get the last trend and calculate the trend
                    if last_trend == TREND_DOWN:
                        result[curr_idx] = calculate_down_trend(val, target[curr_idx: curr_idx + future_bars])
                    elif last_trend == TREND_UP:
                        result[curr_idx] = calculate_up_trend(val, target[curr_idx: curr_idx + future_bars])

        return pd.DataFrame(result).rename(columns={'close': 'trend_{}'.format(para[1:4])})
        # return pd.DataFrame(result, columns=['trend_{}'.format(para[1:4])])


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
                result = method(meta_info)
                if df_indicators.empty:
                    df_indicators = result if result is not None else df_indicators
                else:
                    df_indicators = df_indicators.join(result) if result is not None else df_indicators

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
                            df_result = pd.DataFrame(res, columns=[method_name + '_' + str(idx)])
                        else:
                            df_result = df_result.join(pd.DataFrame(res, columns=[method_name + '_' + str(idx)]))
                    # df_result = pd.DataFrame(result[0], columns=[method_name])

                if df_indicators.empty:
                    df_indicators = df_result
                else:
                    df_indicators = df_indicators.join(df_result)

                df_result = None  # clean the data frame

        return df_indicators

    def add_analysis(self):
        return self._origin_frame.join(self.analyze())

    def get_instance(self):
        """ initialize a instance """
        ds_cls = get_attribute(
            inspect.__package__ + inspect.getmodulename(__file__) + '.{}'.format(self.__class__.__name__))
        return ds_cls

if __name__ == '__main__':
    IndicatorAnalysis1 = IndicatorAnalysis()
    IndicatorAnalysis1.trend()