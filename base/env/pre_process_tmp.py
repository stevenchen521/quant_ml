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

# from sklearn.preprocessing import MinMaxScaler

LOGGER = get_logger(__name__)





""" fetch action """


def fetch_csv_single(self):
    state_code = str(file_path.split('/')[-1:][0]).split('.')[0]

    self._state_code = state_code

    data_frame = pd.DataFrame(pd.read_csv(file_path))

    dates = data_frame.iloc[:, 0].get_values()

    data = data_frame.iloc[:, 1:data_frame.shape[1]]
    dates = pd.DataFrame(dates, index=dates, columns=['date'])

    df_ori = pd.DataFrame(data.get_values(), index=dates.get_values().flatten(), columns=data.columns)
    self._origin_frame[state_code] = df_ori.loc[self._start_date:self._end_date]
    self._dates = dates.loc[self._start_date:self._end_date]



""" before process action """
def before_process(self):
    pass



""" analyze action """
def analyze(self, pre_frames):
    for state_code in self._state_codes:
        analyze_frames = IndicatorAnalysis(pre_frames[state_code], self._indicators).add_analysis()
        # dates = post_frame.iloc[:, 0]
        # self._dates = dates
        # instruments = post_frame.iloc[:, 1:post_frame.shape[1]]

        self._analyze_frames[state_code] = pd.DataFrame(data=analyze_frames.get_values(),
                                                        index=self._dates.get_values().flatten(),
                                                        columns=analyze_frames.columns)


""" after process action """
def after_process_csv_single(self):
    scales = self._scaler
    post_frame = self._post_frames[self._state_code].copy()
    scales.fit(post_frame)
    instruments_scaled = scales.transform(post_frame)
    post_frame = instruments_scaled
    self._scaled_frames[self._state_code] = pd.DataFrame(data=post_frame, index=self._dates.get_values().flatten(),
                                                         columns=self._post_frames[self._state_code].columns)

    state_code = self._state_code
    self._origin_frame[state_code] = self._origin_frame[state_code].dropna(axis=0)
    self._post_frames[state_code] = self._post_frames[state_code].dropna(axis=0)
    self._scaled_frames[state_code] = self._scaled_frames[state_code].dropna(axis=0)
    # df_dates = self._dates.loc[self._start_date:self._end_date]
    self._dates = list(self._scaled_frames[state_code].index)

    # self._scaled_frames[state_code] = self._scaled_frames[state_code].drop(['open', 'high', 'low'], axis=1)
