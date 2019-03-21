
import pandas as pd
from base.env.pre_process import Fetch, PreAnalyze, PostAnalyze
from base.env.pre_process_conf import get_folder


strategy_data_mining = {
    'name':'data_mining',
    'module': 'back_testing.data_mining.dm_pre_process',
    'source': "{}/../../data/SH_index.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingleDM',
    'pre_analyze': 'PreAnalyzeDataMining',
    'analyze': [
                'trend|close|3_3_6',
                ],
    'post_analyze': 'PostAnalyzeDataMining',
    'label': 'trend_3_5_20'
}


class FetchCSVSingleDM(Fetch):

    @staticmethod
    def fire(self):
        # file_path = get_attribute(active_stragery).get('source')
        file_path = strategy_data_mining.get('source')
        state_code = str(file_path.split('/')[-1:][0]).split('.')[0]

        self._state_code = state_code

        data_frame = pd.DataFrame(pd.read_csv(file_path))

        dates = data_frame.iloc[:, 0].get_values()

        data = data_frame.iloc[:, 1:data_frame.shape[1]]
        dates = pd.DataFrame(dates, index=dates, columns=['date'])

        df_ori = pd.DataFrame(data.get_values(), index=dates.get_values().flatten(), columns=data.columns)

        mask = (df_ori.index > self._start_date) & (df_ori.index <= self._end_date)
        self._origin_frames[state_code] = df_ori.loc[mask]
        self._dates = dates.loc[mask]


class PreAnalyzeDataMining(PreAnalyze):

    @staticmethod
    def fire(self, origin_frames):
        self._pre_frames = origin_frames



class PostAnalyzeDataMining(PostAnalyze):
    ### single csv data
    @staticmethod
    def fire(self, analyze_frames):
        # print("this is PostAnalyzeDefault")
        # scales = self._scaler
        # for state_code in self._state_codes:
        #     self._analyze_frames[state_code].to_csv("../../back_testing/data/{}.csv".format(state_code))
        post_frame = analyze_frames[self._state_code].copy()
        # scales.fit(post_frame)
        # instruments_scaled = scales.transform(post_frame)
        # post_frame = instruments_scaled
        self._post_frames[self._state_code] = pd.DataFrame(data=post_frame, index=self._dates.get_values().flatten(),
                                                           columns=analyze_frames[self._state_code].columns)

        state_code = self._state_code
        self._origin_frames[state_code] = self._origin_frames[state_code].dropna(axis=0)
        self._post_frames[state_code] = self._post_frames[state_code].dropna(axis=0)
        # self._scaled_frames[state_code] = self._scaled_frames[state_code].dropna(axis=0)
        # df_dates = self._dates.loc[self._start_date:self._end_date]
        self._dates = list(self._post_frames[state_code].index)

