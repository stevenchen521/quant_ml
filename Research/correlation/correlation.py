from base.env.pre_process import Fetch, PreAnalyze, PostAnalyze
import pandas as pd
from Research.correlation.corr_config import active_config
import base.env.pre_process as pre_process
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import numpy as np

class FetchCSVResearch(Fetch):

    @staticmethod
    def fire(self):
        # file_path = get_attribute(active_stragery).get('source')
        file_path = active_config.get('source')
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


class PreAnalyzeResearch(PreAnalyze):

    @staticmethod
    def fire(self, origin_frames):
        self._pre_frames = origin_frames



class PostAnalyzeResearch(PostAnalyze):
    ### single csv data
    @staticmethod
    def fire(self, analyze_frames, col_order=None):
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



class CorrResearch(object):
    def __init__(self, code, config, start_date, end_date):
        self.code = code
        self.config = config
        self._start_date = start_date
        self._end_date = end_date
        self.now = datetime.datetime.now().strftime('20%y%m%d_%H_%M')
        # self._bt_strategy = bt_strategy_setup()
        # self._bt_datafeed = bt_datafeed_setup()


    @staticmethod
    def preprocess_for_backtest(df, add_col=None):
        df['date'] = df.index
        df['date'] = df['date'].apply(
            lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
        df['openinterest'] = 0
        basic_col = ['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
        if add_col:
            col_order = basic_col + add_col
        else:
            col_order = basic_col
        df = df[col_order]
        df.columns = [x.lower() for x in list(df.columns) if x in basic_col] + add_col
        df.dropna(how="any", inplace=True)
        df.index = range(len(df))
        # df.set_index(['date'], inplace=True
        return df


    def format_postframe(self):
        dates, a, b, post_frames = \
            pre_process.ProcessStrategy(  # action_fetch, action_pre_analyze, action_analyze, action_post_analyze,
                self.code, self._start_date, self._end_date, MinMaxScaler(), self.config).process()
        self.post_frames = post_frames.get(self.code[0])


    def generate_correlation_graph(self):
        self.format_postframe()
        self.corr = self.post_frames.corr()
        # mask = np.zeros_like(self.corr)
        # mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(25, 20))
        sns.set(font_scale=1.4)
        ax = sns.heatmap(self.corr, annot=True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        # plt.show()
        name = self.config.get("name")
        path = "corr_pict/{}_{}".format(name, self.now)
        # fig = ax.get_figure()
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    D = CorrResearch(code=['SH_index_all'],
                     config=active_config,
                     start_date="2008-01-01",
                     end_date="2019-02-01")
    D.generate_correlation_graph()



