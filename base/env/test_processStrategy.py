from unittest import TestCase

import base.env.pre_process as pre_process
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper.util import get_attribute
from base.env.pre_process_conf import active_stragery, get_strategy_analyze
import base.env.pre_process


class TestProcessStrategy(TestCase):

    def test_get_active_strategy(self):
        self.action_fetch, self.action_pre_analyze, self.indicators, self.action_post_analyze, self._label = \
            pre_process.get_active_strategy(strategy=None)

        self.assertIsNotNone(self.action_fetch)
        self.assertIsNotNone(self.action_pre_analyze)
        self.assertIsNotNone(self.indicators)
        self.assertIsNotNone(self.action_post_analyze)

    def test_process(self):
        self.test_get_active_strategy()
        dates, pre_frames, origin_frames, post_frames = pre_process.ProcessStrategy(
                                                        # self.action_fetch, self.action_pre_analyze,self.indicators, self.action_post_analyze,
                                               ['SH_index'], "2008-01-01", "2019-02-01", MinMaxScaler(), active_stragery).process()
        result = post_frames['nasdaq'].dropna()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotEqual(result.values.size, 0)

    def test_palyaround(self):
        # action_post_analyze = get_attribute('.'.join([active_stragery.get('module'), 'PreAnalyzeDefault']))
        # action_post_analyze.fire(None, None)

        pre_process.PreAnalyzeDefault.fire(None, None)


        # print(class_post_analyze.fire(None, None))


    def test_get_strategy_analyze(self):

        self.assertIsNotNone(get_strategy_analyze(get_attribute(active_stragery)))
