from unittest import TestCase

from back_testing.automation import Automation, input_strategy_setup, market_setup_sl, algorithm_setup_da_rnn, bt_strategy_setup
from base.env.pre_process_conf import strategy_SH_index

class TestAutomation(TestCase):


    def test_automation(self):
        auto = Automation(input_strategy_setup=input_strategy_setup, input_strategy=strategy_SH_index, market_setup=market_setup_sl,
                   algorithm_setup=algorithm_setup_da_rnn, bt_strategy_setup=bt_strategy_setup)
        auto.process()


    def test_input_strategy_setup(self):
        input_strategy_setup()


