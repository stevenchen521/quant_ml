from rqalpha import run_file

config = {
  "base": {
    "data_bundle_path": "D:\\rqalpha\data_bundle\\bundle",
    "start_date": "2017-06-01",
    "end_date": "2019-06-10",
    "benchmark": "000300.XSHG",
    "accounts": {
        "stock": 1000000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": True,
      "output-file": "bolling_bond.pkl"
    }
  }
}

strategy_file_path = "./bolling_bond.py"
run_file(strategy_file_path, config)

