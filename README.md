# QuantAI

QuantAI uses OpenAI Gym and Zipline to build reinforcement learning based trading algorithms.

---

## Installation

QuantAI requires zipline and keras (tensorflow backend). It is recommended to work within a conda installation to ensure the proper dependencies are installed. 

```sh
$ conda install -n QuantAI
$ source activate QuantAI
(QuantAI) $ conda install -c Quantopian zipline
(QuantAI) $ pip install gym
(QuantAI) $ zipline ingest -b Quantopian-quandl
(QuantAI) $ pip install tensorflow
(QuantAI) $ pip install scipy
(QuantAI) $ pip install scikit-learn
(QuantAI) $ pip install pillow
(QuantAI) $ pip install h5py
(QuantAI) $ pip install keras
```

Once your environment is set up, you will need to navigate to the source code for zipline and edit the following files.

### benchmarks.py
*../miniconda2/pkgs/zipline-1.0.2-np110py27_0/lib/python2.7/site-packages/zipline/data*
``` python

def get_benchmark_returns(symbol, start_date, end_date):
    print("getting benchmark returns")
    """
    Get a Series of benchmark returns from Google Finance.
    Returns a Series with returns from (start_date, end_date].
    start_date is **not** included because we need the close from day N - 1 to
    compute the returns for day N.
    """
    if symbol == "^GSPC":
        symbol = "spy"
    benchmark_frame = web.DataReader(symbol, 'google', start_date, end_date)
    return benchmark_frame["Close"].sort_index().tz_localize('UTC') \
        .pct_change(1).iloc[1:]
```

### tradesimulation.py
…/miniconda2/pkgs/zipline-1.0.2-np110py27_0/lib/python2.7/site-packages/zipline/gens
``` python
        def handle_benchmark(date, benchmark_source=self.benchmark_source):
            try:
                algo.perf_tracker.all_benchmark_returns[date] = \
                    benchmark_source.get_value(date)
            except:

                log.warning("error reading benchmark for date: " + str(date))
                yesterday = benchmark_source.get_value(date + pd.to_timedelta(-1, unit="d"))
                try:
                    algo.perf_tracker.all_benchmark_returns[date] = benchmark_source.get_value(yesterday)
                except:
                    algo.perf_tracker.all_benchmark_returns[date] = 1

```
---

## Interface

### Pipeline

Create pipeline functions to bring data into the algorithm. These functions must returns a pipeline object and the number of factors. To use existing zipline factors, add them in the import statement. You must add code to pipeline list.

pipeline file is located at *data/pipelines.py*

``` python
def sample_pipeline():

    price_filter = USEquityPricing.close.latest >= 5
    universe = price_filter

    columns = {}
    columns["Last"] = USEquityPricing.close.latest

    return Pipeline(columns = columns, screen = universe), len(columns)
```

Bring custom factors into the algorithm in *data/factors.py*

### Algorithm

**Initialize**

Method: TradingEnv
Arguments:
- code: pipeline_code from pipelines.py

``` python
my_algo = TradingEnv("sample_pipeline")
```


**Execute Trades (Optional)**

You have the option of overwriting the execute function.


Method: ExecuteTrades

NOTE: You must set context.shorts, context.longs and context.securities

``` python
def my_execute_trades(self, context, data, state, action):

    context.shorts = []
    context.longs = []
    context.securities = []

my_algo = TradingEnv("sample_pipeline")
my_algo.execute_trades = my_execute_trades

```

**Run Algorithm**
METHOD: run_algorithm
PARAMETERS
    - start_date
    - end_date
    - base_capital
    - reward:  possible values are "sharpe", "returns", "sortino"
    - name:    used to save model/backtest output files, 
            should also call backtest outupt file by this name
    - print_iter:  how often the algorithm should print
    - n_backtests: how many backtests the algo should train on

``` python
        algo.run_backtest("2002-1-1", "2016-1-1", 100000, name = "sample_pipeline")
```

---

## Backtest

``` sh
python sample_algo.py > backtests/your_backtest_name.txt
```




