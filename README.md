# QuantAI

QuantAI uses OpenAI Gym and Zipline to build reinforcement learning based trading algorithms.

---

## Installation

QuantAI requires zipline and OpenAI Gym. It is recommended to work within a conda installation to ensure the proper dependencies are installed. 

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