

class State(object):

    def __init__(self):

        self.should_work

        self._market = None
        self.market = None

        self._returns = None
        self.returns = None

        self._risk = None
        self.risk = None

    def update_state(self, market, risk, returns):

        self._market = self.market
        self.market = market

        self._risk = self.risk
        self.risk = risk

        self._returns = self.returns
        self.returns = returns




