import numpy as np

__RISK_FREE_RATE_DAILY__ = .0002


def sharpe_ratio(context):

	returns = np.array(context.returns)

	if not np.count_nonzero(returns):
		return 0

	mean = np.mean(returns) - __RISK_FREE_RATE_DAILY__
	std = np.std(returns)

	return (mean / std)*np.sqrt(252) if std > 0 else 0

def sortino_ratio(context):

	returns = np.array(context.returns)

	if not np.count_nonzero(returns):
		return 0
	
	negative_returns = np.array([r for r in returns if r < 0])

	mean = np.mean(returns) - __RISK_FREE_RATE_DAILY__
	std = np.std(negative_returns)

	return (mean / std)*np.sqrt(252) if std > 0 else 0

def average_daily_returns(context):

	returns = np.array(context.returns)

	if not np.count_nonzero(returns):
		return 0

	return np.mean(returns)

def total_returns(context):

	return context.portfolio.returns * 100

def portfolio_value(context):

	return context.portfolio.portfolio_value




__REWARDS_LIST__ = {
	
	"sharpe": sharpe_ratio,
	"sortino": sortino_ratio,
	"average_daily_returns": average_daily_returns,
	"total_returns": total_returns,
	"portfolio_value": portfolio_value


}


def get_reward(code):

	return __REWARDS_LIST__[code]


