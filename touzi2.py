import pandas as pd
import numpy as np
import datetime, time
from dateutil.relativedelta import relativedelta
import talib
import math, random
from scipy.stats import ttest_ind
from scipy.stats import levene
from sklearn.linear_model import LinearRegression

def init(context):

	# 最大持股数量(大小盘各一半)
	context.hold_max = 40

	# 初始因子池（包括技术因子和财务因子在内的共计107个初始因子）
	context.need=['factor.pe',# 市盈率
	'factor.pb',# 市净率
	'factor.pcf_cash_flow_ttm',# 市现率PCF
	'factor.ps',# 市销率
	'factor.dividend_rate',# 股息率
	'factor.market_cap',# 总市值
	'factor.current_market_cap',# 流通市值
	'factor.capitalization',# 总股本
	'factor.circulating_cap',# 流通股本
	'factor.current_ratio',# 流动比率
	'factor.equity_ratio',# 产权比率负债合计／归属母公司股东的权益
	'factor.quick_ratio',# 速动比率
	'factor.tangible_assets_liabilities',# 有形资产／负债合计
	'factor.tangible_assets_int_liabilities',# 有形资产／带息债务
	'factor.net_debt_equity',# 净债务／股权价值
	'factor.long_term_debt_to_opt_capital_ratio',# 长期债务与营运资金比率
	'factor.tangible_assets_net_liabilities',# 有形资产／净债务
	'factor.overall_income_growth_ratio',# 营业总收入同比增长率
	# 'factor.net_cashflow_psg_rowth_ratio',# 每股经营活动产生的现金流量净额同比增长率(报错，原因询问客服ing)
	# 'factor.opt_profit_grow_ratio',# 营业利润同比增长率
	# 'factor.total_profit_growth_ratio',# 利润总额同比增长率
	'factor.diluted_net_asset_growth_ratio',# 净资产收益率摊薄同比增长率
	'factor.net_cashflow_from_opt_act_growth_ratio',# 经营活动产生的现金流量净额同比增长率
	'factor.net_profit_growth_ratio',# 净利润同比增长率
	# 'factor.basic_pey_ear_growth_ratio',# 基本每股收益同比增长率
	'factor.turnover_of_overall_assets',# 总资产周转率
	'factor.turnover_ratio_of_account_payable',# 应付账款周转率
	'factor.turnover_of_current_assets',# 流动资产周转率
	'factor.turnover_of_fixed_assets',# 固定资产周转率
	'factor.cash_cycle',# 现金循环周期
	'factor.inventory_turnover_ratio',# 存货周转率
	'factor.turnover_ratio_of_receivable',# 应收账款周转率
	'factor.weighted_roe',# 净资产收益率roe加权
	'factor.overall_assets_net_income_ratio',# 总资产净利率roa
	'factor.net_profit_margin_on_sales',# 销售净利率
	'factor.before_tax_profit_div_income',# 息税前利润／营业总收入
	'factor.sale_cost_div_income',# 销售费用／营业总收入
	'factor.roa',# 总资产报酬率roa
	'factor.ratio_of_sales_to_cost',# 销售成本率
	'factor.net_profit_div_income',# 净利润／营业总收入
	'factor.opt_profit_div_income',# 营业利润／营业总收入
	'factor.opt_cost_div_income',# 营业总成本／营业总收入
	'factor.administration_cost_div_income',# 管理费用／营业总收入
	'factor.financing_cost_div_income',# 财务费用／营业总收入
	'factor.impairment_loss_div_income',# 资产减值损失／营业总收
	# 以下为技术因子
	'factor.vr_rate',# 成交量比率
	'factor.vstd',# 成交量标准差
	'factor.arbr',# 人气意愿指标
	'factor.srdm',# 动向速度比率
	'factor.vroc',# 量变动速率
	'factor.vrsi',# 量相对强弱指标
	'factor.cr',# 能量指标
	'factor.mfi',# 资金流向指标
	'factor.vr',# 量比
	'factor.mass',# 梅丝线
	'factor.obv',# 能量潮
	'factor.pvt',# 量价趋势指标
	'factor.wad',# 威廉聚散指标
	'factor.bbi',# 多空指数
	'factor.mtm',# 动力指标
	'factor.dma',# 平均线差
	'factor.ma',# 简单移动平均
	'factor.macd',# 指数平滑异同平均
	'factor.expma',# 指数平均数
	'factor.priceosc',# 价格振荡指标
	'factor.trix',# 三重指数平滑平均
	'factor.dbcd',# 异同离差乖离率
	'factor.dpo',# 区间震荡线
	'factor.psy',# 心理指标
	'factor.vma',# 量简单移动平均
	'factor.vmacd',# 量指数平滑异同平均
	'factor.vosc',# 成交量震荡
	'factor.tapi',# 加权指数成交值
	'factor.micd',# 异同离差动力指数
	'factor.rccd',# 异同离差变化率指数
	'factor.ddi',# 方向标准差偏离指数
	'factor.bias',# 乖离率
	'factor.cci',# 顺势指标
	'factor.kdj',# 随机指标
	'factor.lwr',# L威廉指标
	'factor.roc',# 变动速率
	'factor.rsi',# 相对强弱指标
	'factor.si',# 摆动指标
	'factor.wr',# 威廉指标
	'factor.wvad',# 威廉变异离散量
	'factor.bbiboll',# BBI多空布林线
	'factor.cdp',# 逆势操作
	'factor.env',# ENV指标
	'factor.mike',# 麦克指标
	'factor.adtm',# 动态买卖气指标
	'factor.mi',# 动量指标
	'factor.rc',# 变化率指数
	'factor.srmi',# SRMIMI修正指标
	'factor.dptb',# 大盘同步指标
	'factor.jdqs',# 阶段强势指标
	'factor.jdrs',# 阶段弱势指标
	'factor.zdzb',# 筑底指标
	'factor.atr',# 真实波幅
	'factor.std',# 标准差
	'factor.vhf',# 纵横指标
	'factor.cvlt']# 佳庆离散指标

	# 最大因子数目
	context.need_max = 10

	# 上一次调仓期
	context.last_date = ''

	# 设置调仓周期，每月倒数第一个交易日运行
	run_monthly(func = reallocate, date_rule = -1)

	# 设置因子测试周期为12个月
	context.need_tmonth = 12

	# 选择沪深300
	get_iwencai('沪深300', 'hs_300')

	# 选择中证500
	get_iwencai('中证500', 'zz_500')

def reallocate(context, bar_dict):
	# 获取上一个交易日的日期
	date = get_last_datetime().strftime('%Y%m%d')
	log.info('上个交易日日期为：' + date)

	# 获取上个月末调仓日期
	context.last_date = func_get_end_date_of_last_month(date)

	log.info('上月月末调仓日期为：' + context.last_date)

	hs_needs, zz_needs = get_needs(context, bar_dict, date, context.hs_300), get_needs(context, bar_dict, date, context.zz_500)
	# log.info('这是沪深300股票池的因子：')
	# log.info(hs_needs)

	# log.info('这是中证500股票池的因子：')
	# log.info(zz_needs)
	# 用来存放大小盘股票，0是小盘，1是大盘
	tempstocks = [[], []]
	tempstocks[0].append(get_stocks(context, bar_dict, zz_needs, date, context.zz_500))
	tempstocks[1].append(get_stocks(context, bar_dict, hs_needs, date, context.hs_300))
	'''
	tempstocks = []
	tempstocks.append(get_stocks(context, bar_dict, zz_needs, date, context.zz_500))
	tempstocks.append(get_stocks(context, bar_dict, hs_needs, date, context.hs_300))
	log.info(tempstocks)

	func_do_trade(context, bar_dict, tempstocks)
	'''

# 获得因子函数
def get_needs(context, bar_dict, date, stocks):
	time_tuple = time.strptime(date, '%Y%m%d')
	year, month, day = time_tuple[:3]
	# 转化为datetime.date类型
	date = datetime.date(year, month, day)

	last_year_date =(date - relativedelta(years = 1)).strftime('%Y%m%d')
	log.info('上个交易日前一年的日期为：' + last_year_date)

	# 上一年内的所有交易日期
	trade_days = get_trade_days(last_year_date, date).strftime('%Y-%m-%d')

	# 所有因子
	need_all = ','.join(context.need)

	# 1.分组单因子有效性检验，参考资料：https://www.ricequant.com/community/topic/702/%E5%8D%95%E4%B8%80%E5%9B%A0%E5%AD%90%E6%9C%89%E6%95%88%E6%80%A7%E6%A3%80%E6%B5%8B/2
	# 每隔25天（间隔越短，收益就会越随机）
	period = 25

	# 有效因子列表
	eff_need = []

	# 每组股票数10%只股票
	num = math.ceil(0.1 * len(stocks))

	# 获取开始日期因子值
	q = query(
		factor.symbol,
		need_all
	).filter(
		factor.symbol.in_(stocks),
		# 20180430的数据消失，原因是：last_year_date可能不是交易日，这样就获取不到数据，因此为trade_days[0]
		factor.date == trade_days[0]
	)

	df = get_factors(q)

	for col in df.columns.values[1:]:
		# 分别按正序和倒序提取因子值前10%和后10%的股票
		a_hign = df.sort_values(col, axis=0, ascending=False)
		a_low = df.sort_values(col, axis=0, ascending=True)

		hign = a_hign.iloc[0:num, 0].values.tolist()
		low = a_low.iloc[0:num, 0].values.tolist()
		hign_price = get_price(hign, last_year_date, date, '1d', ['close'], skip_paused = False, fq = 'pre', is_panel = 1)
		low_price = get_price(low, last_year_date, date, '1d', ['close'], skip_paused = False, fq = 'pre', is_panel = 1)

		hign_price = hign_price['close']
		low_price = low_price['close']

		# 计算平均每组股票每日价格
		high_series = hign_price.T.mean()
		low_series = low_price.T.mean()

		# 计算收益率
		hign_rr = high_series.pct_change(period).dropna()
		low_rr = low_series.pct_change(period).dropna()
		
		# 判断两组数据的方差是否显著不同
		le = levene(hign_rr, low_rr)

		# 判断两组数据的均值是否显著不同
		tt = ttest_ind(hign_rr, low_rr, equal_var = False)

		# 取置信度为0.05
		if le.pvalue < 0.05 and tt.pvalue < 0.05:
			eff_need.append('factor.%s' % col)

	# 第一步有效因子个数
	log.info(len(eff_need))

	effneed = ','.join(eff_need)

	# 第二步,股票池股票分十档计算IC、IR、根据IC特征设定方向，根据每组IC均值设定因子权重

	# gp = 2
	q = query(
		factor.symbol,
		effneed
	).filter(
		factor.symbol.in_(stocks),
		factor.date == trade_days[0]
	)

	df = get_factors(q)

	# step档
	step = 10

	# 周期
	nd = 22

	# 用来存放各指标值
	need_data = {}

	# 循环每一个因子
	for col in df.columns.values[1:]:
		# 因子值从大到小排列
		a_hign = df.sort_values(col, axis=0, ascending=False)
		hign = a_hign.iloc[:, 0].values.tolist()

		# 用来存放step组数据IC均值、ic、pr值列表
		avr = [[], [], []]

		# 对所取两组进行迭代(gp_random：stocks为股票池， gp为取gp组，num为每组num只股票)
		# group()将一个list分为step组
		for sct in group(list(hign), step):
			
			# 用来存放一组内每期ic值
			ics = []

			# 每个分期时间点
			days = trade_days[0:len(trade_days):nd]

			for i, rate in enumerate(g_rate(sct, days, nd)):
				q = query(
					col
				).filter(
					factor.symbol.in_(sct),
					factor.date == days[i]
				)

				df_sct = get_factors(q).fillna(0)

				# col = df_sct.columns.values[0]
				df_sct[col] = standardize(filter_extreme_3sigma(df_sct[col])).fillna(0)

				# ic值
				ic = pd.Series(df_sct[col]).corr(pd.Series(list(rate)))

				ics.append(ic)

			
			# 一组内三个指标
			avr_ic, ir, pr = ics_return(ics)
			avr[0].append(avr_ic)
			avr[1].append(ir)
			avr[2].append(pr)

		need_data[col] = {
			'avr_ic' : avr_6(avr[0]),
			'avr_ir' : avr_6(avr[1]),
			'avr_pr' : avr_6(avr[2])
		}

	# 对因子进行排序打分
	# 存放最终因子
	f_needs = {}

	data = pd.DataFrame(need_data).T.dropna()

	# 按照avr_ir降序打分
	data = data.sort_values('avr_ic', axis=0, ascending=False)
	data['avr_ic_score'] = list(reversed(range(len(data))))
	
	# 按照avr_ir降打分
	data = data.sort_values('avr_ir', axis=0, ascending=False)
	data['avr_ir_score'] = list(range(len(data)))

	# 分数相加
	data['total_socre'] = data['avr_ic_score'] + data['avr_ir_score']
	data = data.sort_values('total_socre', axis=0, ascending=False)

	# 获取排序后的因子
	fneeds = data._stat_axis.values.tolist()
	if len(fneeds) >= context.need_max:
		fneeds = fneeds[0:context.need_max]

	# 获取因子权重
	for i in fneeds:
		weight = data.loc[i, 'avr_ic']

		if data.loc[i, 'avr_pr'] > 0.5:
			weight = - weight

		f_needs['factor.%s' % i] = {
			'weight' : weight
		}

	return f_needs

# 获得股票函数
def get_stocks(context, bar_dict, needs, date, stocks):
	# needs为如下字典：
	# 例{'factor.dbcd': {'weight': -0.146216}, 'factor.weighted_roe': {'weight': 0.108531}, 'factor.turnover_of_current_assets': {'weight': -0.111679}, 'factor.turnover_of_overall_assets': {'weight': -0.148806}, 'factor.administration_cost_div_income': {'weight': 0.118929}}
	need_all = ','.join(list(needs.keys()))
	q = query(
		factor.symbol,
		need_all
	).filter(
		factor.symbol.in_(stocks),
		factor.date == date
	)
	df = get_factors(q).fillna(0)

	# 用来存放股票分值
	score = []

	weight = [needs[i]['weight'] for i in needs]

	for i in range(len(df)):
		result = sum(list(map(lambda x,y:x*y,list(df.iloc[i, 1:]), weight)))
		score.append(result)

	df['score'] = score

	df = df.sort_values('score', axis=0, ascending = False)

	# 大小盘默认持股数
	hold = int(context.hold_max / 2)

	# 存放选取的股票
	stocks_c = list(df.iloc[0 : hold, 0])

	return stocks_c

# 1. 获取上月月末日期
def func_get_end_date_of_last_month(current_date):
	# 获取从上一个交易日前一个月中的所有交易日，日期排序从前至后
	trade_days = list(get_trade_days(None, current_date, count=30))

	# 转化为%Y%m%d格式
	for i in range(len(trade_days)):
		trade_days[i] = trade_days[i].strftime('%Y%m%d')

	# 只要交易日的date和当前交易日的月份不同即为上一个月月末日期，例如[20171013]-[20170928]
	# reversed反转序列，便于快速找到月末日期
	for date in reversed(trade_days):
		if date[5] != current_date[5]:
			return date

	log.info('找不到上个月末调仓日!')
	return

# 获取收益率序列
def g_rate(sct, days, nd):
	for i, day in enumerate(days):
		if i >= 1:
			prices = get_price(sct, days[i-1], days[i], '%dd' % nd, ['close'], skip_paused = False, fq = 'pre',is_panel = 1)['close']
			close_rate = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
			yield close_rate

# 计算一组数据的IC绝对值均值、正向比例、均值/标准差
def ics_return(l):

	# 信息系数绝对值均值
	a = [abs(i) for i in l]
	avr_ic = sum(a) / len(l)

	# 信息比率
	b = np.array(l)
	if b.std() != 0:
		ir = b.mean() / b.std()
	else:
		ir = 0

	# 正向比例
	c = [i for i in l if i > 0]
	pr = len(c) / len(l)
	return avr_ic, ir, pr

# 求一组数的均值并保留6位小数
def avr_6(l):
	b = round(np.array(l).mean(), 6)
	return b

# 3 sigma 去极值
def filter_extreme_3sigma(series, n=3):
	# 均值
	mean = series.mean()

	# 标准差
	std = series.std()

	max_range = mean + n*std
	min_range = mean - n*std

	# clip函数用于将超出范围的值填充为min_range,max_range
	return np.clip(series, min_range, max_range)

# z-score标准化
def standardize(series):
	std = series.std()
	mean = series.mean()

	return (series - mean) / std

# 分组迭代器
def group(l, s):
	length = len(l)
	for i in range(s):
		que = []
		left = i * (length // s)
		if (i+1) * (length // s) < length:
			right = (i+1) * (length // s)
		else:
			right = length
		for j in l[left:right]:
			que.append(j)
		yield que

# 每日运行函数
def handle_bar(context, bar_dict):
	last_date = get_last_datetime().strftime('%Y%m%d')
	# log.info('上个交易日日期为：' + last_date)

	if last_date != context.last_date and len(list(context.portfolio.stock_account.positions.keys())) > 0:
		# 如果不是调仓日且有持仓，择时买入
		func_stop_loss(context, bar_dict)


def func_stop_loss(context, bar_dict):
	#获取账户持仓信息
	holdstock = list(context.portfolio.stock_account.positions.keys()) 
	if len(holdstock) > 0:
		num = -0.1
		for stock in holdstock:
			close = history(stock,['close'],1,'1d').values
			if close/context.portfolio.positions[stock].last_price -1 <= num:
				order_target(stock,0)
				log.info('股票{}已止损'.format(stock))

	#获取账户持仓信息
	holdstock = list(context.portfolio.stock_account.positions.keys()) 
	if len(holdstock) > 0:
		num = - 0.13
		T = history('000001.SH',['quote_rate'],7,'1d').values.sum()
		if T < num*100:
			log.info('上证指数连续三天下跌{}已清仓'.format(T))
			for stock in holdstock:
				order_target(stock,0)

# 4.下单函数 
def func_do_trade(context, bar_dict, tempstocks):
	# 先清空所有持仓
	if len(list(context.portfolio.stock_account.positions.keys())) > 0:
		for stock in list(context.portfolio.stock_account.positions.keys()):
			order_target(stock, 0)
	# 等权重分配资金
	cash = context.portfolio.available_cash/context.max_stocks

	# 买入新调仓股票
	for security in tempstocks:
		order_target_value(security, cash)
	return