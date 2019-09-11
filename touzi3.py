import pandas as pd
import numpy as np
import datetime, time
from dateutil.relativedelta import relativedelta
import talib
import talib as ta
import math, random
from scipy.stats import ttest_ind
from scipy.stats import levene
from sklearn.linear_model import LinearRegression
import copy

g.record = pd.DataFrame({'symbol': [], 'add_time': [], 'last_buy_price': []})


def init(context):
    g.num = 0
    log.info('begin')
    # get_iwencai('沪深300')
    # set_benchmark('000905.SH')
    # 设置最大持股数
    context.stoplossmultipler = 0.95
    # 设置交易股票数量
    context.max_stocks = 40
    matrix1 = [[0 for i in range(10)]]
    g.matrix = matrix1
    # 用于计数
    g.means1 = g.means2 = g.means3 = g.means4 = g.means5 = g.means6 = g.means7 = 0

    # 最大持股数量(大小盘各一半)
    context.hold_max = 40

    # 初始因子池（包括技术因子和财务因子在内的共计107个初始因子）
    context.need = ['factor.pe',  # 市盈率
                    'factor.pb',  # 市净率
                    'factor.pcf_cash_flow_ttm',  # 市现率PCF
                    'factor.ps',  # 市销率
                    'factor.dividend_rate',  # 股息率
                    'factor.market_cap',  # 总市值
                    'factor.current_market_cap',  # 流通市值
                    'factor.capitalization',  # 总股本
                    'factor.circulating_cap',  # 流通股本
                    'factor.current_ratio',  # 流动比率
                    'factor.equity_ratio',  # 产权比率负债合计／归属母公司股东的权益
                    'factor.quick_ratio',  # 速动比率
                    'factor.tangible_assets_liabilities',  # 有形资产／负债合计
                    'factor.tangible_assets_int_liabilities',  # 有形资产／带息债务
                    'factor.net_debt_equity',  # 净债务／股权价值
                    'factor.long_term_debt_to_opt_capital_ratio',  # 长期债务与营运资金比率
                    'factor.tangible_assets_net_liabilities',  # 有形资产／净债务
                    'factor.overall_income_growth_ratio',  # 营业总收入同比增长率
                    # 'factor.net_cashflow_psg_rowth_ratio',# 每股经营活动产生的现金流量净额同比增长率(报错，原因询问客服ing)
                    # 'factor.opt_profit_grow_ratio',# 营业利润同比增长率
                    # 'factor.total_profit_growth_ratio',# 利润总额同比增长率
                    'factor.diluted_net_asset_growth_ratio',  # 净资产收益率摊薄同比增长率
                    'factor.net_cashflow_from_opt_act_growth_ratio',  # 经营活动产生的现金流量净额同比增长率
                    'factor.net_profit_growth_ratio',  # 净利润同比增长率
                    # 'factor.basic_pey_ear_growth_ratio',# 基本每股收益同比增长率
                    'factor.turnover_of_overall_assets',  # 总资产周转率
                    'factor.turnover_ratio_of_account_payable',  # 应付账款周转率
                    'factor.turnover_of_current_assets',  # 流动资产周转率
                    'factor.turnover_of_fixed_assets',  # 固定资产周转率
                    'factor.cash_cycle',  # 现金循环周期
                    'factor.inventory_turnover_ratio',  # 存货周转率
                    'factor.turnover_ratio_of_receivable',  # 应收账款周转率
                    'factor.weighted_roe',  # 净资产收益率roe加权
                    'factor.overall_assets_net_income_ratio',  # 总资产净利率roa
                    'factor.net_profit_margin_on_sales',  # 销售净利率
                    'factor.before_tax_profit_div_income',  # 息税前利润／营业总收入
                    'factor.sale_cost_div_income',  # 销售费用／营业总收入
                    'factor.roa',  # 总资产报酬率roa
                    'factor.ratio_of_sales_to_cost',  # 销售成本率
                    'factor.net_profit_div_income',  # 净利润／营业总收入
                    'factor.opt_profit_div_income',  # 营业利润／营业总收入
                    'factor.opt_cost_div_income',  # 营业总成本／营业总收入
                    'factor.administration_cost_div_income',  # 管理费用／营业总收入
                    'factor.financing_cost_div_income',  # 财务费用／营业总收入
                    'factor.impairment_loss_div_income',  # 资产减值损失／营业总收
                    # 以下为技术因子
                    'factor.vr_rate',  # 成交量比率
                    'factor.vstd',  # 成交量标准差
                    'factor.arbr',  # 人气意愿指标
                    'factor.srdm',  # 动向速度比率
                    'factor.vroc',  # 量变动速率
                    'factor.vrsi',  # 量相对强弱指标
                    'factor.cr',  # 能量指标
                    'factor.mfi',  # 资金流向指标
                    'factor.vr',  # 量比
                    'factor.mass',  # 梅丝线
                    'factor.obv',  # 能量潮
                    'factor.pvt',  # 量价趋势指标
                    'factor.wad',  # 威廉聚散指标
                    'factor.bbi',  # 多空指数
                    'factor.mtm',  # 动力指标
                    'factor.dma',  # 平均线差
                    'factor.ma',  # 简单移动平均
                    'factor.macd',  # 指数平滑异同平均
                    'factor.expma',  # 指数平均数
                    'factor.priceosc',  # 价格振荡指标
                    'factor.trix',  # 三重指数平滑平均
                    'factor.dbcd',  # 异同离差乖离率
                    'factor.dpo',  # 区间震荡线
                    'factor.psy',  # 心理指标
                    'factor.vma',  # 量简单移动平均
                    'factor.vmacd',  # 量指数平滑异同平均
                    'factor.vosc',  # 成交量震荡
                    'factor.tapi',  # 加权指数成交值
                    'factor.micd',  # 异同离差动力指数
                    'factor.rccd',  # 异同离差变化率指数
                    'factor.ddi',  # 方向标准差偏离指数
                    'factor.bias',  # 乖离率
                    'factor.cci',  # 顺势指标
                    'factor.kdj',  # 随机指标
                    'factor.lwr',  # L威廉指标
                    'factor.roc',  # 变动速率
                    'factor.rsi',  # 相对强弱指标
                    'factor.si',  # 摆动指标
                    'factor.wr',  # 威廉指标
                    'factor.wvad',  # 威廉变异离散量
                    'factor.bbiboll',  # BBI多空布林线
                    'factor.cdp',  # 逆势操作
                    'factor.env',  # ENV指标
                    'factor.mike',  # 麦克指标
                    'factor.adtm',  # 动态买卖气指标
                    'factor.mi',  # 动量指标
                    'factor.rc',  # 变化率指数
                    'factor.srmi',  # SRMIMI修正指标
                    'factor.dptb',  # 大盘同步指标
                    'factor.jdqs',  # 阶段强势指标
                    'factor.jdrs',  # 阶段弱势指标
                    'factor.zdzb',  # 筑底指标
                    'factor.atr',  # 真实波幅
                    'factor.std',  # 标准差
                    'factor.vhf',  # 纵横指标
                    'factor.cvlt']  # 佳庆离散指标

    # 最大因子数目
    context.need_max = 10

    # 上一次调仓期
    context.last_date = ''
    # 设置调仓周期，每月倒数第一个交易日运行
    run_monthly(func=reallocate, date_rule=1)

    # 设置因子测试周期为12个月
    context.need_tmonth = 12

    # 选择沪深300
    get_iwencai('沪深300', 'hs_300')

    # 选择中证500
    get_iwencai('中证500', 'zz_500')


def reallocate(context, bar_dict):
    log.info(
        "VOL5上穿VOL10次数:{},OL5下穿VOL10:{},DIF{}，MA5上穿MA10:{}，MA5下穿MA10:{}，布林线:{}".format(g.means1, g.means2, g.means3,
                                                                                       g.means4, g.means5, g.means6))
    # 获取上一个交易日的日期
    date = get_last_datetime().strftime('%Y%m%d')
    log.info('上个交易日日期为：' + date)

    # 获取上个月末调仓日期
    context.last_date = func_get_end_date_of_last_month(date)

    log.info('上月月末调仓日期为：' + context.last_date)

    hs_needs, zz_needs = get_needs(context, bar_dict, date, context.hs_300), get_needs(context, bar_dict, date,
                                                                                       context.zz_500)
    log.info('这是沪深300股票池的因子：')
    log.info(hs_needs)

    log.info('这是中证500股票池的因子：')
    log.info(zz_needs)

    # 用来存放大小盘股票，0是小盘，1是大盘
    tempstocks = [[], []]
    dapan_stocks = get_stocks(context, bar_dict, zz_needs, date, context.zz_500)
    for stock in dapan_stocks:
        tempstocks[0].append(stock)
        tempstocks[1].append(1)
    xiaopan_stocks = get_stocks(context, bar_dict, hs_needs, date, context.zz_500)
    for stock in xiaopan_stocks:
        tempstocks[0].append(stock)
        tempstocks[1].append(0)
    # log.info(tempstocks)
    log.info('这是沪深300股票池：')
    log.info(dapan_stocks)

    log.info('这是中证500股票池：')
    log.info(xiaopan_stocks)
    # unit = calcUnit(context.portfolio.portfolio_value, atr)  # 计算加仓时应购入的股票数量
    # for s in tempstocks[0]:
    #     order(s, 1 * 2000)  # 七个0时约2000
    log.info("股票池:")
    log.info(tempstocks[0])
    # 第一个月
    if g.matrix[0][0] == 0:
        # log.info("第一个月")
        # log.info(g.matrix)
        g.matrix.pop(0)  # 将此股票从数组中删除
        # log.info(g.matrix)
        for num, tempstock in enumerate(tempstocks[0]):
            matrix2 = [0 for i in range(10)]
            matrix2[0] = tempstock
            # 1是大盘 0是小盘
            matrix2[8] = tempstocks[1][num]
            matrix2[9] = 1
            matrix3 = copy.deepcopy(matrix2)
            g.matrix.append(matrix3)
        log.info(g.matrix)
    else:
        # 先清除标记
        log.info("开始新一轮的选股")
        log.info(g.matrix)
        for n, s in enumerate(g.matrix):
            for i in range(0, 9):
                if i == 0 or i == 8:
                    continue
                g.matrix[n][i] = 0
        log.info(g.matrix)
        # 需要比较是否与前一个月重复
        for num, tempstock in enumerate(tempstocks[0]):
            flag_repeat = 0
            for n, stock in enumerate(g.matrix):
                if stock[0] == tempstock:
                    flag_repeat = 1
                    # 用来标记是否新买入的股票
                    g.matrix[n][9] = 1
                    log.info(tempstock + "原本就有")
                    break
            if not flag_repeat:
                # 不重复 就新加
                matrix2 = [0 for i in range(10)]
                matrix2[0] = tempstock
                # 用来标记是否新买入的股票
                matrix2[9] = 1
                matrix2[8] = tempstocks[1][num]
                matrix3 = copy.deepcopy(matrix2)
                g.matrix.append(matrix3)
                log.info(tempstock + "新买入")
        log.info(g.matrix)
        # 将这个月不合适的股票进行清仓
        # for num, stock in enumerate(g.matrix):
        #     if stock[9] == 0:
        #         sec = g.matrix[num][0]
        #         order_target_value(sec, 0)  # 清仓离场
        #         g.record = g.record[g.record['symbol'] != sec]  # 将卖出股票的记录清空
        #         g.matrix.pop(num)  # 将此股票从数组中删除

    # 用于计数
    g.means1 = g.means2 = g.means3 = g.means4 = g.means5 = g.means6 = 0


# log.info(tempstocks)

# 获得因子函数
def get_needs(context, bar_dict, date, stocks):
    time_tuple = time.strptime(date, '%Y%m%d')
    year, month, day = time_tuple[:3]
    # 转化为datetime.date类型
    date = datetime.date(year, month, day)

    last_year_date = (date - relativedelta(years=1)).strftime('%Y%m%d')
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
        low = a_low.ilo+c[0:num, 0].values.tolist()
        hign_price = get_price(hign, last_year_date, date, '1d', ['close'], skip_paused=False, fq='pre', is_panel=1)
        low_price = get_price(low, last_year_date, date, '1d', ['close'], skip_paused=False, fq='pre', is_panel=1)

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
        tt = ttest_ind(hign_rr, low_rr, equal_var=False)
        
        # 取置信度为0.05
        if le.pvalue < 0.05 and tt.pvalue < 0.05:
            eff_need.append('factor.%s' % col)

    # 第一步有效因子个数
    # log.info(len(eff_need))

    effneed = ','.join(eff_need)

    # 第二步,取前50只股票来计算IC、IR、根据IC特征设定方向，根据每组IC均值设定因子权重（降低运算数量）
    q = query(
        factor.symbol,
        effneed
    ).filter(
        factor.symbol.in_(stocks),
        factor.date == trade_days[0]
    )

    df = get_factors(q)

    step = 50
    # 周期
    nd = 22

    # 用来存放各指标值
    need_data = {}

    # 循环每一个因子
    for col in df.columns.values[1:]:

        avr = [[], [], []]
        a_hign = df.sort_values(col, axis=0, ascending=False)
        sct = a_hign.iloc[0:step, 0].values.tolist()

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

            col = df_sct.columns.values[0]
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
            'avr_ic': round(avr[0][0], 6),
            'avr_ir': round(avr[1][0], 6),
            'avr_pr': round(avr[2][0], 6)
        }

    # 对因子进行排序打分
    # 存放最终因子
    f_needs = {}
    log.info('未排序的因子字典：')
    log.info(need_data)

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
    data_stock = {col:data[col].tolist() for col in data.columns}

    # 获取排序后的因子
    fneeds = data._stat_axis.values.tolist()
    data_stock['factor'] = fneeds
    log.info('因子排序指标字典：')
    log.info(data_stock)
    if len(fneeds) >= context.need_max:
        fneeds = fneeds[0:context.need_max]

    # 获取因子权重
    for i in fneeds:
        weight = data.loc[i, 'avr_ic']

        if data.loc[i, 'avr_pr'] < 0.5:
            weight = - weight

        f_needs['factor.%s' % i] = {
            'weight': weight
        }
    log.info(f_needs)
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
        result = sum(list(map(lambda x, y: x * y, list(df.iloc[i, 1:]), weight)))
        score.append(result)

    df['score'] = score

    df = df.sort_values('score', axis=0, ascending=False)
    df_stock = {col:df[col].tolist() for col in df.columns}
    
    log.info('股票排序字典：')
    log.info(df_stock)
    # 大小盘默认持股数
    hold = int(context.hold_max / 2)

    # 存放选取的股票
    stocks_c = list(df.iloc[0: hold, 0])

    return stocks_c


# 每日运行函数
def handle_bar(context, bar_dict):
    last_date = get_last_datetime().strftime('%Y%m%d')
    trade_before_new(context, bar_dict)
    trade_new(context, bar_dict)
    log.info(
        "VOL5上穿VOL10次数:{},OL5下穿VOL10:{},DIF{}，MA5上穿MA10:{}，MA5下穿MA10:{}，布林线:{}".format(g.means1, g.means2, g.means3,
                                                                                       g.means4, g.means5, g.means6))


# log.info('上个交易日日期为：' + last_date)
def trade_new(context, bar_dict):
    a = get_datetime()
    # current_universe = context.iwencai_securities  # 获取当前除停牌外的所有可供投资股票（universe）

    # security_position = context.portfolio.stock_account.positions.keys()  # 字典数据，上一K线结束后的有效证券头寸，即持仓数量大于0的证券及其头寸
    for index, jj in enumerate(g.matrix):

        stock = g.matrix[index][0]  # 获得备选列表中的股票代码
        sec = stock
        curent = get_candle_stick([stock], end_date=a, fre_step='1m', fields=['open', 'close', 'high', 'low', 'volume'],
                                  skip_paused=False, fq=None, bar_count=5, is_panel=0)
        current_price = curent[stock]['open'][-1]

        value = history([stock], ['close', 'open', 'low', 'turnover', 'high'], 20, '1d', False, None)

        value1 = history(stock, ['close', 'open', 'low', 'volume', 'high'], 60, '1d', True, 'pre')
        value1 = value1.dropna()
        close = value1.close.values
        # log.info(close, value1, stock)
        low = value1.low.values
        # high = value1['high'].as_matrix()
        high = value1.high.values
        vol = value1['volume'].as_matrix()
        current_price = close[-1]  # 获得当前时刻价格
        # short_ma = ta.MA(close, 5)  # 5天均线
        # long_ma = ta.MA(close, 10)  # 10天均线
        # mid_ma = ta.MA(close, 20)  # 20天均线
        # up, mid,down = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # print(up)
        # 数据处理（策略部分）：计算真实波幅，
        try:
            atr = talib.ATR(high, low, close, 20)[-1]  # 计算真实波幅
        except:
            print(high, low, close)

        upperband, middleband, lowerband = talib.BBANDS(np.asarray(value[stock]['close']), timeperiod=10, nbdevup=1.96,
                                                        nbdevdn=1.96, matype=0)
        # log.info(g.matrix[index][0],g.matrix[index][1],g.matrix[index][3],g.matrix[index][4])
        if g.matrix[index][8] == 1:
            flag_buy = 0
            for t in g.record['symbol']:
                if t == sec:
                    flag_buy = 1
                    if (g.record[g.record['symbol'] == sec]['last_buy_price'].empty):
                        continue
                    else:

                        last_price = float(g.record[g.record['symbol'] == sec]['last_buy_price'])  # 上一次的买入价格（查找、判断、删除）
                        add_price = last_price + 0.5 * atr  # 计算是否加仓的判断价格
                        add_price_top = last_price + 1.5 * atr
                        add_unit = float(g.record[g.record['symbol'] == sec]['add_time'])  # 已加仓次数
                        # if current_price > add_price and add_unit < 4 and g.matrix[index][9]==1:  # 价格上涨超过0.5N并且加仓次数小于4次
                        if current_price > add_price and g.matrix[index][9] == 1:  # 价格上涨超过0.5N并且加仓次数小于4次
                            log.info("加仓")
                            log.info(sec)
                            unit = calcUnit(context.portfolio.available_cash, atr)  # 计算加仓时应购入的股票数量
                            log.info(unit)
                            try:
                                if g.record[g.record['symbol'] == sec]['add_time'] == 0:
                                    order(sec, 1 * unit)  # 买入2unit的股票
                                    log.info("买入2unit的股票")
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 1:
                                    order(sec, 1 * unit)  # 买入1.5unit的股票
                                    log.info("买入1.5unit的股票")
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 2:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                                    log.info("买入1unit的股票")
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 3:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                                    log.info("买入0.5unit的股票")
                            except:
                                order(sec, 1 * unit)  # 买入2unit的股票
                                log.info("买入2unit的股票")
                            g.record.loc[g.record['symbol'] == sec, 'add_time'] = g.record[g.record['symbol'] == sec][
                                                                                      'add_time'] + 1  # 加仓次数+1（先找到值再找到位置，然后赋值）
                            g.record.loc[g.record['symbol'] == sec, 'last_buy_price'] = current_price  # 加仓次数+1
                        # 策略部分(离场：止损或止盈)：当前价格下穿到BOLL上轨或相对上个买入价下跌 2ATR时（止损）或当股价5日均线下穿10日均线并且量均线死叉，清仓离场
                        if current_price < (last_price - 2 * atr) or current_price < low[-10:-1].min():
                            log.info("抛售")
                            log.info(sec)
                            g.means7 += 1
                            order_target_value(sec, 0)  # 清仓离场

                            g.record = g.record[g.record['symbol'] != sec]  # 将卖出股票的记录清空

            if ((g.matrix[index][1] == True and g.matrix[index][3] == True and g.matrix[index][4] == True) or (
                    g.matrix[index][6] == True and flag_buy == 0)) and g.matrix[index][9] == 1:
                g.num += 1
                order_target_value(stock, context.portfolio.available_cash / 40)
                log.info("购入大盘")
                if len(g.record) != 0:
                    g.record = g.record[
                        g.record[
                            'symbol'] != stock]  # 清空g.record中sec过期的记录，因为如果之前记录就有000001，现在将这个股票建仓，就要先删除原来为symbol的记录。

                g.record = g.record.append(pd.DataFrame(
                    {'symbol': [stock], 'add_time': [1], 'last_buy_price': [current_price]}))  # 记录股票，加仓次数及买入价格
                continue

        if g.matrix[index][8] == 0:
            flag_buy = 0
            for t in g.record['symbol']:
                if t == sec:
                    flag_buy = 1
                    if (g.record[g.record['symbol'] == sec]['last_buy_price'].empty):
                        continue
                    else:

                        last_price = float(g.record[g.record['symbol'] == sec]['last_buy_price'])  # 上一次的买入价格（查找、判断、删除）
                        add_price = last_price + 0.5 * atr  # 计算是否加仓的判断价格
                        add_price_top = last_price + 1.5 * atr
                        add_unit = float(g.record[g.record['symbol'] == sec]['add_time'])  # 已加仓次数
                        # if current_price > add_price and add_unit < 4 and current_price >= (
                        #         last_price - 2 * atr)and g.matrix[index][9]==1:  # 价格上涨超过0.5N并且加仓次数小于4次
                        if current_price > add_price and current_price >= (
                                last_price - 2 * atr) and g.matrix[index][9] == 1:  # 价格上涨超过0.5N并且加仓次数小于4次
                            log.info("加仓")
                            log.info(sec)
                            unit = calcUnit(context.portfolio.available_cash, atr)  # 计算加仓时应购入的股票数量
                            log.info(unit)
                            try:
                                if g.record[g.record['symbol'] == sec]['add_time'] == 0:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 1:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 2:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                                elif g.record[g.record['symbol'] == sec]['add_time'] == 3:
                                    order(sec, 1 * unit)  # 买入1unit的股票
                            except:
                                order(sec, 1 * unit)  # 买入1unit的股票
                            g.record.loc[g.record['symbol'] == sec, 'add_time'] = g.record[g.record['symbol'] == sec][
                                                                                      'add_time'] + 1  # 加仓次数+1（先找到值再找到位置，然后赋值）
                            g.record.loc[g.record['symbol'] == sec, 'last_buy_price'] = current_price  # 加仓次数+1
                        # 策略部分(离场：止损或止盈)：当前价格下穿到BOLL上轨或相对上个买入价下跌 2ATR时（止损）或当股价5日均线下穿10日均线并且量均线死叉，清仓离场
                        if current_price < (last_price - 2 * atr) or current_price < low[-10:-1].min():
                            log.info("抛售")
                            g.means7 += 1
                            log.info(sec)
                            order_target_value(sec, 0)  # 清仓离场

                            g.record = g.record[g.record['symbol'] != sec]  # 将卖出股票的记录清空

            if ((g.matrix[index][1] == True and g.matrix[index][3] == True and g.matrix[index][4] == True) or (
                    g.matrix[index][6] == False) and (flag_buy == 0)) and g.matrix[index][9] == 1:

                log.info("购入小盘")
                g.num += 1
                # log.info(stock)
                #     log.info('g.matrix[index][4]==True and g.matrix[index][6]==True and g.matrix[index][2]==True')
                order_target_value(stock, context.portfolio.available_cash / 40)
                if len(g.record) != 0:
                    g.record = g.record[
                        g.record[
                            'symbol'] != stock]  # 清空g.record中sec过期的记录，因为如果之前记录就有000001，现在将这个股票建仓，就要先删除原来为symbol的记录。

                g.record = g.record.append(pd.DataFrame(
                    {'symbol': [stock], 'add_time': [1], 'last_buy_price': [current_price]}))  # 记录股票，加仓次数及买入价格
                continue


def trade_before_new(context, bar_dict):
    if g.matrix[0][0] == 0:
        reallocate(context, bar_dict)
        # log.info(g.matrix)
    a = get_datetime()
    xiaopan = '000905.SH'
    dapan = '000300.SH'
    Indexprice_xiao = get_price(xiaopan, None, get_datetime().strftime("%Y%m%d"), '1d', ['close'], True, None, 100, is_panel=1)
    Indexprice_da = get_price(dapan, None, get_datetime().strftime("%Y%m%d"), '1d', ['close'], True, None, 100, is_panel=1)
    strong_weak = np.log(Indexprice_xiao.close) - np.log(Indexprice_da.close)
    strong_weak = strong_weak[20:]
    lowerband_new = pd.rolling_mean(strong_weak, 20) - 2 * pd.rolling_std(strong_weak, 20)
    up_new = pd.rolling_mean(strong_weak, 20) + 2 * pd.rolling_std(strong_weak, 20)
    for index, jj in enumerate(g.matrix):

        # 基本数据获取
        stock = g.matrix[index][0]  # 获得备选列表中的股票代码
        # log.info(g.matrix[index])
        sec = stock
        curent = get_candle_stick([stock], end_date=a, fre_step='1m', fields=['open', 'close', 'high', 'low', 'volume'],
                                  skip_paused=False, fq=None, bar_count=5, is_panel=0)
        value = history([stock], ['close', 'open', 'low', 'turnover', 'high'], 20, '1d', False, None)

        value1 = history(stock, ['close', 'open', 'low', 'volume', 'high'], 110, '1d', True, 'pre')
        value1 = value1.dropna()
        close = value1.close.values
        low = value1.low.values
        high = value1['high'].as_matrix()
        vol = value1['volume'].as_matrix()

        # current_price = close[-1]  # 获得当前时刻价格
        # log.info(close)
        short_ma = ta.MA(close, 5)  # 5天均线
        long_ma = ta.MA(close, 10)  # 10天均线
        mid_ma = ta.MA(close, 20)  # 20天均线
        up, mid, low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # 数据处理（策略部分）：计算真实波幅，
        # atr = talib.ATR(high, low, close, 20)[-1]  # 计算真实波幅

        # 第一部分 计算#VOL5上穿VOL10

        short_vol = ta.MA(vol, 5)  # 过去5天均量线
        long_vol = ta.MA(vol, 10)  # 过去10天均量线

        if short_vol[-1] > short_vol[-2] and long_vol[-1] > long_vol[-2] and short_vol[-2] <= long_vol[-2] and \
                short_vol[-1] > long_vol[-1]:
            g.matrix[index][1] = True  # VOL5上穿VOL10且多头向上排列，可建仓标志
            g.means1 += 1
        else:
            g.matrix[index][1] = False  # 不可建仓标志

        if (short_vol[-2] >= long_vol[-2] and short_vol[-1] < long_vol[-1]):
            g.matrix[index][2] = True  # VOL5下穿VOL10，可清仓标志
            g.means2 += 1
        else:
            g.matrix[index][2] = False  # 不可清仓标志

        # 第二部分 #MACD中的DIF>DEA & DIF>0 & DEA>0

        close_macd = value[stock]['close'].as_matrix()

        DIF, DEA, MACD = talib.MACD(close, fastperiod=12, slowperiod=26,
                                    signalperiod=9)  # talib提供的MACD计算函数，计算DIF,DEF以及MACD的取值

        if DIF[-1] > DEA[-1] and DIF[-1] > 0 and DEA[-1] > 0:
            g.matrix[index][3] = True
            g.means3 += 1

        else:
            g.matrix[index][3] = False

        # 第三部分 是否触发MA5与MA10建仓或离场标志

        if short_ma[-1] > mid_ma[-2] and short_ma[-1] > short_ma[-2] and mid_ma[-1] > mid_ma[-2] and short_ma[-1] > \
                mid_ma[-1] and mid_ma[-1] > long_ma[-1] and short_ma[-2] <= mid_ma[-2] and short_ma[-1] > mid_ma[-1]:
            g.matrix[index][4] = True  # MA5上穿MA10且三线多头向上排列，可建仓标志
            g.means4 += 1
        else:
            g.matrix[index][4] = False  # 不可建仓标志
        if (short_ma[-2] >= mid_ma[-2] and short_ma[-1] < mid_ma[-1]):
            g.matrix[index][5] = True  # MA5下穿MA10或者当前价跌破MA10，可清仓标志
            g.means5 += 1
        else:
            g.matrix[index][5] = False  # 不可清仓标志

        # 第四部分 创新穿越次数

        # value_new = history([stock], ['close', 'open', 'low', 'turnover', 'high'], 110, '1d', False, None)
        # upperband_new, middleband_new, lowerband_new = talib.BBANDS(np.asarray(value_new[stock]['close']),
        #                                                             timeperiod=10, nbdevup=1.96, nbdevdn=1.96, matype=0)
        flag_shangchuan = 0
        flag_xiachuan = 0
        flag_shangchuancishu = 0
        flag_xiachuancishu = 0

        # for t in range(9, 109):
        #     if flag_shangchuan == 0:
        #         if strong_weak[t] < upperband_new[t]:
        #             # 定义标识数
        #             flag_shangchuan = 1
        #     if flag_shangchuan == 1:
        #         if strong_weak[t] > upperband_new[t]:
        #             flag_shangchuancishu = flag_shangchuancishu + 1
        #             flag_shangchuan = 2
        #     if flag_shangchuan == 2:
        #         if strong_weak[t] > upperband_new[t]:
        #             flag_shangchuan = 2
        #         if strong_weak[t] < upperband_new[t]:
        #             flag_shangchuan = 0
        if (strong_weak[-2] > lowerband_new[-2]) and (strong_weak[-1] < lowerband_new[-1]):
            # log.info(strong_weak[-2],lowerband_new[-2])
            # log.info(strong_weak[-1],lowerband_new[-1])
            g.matrix[index][6] = True
            g.means6 += 1
        else:
            g.matrix[index][6] = False
    flag_xiachuancishu = 0
    # for t in range(1, 80):
    #     if flag_xiachuan == 0:
    #         if strong_weak[t] > lowerband_new[t]:
    #             # 定义标识数
    #             flag_xiachuan = 1
    #     if flag_xiachuan == 1:
    #         if strong_weak[t] <= lowerband_new[t]:
    #             flag_xiachuancishu = 1
    #             flag_xiachuan = 2
    #
    # if flag_xiachuancishu != 0:
    #     # log.info("!=0")
    #     # log.info(stock+"触碰布林线下穿线")
    #     g.matrix[index][6] = True
    #     g.means6+=1
    # else:
    #     # log.info("==0")
    #     # log.info(stock + "不触碰布林线下穿线")
    #     g.matrix[index][6] = False

    # trade(context)


# #收盘上穿20日BOLL下轨


# 日均线穿布林线策略。该函数返回0 1 2 三类数据。0代表 上穿次数大于下 1代表相等 2代表 上穿小于下穿

# #加仓条件

def calcUnit(portfolio_value, ATR):
    value = portfolio_value * 0.01  # trade_percent
    return int((value / ATR) / 100) * 100


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
            prices = \
                get_price(sct, days[i - 1], days[i], '%dd' % nd, ['close'], skip_paused=False, fq='pre', is_panel=1)[
                    'close']
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

    max_range = mean + n * std
    min_range = mean - n * std

    # clip函数用于将超出范围的值填充为min_range,max_range
    return np.clip(series, min_range, max_range)


# z-score标准化
def standardize(series):
    std = series.std()
    mean = series.mean()

    return (series - mean) / std


# 随机迭代器
def gp_random(stocks, gp, num):
    for i in range(gp):
        yield random.sample(stocks, num)

