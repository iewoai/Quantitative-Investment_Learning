import pandas as pd
import numpy as np
import datetime

def init(context):
    # 使用智能选股函数设置股票池 
    get_iwencai('沪深300')
    # 上一次调仓期
    context.last_date = ''
    # 设置最大持股数
    context.max_stocks = 10 
    # 设置调仓周期，每月月末
    run_monthly(reallocation,date_rule=-1)

def reallocation(context, bar_dict):
    # 获取上一个交易日的日期
    date = get_last_datetime().strftime('%Y%m%d')

    # 获取上个月末调仓日期
    context.last_date = func_get_end_date_of_last_month(date)

    log.info('############################## ' + str(date) + ' ###############################')
    '''
    # 每个调仓日先清仓持有的股票
    for security in list(context.portfolio.stock_account.positions.keys()):
        order_target(security, 0)
    '''
    # 首先获得当前日期
    time = get_datetime()
    date = time.strftime('%Y%m%d')
    # 获得股票池列表
    sample = context.iwencai_securities
    # 创建字典用于存储因子值
    df = {'security':[], 1:[], 2:[], 3:[], 'score':[]}
    
    # 因子选择
    for security in sample:
        q=query(
            profit.roic,# 投资回报率
            valuation.pb,# 市净率
            valuation.ps_ttm,# 市销率
        ).filter(
            profit.symbol==security
        )
        
        # 缺失值填充为0
        # fillna(0)缺失值填充0
        fdmt = get_fundamentals(q, date=date).fillna(0)
        
        # 判断是否有数据
        if (not (fdmt['profit_roic'].empty or
                fdmt['valuation_pb'].empty or
                fdmt['valuation_ps_ttm'].empty)):
            # 计算并填充因子值
            df['security'].append(security)
            df[1].append(fdmt['profit_roic'][0])# 因子1：投资回报率
            df[2].append(fdmt['valuation_pb'][0])# 因子2：市净率
            df[3].append(fdmt['valuation_ps_ttm'][0])#因子3：市销率
    
    for i in range(1, 4):
        # 因子极值处理，中位数去极值法
        m = np.mean(df[i])
        s = np.std(df[i])
        for j in range(len(df[i])):
            if df[i][j] <= m-3*s:
                df[i][j] = m-3*s
            if df[i][j] >= m+3*s:
                df[i][j] = m+3*s
        m = np.mean(df[i])
        s = np.std(df[i])
        
        # 因子无量纲处理，标准化法
        for j in range(len(df[i])):
            df[i][j] = (df[i][j]-m)/s
    
    # 计算综合因子得分
    for i in range(len(df['security'])):
        # 等权重计算(注意因子方向)
        s = (df[1][i]-df[2][i]-df[3][i])
        df['score'].append(s)
        
    # 按综合因子得分由大到小排序
    df = pd.DataFrame(df).sort_values(by ='score', ascending=False)
    '''
    # 等权重分配资金
    cash = context.portfolio.available_cash/context.max_stocks
    
    # 买入新调仓股票
    for security in df[:context.max_stocks]['security']:
        order_target_value(security, cash)
    '''
    # 买入挑选的股票
    func_do_trade(context, bar_dict, df)
    
    context.last_date = date

def handle_bar(context, bar_dict):
    last_date = get_last_datetime().strftime('%Y%m%d')
    if last_date != context.last_date and len(list(context.portfolio.stock_account.positions.keys())) > 0:
        # 如果不是调仓日且有持仓，判断止损条件
        func_stop_loss(context, bar_dict)

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

    log.info('Cannot find the end date of last month.')
    return

#### 4.下单函数 ###################################################################
def func_do_trade(context, bar_dict, df):
    # 先清空所有持仓
    if len(list(context.portfolio.stock_account.positions.keys())) > 0:
        for stock in list(context.portfolio.stock_account.positions.keys()):
            order_target(stock, 0)
    '''
    # 买入前30支股票
    for stock in context.selected:
        order_target_percent(stock, 1./context.hold_max)
        if len(list(context.portfolio.stock_account.positions.keys())) >= context.hold_max:
            break
    '''
    # 等权重分配资金
    cash = context.portfolio.available_cash/context.max_stocks
    
    # 买入新调仓股票
    for security in df[:context.max_stocks]['security']:
        order_target_value(security, cash)
    return


#### 5.止损函数 ####################################################################
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