import datetime
import numpy as np
import pandas as pd
import math

# 中位数去极值
def winsorize(df, afactor_name, n=20):
    '''
    df为DataFrame数据
    factor为需要去极值的列名称（str类型即可）
    n 为判断极值上下边界的常数
    '''

    # 提取该列的数据
    ls_raw = np.array(df[afactor_name].values)
    # 排序 axis=0，按列排列
    ls_raw.sort(axis = 0)
    # 获取中位数
    # 返回数组元素的中位数
    D_M = np.median(ls_raw)
    
    # 计算离差值
    ls_deviation = abs(ls_raw - D_M)
    # 排序
    ls_deviation.sort(axis = 0)
    # 获取离差中位数
    D_MAD = np.median(ls_deviation)
    
    # 将大于中位数n倍离差中位数的值赋为NaN
    df.ix[df[afactor_name] >= D_M + n * D_MAD] = None
    # 将小于中位数n倍离差中位数的值赋为NaN
    df.ix[df[afactor_name] <= D_M - n * D_MAD] = None
    return df

# 因子无量纲处理，标准化处理因子值
def standardize(df, afactor_name):
    '''
    df为DataFrame数据
    factor为因子名称，string格式
    '''
    df[afactor_name] = (df[afactor_name] - df[afactor_name].mean())/df[afactor_name].std()
    return df

#得到IC、IR等有效因子指标
def get_effectiveness(index, afactor):
    index = index
    end_date = datetime.date(2018, 4, 30)
    holding_period = 20
    period_number = 12
    quantile_number = 5

    # datetime.timedelta对象代表两个时间之间的时间差
    # 获取end_date前两轮周期前的交易开始时间start_date
    # 两轮周期即一轮因子检验期，一轮回测期？
    start_date = end_date - datetime.timedelta(days = 2 * holding_period * period_number)

    # get_trade_days函数的主要功能获取指定时间段内的交易日
    # 获取从start_date开始的第一轮周期的所有日期
    # 排序为由start_date至end_date
    dates = get_trade_days(start_date, end_date)[- holding_period * period_number - 1:].strftime('%Y%m%d')

    trade_dates = []
    for i in range(0, len(dates), holding_period):

        # 获取从start_date开始的第一轮周期的所有日期中的交易日
        trade_dates.append(dates[i])

    # 转换为DataFrame数据
    tradeDates_df = pd.DataFrame(np.array(trade_dates), columns=['Trade_Dates'])

    # 获取index对应的成分股股票代码
    securities = get_index_stocks(index, trade_dates[0])
    
    q = query(
        factor.symbol,
        afactor
    ).filter(	
        factor.symbol.in_(securities),
        factor.date == trade_dates[0]
    )	
    df = get_factors(q).fillna(0)
    	
    afactor_name = list(afactor)
    afactor_name[6] = '_'
    afactor_name=''.join(afactor_name)
    
    afactor_name=afactor_name[7:]

    df=winsorize(df,afactor_name,20).copy()
    df=df.dropna()
    	
    # 标准化
    df = standardize(df, afactor_name).copy()
	
    # 获取股票收盘价数据
    selected_securities = list(df['factor_symbol'].values)
    prices = get_price(selected_securities, trade_dates[0], trade_dates[1], '1d', ['close', 'is_paused'], skip_paused = False, fq = 'pre',is_panel = 0)
    	
    return_list = []
    for stock in selected_securities:
        # 将持仓周期开始日期停牌的股票收益赋为NaN，便于之后剔除
        if prices[stock]['is_paused'][0]:
            return_list.append(None)
        else:	
            quote_rate = (prices[stock]['close'][-1] - prices[stock]['close'][0]) / prices[stock]['close'][0]
            return_list.append(quote_rate)
    	
    df['returns'] = pd.Series(return_list, index = df.index)
    # 剔除开始日期停牌的股票
    df = df.dropna()
    	
    sorted_list = df.sort_values([afactor_name], ascending = True).copy()
    	
    n = round(1./quantile_number * len(sorted_list))
    long_securities = list(sorted_list['factor_symbol'][:n].values)
    	
    current_return = 0
    for stock in long_securities:
    	
        current_return += (prices[stock]['close'][-1] - prices[stock]['close'][0])/ prices[stock]['close'][0]
    current_return = current_return / n
	
    # 储存各期的IC值
    factor_ic = []
    #用于存储因子各期回报的列表
    factor_return = []
    	
    # 遍历调仓日日期
    for i in range(1, len(trade_dates)):
        securities = get_index_stocks(index, trade_dates[i-1])
        	
        q=query(
            factor.symbol,
            afactor
        ).filter(
            factor.symbol.in_(securities),
            factor.date==trade_dates[i-1]
        )	
        df = get_factors(q).fillna(0)
    	
        #中位数去极值
        df = winsorize(df, afactor_name, 20).copy()
        df = df.dropna()
        	
        #标准化
        df = standardize(df, afactor_name).copy()
        	
        # 获取股票列表
        selected_securities = list(df['factor_symbol'].values)
        # 获取股票收盘价数据
        prices = get_price(selected_securities, trade_dates[i-1], trade_dates[i], '1d', ['close', 'is_paused'], skip_paused = False, fq = 'pre',is_panel = 0)
        	
        return_list = []
        for stock in selected_securities:
            # 将持仓周期开始日期停牌的股票收益赋为NaN，便于之后剔除
            if prices[stock]['is_paused'][0]:
                return_list.append(None)
            else:
                # 计算股票持仓周期的收益率
                quote_rate = (prices[stock]['close'][-1] - prices[stock]['close'][0]) / prices[stock]['close'][0]
                return_list.append(quote_rate)
    	
        # 将股票收益添加到DataFrame
        df['returns'] = pd.Series(return_list, index = df.index)
        	
        # 剔除开始日期停牌的股票
        df = df.dropna()
        	
        # 计算当期IC值
        factor_ic.append(df[afactor_name].corr(df['returns']))
        	
        ######################### 计算因子回报 #########################   
        # 根据因子的值排序
        sorted_list = df.sort_values([afactor_name], ascending = True).copy()
        n = round(1./quantile_number * len(sorted_list))
        long_securities = list(sorted_list['factor_symbol'][:n].values)
        current_return = 0
        for stock in long_securities:
            current_return += (prices[stock]['close'][-1] - prices[stock]['close'][0]) / prices[stock]['close'][0]
        current_return = current_return / n
        factor_return.append(current_return)
    result_df = pd.DataFrame({'Factor_IC': factor_ic,
                              'Factor_Return': factor_return}, index=trade_dates[1:])
    result_df.index.name = 'Dates'
    	
    factor_ic = np.array(factor_ic)
    factor_return = np.array(factor_return)
    average_ic = factor_ic.mean()
    annual_r = (1 + factor_return.mean()) ** (250./holding_period) - 1
    annual_sigma = factor_return.std() * math.sqrt(250./holding_period)
    info_ratio = annual_r / annual_sigma
    win_rate = len(factor_return[factor_return > 0]) / len(factor_return)
    factor_df = pd.DataFrame([average_ic, info_ratio, annual_r, win_rate], index=['IC','IR','Annual_Return','Win_Rate'], columns=[afactor_name])
    return factor_df

# def init(context):是初始化函数 只在整个回测或模拟交易开始运行时执行一次
def init(context):
    get_iwencai('沪深300')
    context.max_stocks = 10 
    run_monthly(reallocation, date_rule=1)
	
def reallocation(context, bar_dict):
    for security in list(context.portfolio.stock_account.positions.keys()):
        order_target(security, 0)
    time = get_datetime()
    date = time.strftime('%Y%m%d')
    sample = context.iwencai_securities
    df = {'security':[], 1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[], 'score':[]}
    	
    l=[]	
    for security in sample:
        q = query(
            factor
        ).filter(
            factor.symbol == security,
            factor.date == date
        )	
        df1 = get_factors(q).fillna(0)
        l=df1.columns.values
        break
        	
    l_new=[]
    for f in l:	
        string = list(f)
        string[6] = '.'
        string=''.join(string)
        l_new.append(string)
    	
    eff_list=[]
    index='000300.SH'
    for f in l_new:
        if f!='factor.id' and f!='factor.symbol' and f!='factor.date':
            try:
                eff=get_effectiveness(index, f)
            except Exception as e:
                continue
            else:
                eff_list.append(eff)
            	
    chosen_factor=[]
    for security in sample:
        q = query(
            factor.pe
        ).filter(
            factor.symbol == security,
            factor.date == date
        )	
        print(q)
        df1 = get_factors(q).fillna(0)
        print('df1:',df1)
        print('df1.columns.values:',df1.columns.values)
        	
        if (not (df1['factor_pe'].empty)):
            df['security'].append(security)
            df[1].append(df1['factor_pe'][0])
        print('factor_pe的值为：',df1)
    	
	
    	
    for i in range(1, 2):
        # 因子极值处理，中位数去极值法
        m = np.mean(df[i])
        s = np.std(df[i])
        for j in range(0,len(df[i])):
            #print('df[i][j]:',df[i][j][0])
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
        s = (df[1][i])
        #print('s',s[0])
        df['score'].append(s)
    	
    df = pd.DataFrame(df).sort_values(by ='score', ascending=False)
    df_security=list(df['security'])
    top20_df_security=[]
    for i in range(0,20):
        top20_df_security.append(df_security[i])
    print('前20个：',top20_df_security)
    cash = context.portfolio.available_cash/context.max_stocks

import talib
import talib as ta
import pandas as pd
import numpy as np
	
record = pd.DataFrame({'symbol':[],'add_time':[],'last_buy_price':[]} )
	
def init(context):
    log.info('begin')
    get_iwencai('沪深300')
    context.stoplossmultipler = 0.95
    context.max_stocks = 40
    matrix1=[[0 for i in range(10)] for i in range(40)]
    g.matrix=matrix1
   	
def before_trading(context):
    	
     	
	
    if g.matrix[0][0]==0:
        for i in range(0,40):
            g.matrix[i][0]=context.iwencai_securities[i]
    trade_before_new(context)
    	
def handle_bar(context, bar_dict):
    trade_new(context)
    	
def trade_new(context):
    global record
    a=get_datetime()
	
    	
    current_universe = context.iwencai_securities  
    security_position = context.portfolio.stock_account.positions.keys()  
    for index in range(0,40):
	
	
        	
        stock=g.matrix[index][0] #获得备选列表中的股票代码
        sec=stock
        curent=get_candle_stick([stock], end_date=a, fre_step='1m', fields=['open', 'close', 'high', 'low', 'volume'], skip_paused=False, fq=None, bar_count=5, is_panel=0)
        current_price=curent[stock]['open'][-1]
	
        value = history([stock], ['close', 'open', 'low', 'turnover', 'high'], 20, '1d', False, None)
	
    	
        value1 = history(stock, ['close', 'open', 'low', 'volume','high'], 60, '1d', True, 'pre')
        value1 = value1.dropna()
        close = value1.close.values
        low = value1.low.values
        high = value1['high'].as_matrix()
        vol = value1['volume'].as_matrix()
        current_price = close[-1]  
        short_ma = ta.MA(close, 5) 
        long_ma = ta.MA(close, 10)  
        mid_ma = ta.MA(close, 20)  
        up, mid, low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
       	
        atr = talib.ATR(high, low, close, 20)[-1]  
	
        	
        upperband, middleband, lowerband = talib.BBANDS(np.asarray(value[stock]['close']), timeperiod=10, nbdevup=1.96, nbdevdn=1.96, matype=0)
        	
        if index<19:
            	
            if (  (g.matrix[index][1]==True and g.matrix[index][3]==True and g.matrix[index][4]==True) or (g.matrix[index][6]==True)) : 
                        
                  
                        order_target_value(stock, context.portfolio.available_cash / 20)
                        if len(record)!=0:                                                                 
                            record = record[record['symbol']!=stock]                                     
                        
                        record = record.append(pd.DataFrame({'symbol':[stock],'add_time':[1],'last_buy_price':[current_price]}))     
                        continue
            	
            	
        if index>=20:
           	
            if (   (g.matrix[index][1]==True and g.matrix[index][3]==True and g.matrix[index][4]==True) or (g.matrix[index][6]==False) ) :  
                	
                        log.info("买小")
                        log.info(stock)
                        order_target_value(stock, context.portfolio.available_cash / 10)
                        if len(record)!=0:                                                                 
                            record = record[record['symbol']!=stock]                                
                        
                        record = record.append(pd.DataFrame({'symbol':[stock],'add_time':[1],'last_buy_price':[current_price]}))    
                        continue
        	
            	
        for t in record['symbol']:
            if t ==sec:
                	
                if (record[record['symbol'] == sec]['last_buy_price'].empty):
                    continue
                else:
                    log.info("进入判断")
                    last_price = float(record[record['symbol'] == sec]['last_buy_price'])  
                    add_price = last_price + 0.5 * atr  
                    add_unit = float(record[record['symbol'] == sec]['add_time']) 
                    if current_price > add_price*0.5 and add_unit < 4:  
                        log.info("加仓")
                        unit = calcUnit(context.portfolio.portfolio_value, atr)  
                        
                        log.info(unit)
                        order(sec, 2*unit)  
                        record.loc[record['symbol'] == sec, 'add_time'] = record[record['symbol'] == sec][
                                                                              'add_time'] + 1  
                        record.loc[record['symbol'] == sec, 'last_buy_price'] = current_price  
                    
                  
                    log.info(current_price,last_price)
                    if current_price < (last_price - 0.3 * atr) :
                        log.info("抛售")
                        log.info(sec)
                        order_target_value(sec, 0) 
                       
                        record = record[record['symbol'] != sec]      
        	
def trade_before_new(context):
    a=get_datetime()
    for index in range(0,40):
	
        stock=g.matrix[index][0] 
        sec=stock
        curent=get_candle_stick([stock], end_date=a, fre_step='1m', fields=['open', 'close', 'high', 'low', 'volume'], skip_paused=False, fq=None, bar_count=5, is_panel=0)
	
        value = history([stock], ['close', 'open', 'low', 'turnover', 'high'], 20, '1d', False, None)
	
    	
        value1 = history(stock, ['close', 'open', 'low', 'volume','high'], 110, '1d', True, 'pre')
        value10=history('000300.SH',['close', 'open', 'low', 'volume','high'], 110, '1d', True, 'pre')
        log.info(value10)
        value11=history('000905.SH',['close', 'open', 'low', 'volume','high'], 110, '1d', True, 'pre')
        value1 = value1.dropna()
        close = value1.close.values
        low = value1.low.values
        high = value1['high'].as_matrix()
        vol = value1['volume'].as_matrix()
        	
        current_price = close[-1] 
        short_ma = ta.MA(close, 5)  
        long_ma = ta.MA(close, 10) 
        mid_ma = ta.MA(close, 20) 
        up, mid, low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        	
        atr = talib.ATR(high, low, close, 20)[-1]  
        upperband, middleband, lowerband = talib.BBANDS(np.asarray(value[stock]['close']), timeperiod=10, nbdevup=1.96, nbdevdn=1.96, matype=0)
        	
        short_vol=ta.MA(vol,5)                                                                      
        long_vol=ta.MA(vol,10)                                                                      
	
        if short_vol[-1]>short_vol[-2] and long_vol[-1]>long_vol[-2] and short_vol[-2]<=long_vol[-2] and short_vol[-1]>long_vol[-1]:
            g.matrix[index][1]=True                                                             
        else: 	
            g.matrix[index][1]=False                                                                          
        if  (short_vol[-2]>=long_vol[-2] and short_vol[-1]<long_vol[-1]):
            g.matrix[index][2]=True                                                                        
        else: 	
            g.matrix[index][2]=False                                                                    
        	
        	
        	
        close_macd = value[stock]['close'].as_matrix()
        	
        DIF, DEA, MACD = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)   
        	
        if DIF[-1]>DEA[-1] and DIF[-1]>0 and DEA[-1]>0:
              g.matrix[index][3]=True
              	
        else:	
              g.matrix[index][3]=False
    	
        	
        if short_ma[-1]>mid_ma[-2] and short_ma[-1]>short_ma[-2] and mid_ma[-1]>mid_ma[-2] and short_ma[-1]>mid_ma[-1] and mid_ma[-1]>long_ma[-1] and short_ma[-2]<=mid_ma[-2] and short_ma[-1]>mid_ma[-1]:
            g.matrix[index][4]=True                                                                          
        else: 	
            g.matrix[index][4]=False                                                                          
        if  (short_ma[-2]>=mid_ma[-2] and short_ma[-1]<mid_ma[-1]):
            g.matrix[index][5]=True                                                                          
        else: 	
            g.matrix[index][5]=False                                                                          
        	
        value_new=history([stock], ['close', 'open', 'low', 'turnover', 'high'], 110, '1d', False, None)
        upperband_new, middleband_new, lowerband_new = talib.BBANDS(np.asarray(value_new[stock]['close']), timeperiod=10, nbdevup=1.96, nbdevdn=1.96, matype=0)
       	
        flag_shangchuan=0
        flag_xiachuan=0
        flag_shangchuancishu=0
        flag_xiachuancishu=0
        for t in range(9,109):
            if flag_shangchuan==0:
                if long_ma[t]<upperband_new[t]:
                    flag_shangchuan=1
            if flag_shangchuan==1:
                if long_ma[t]>upperband_new[t]:
                    flag_shangchuancishu=flag_shangchuancishu+1
                    flag_shangchuan=2
            if flag_shangchuan==2:
                if long_ma[t]>upperband_new[t]:
                    flag_shangchuan=2
                if long_ma[t]<upperband_new[t]:
                    flag_shangchuan=0
        for t in range(9,109):
            if flag_xiachuan==0:
                if long_ma[t]>lowerband_new[t]:
                    flag_xiachuan=1
            if flag_xiachuan==1:
                if long_ma[t]<lowerband_new[t]:
                    flag_xiachuancishu=flag_xiachuancishu+1
                    flag_xiachuan=2
            if flag_xiachuan==2:
                if long_ma[t]<lowerband_new[t]:
                    flag_xiachuan=2
                if long_ma[t]>lowerband_new[t]:
                    flag_xiachuan=0
        	
        if flag_xiachuancishu!=0:
           	
            g.matrix[index][6]=True
            	
            	
        if flag_xiachuancishu==0:
	
            g.matrix[index][6]=False
            	
def request1(value,value1,value2,stock):
    	
    close = value[stock]['close'].as_matrix()
    	
    close1 = value1[stock]['close'].as_matrix()
   	
    close2 = value2[stock]['close'].as_matrix()
    	
    upperband, middleband, lowerband = talib.BBANDS(np.asarray(value2[stock]['close']), timeperiod=70, nbdevup=2, nbdevdn=2, matype=0)
    	
    	
    ma10=talib.SMA(close1,timeperiod=70)
    	
    open1 = value[stock]['open']
    close = value[stock]['close']
    low = value[stock]['low']
    	
    cut_ma=ma10[9:109]
    	
    cut_upperband=upperband[19:119]
    	
    cut_middleband=middleband[19:119]
    cut_lowerband=lowerband[19:119]
    flag_shangchuan=0
    flag_xiachuan=0
    flag_shangchuancishu=0
    flag_xiachuancishu
	
def request2(value,stock,index):
    	
    tur = value[stock]['close'].as_matrix()
    short_ma = talib.MA(tur, 5)  
    mid_ma = talib.MA(tur, 10)  
    long_ma = talib.MA(tur, 10) 
    if short_ma[-1]>mid_ma[-2] and short_ma[-1]>short_ma[-2] and mid_ma[-1]>mid_ma[-2] and short_ma[-1]>mid_ma[-1] and mid_ma[-1]>long_ma[-1] and short_ma[-2]<=mid_ma[-2] and short_ma[-1]>mid_ma[-1]:
           	
            g.matrix[index][2]=True   
           	
            log.info("多头排列True")
    else: 	
            g.matrix[index][2]=False
            log.info("多头排列False")
    if  (short_ma[-2]>=mid_ma[-2] and short_ma[-1]<mid_ma[-1]):
            g.matrix[index][3]=True
            log.info("多头排列True")
    else: 	
            g.matrix[index][3]=False
            log.info("多头排列False")
	
def request3(value,stock,index):
    	
    tur_5 = value[stock]['turnover'].as_matrix()
    tur_10 = value[stock]['turnover'].as_matrix()
    short_vol = talib.MA(tur_5, 5) 
    long_vol = talib.MA(tur_10, 10)  
	
    	
    if short_vol[-1]>short_vol[-2] and long_vol[-1]>long_vol[-2] and short_vol[-2]<=long_vol[-2] and short_vol[-1]>long_vol[-1]:
           g.matrix[index][4]=True                                                                          
           	
    else: 	
            g.matrix[index][4]=False                                                                      
	
    if  (short_vol[-2]>=long_vol[-2] and short_vol[-1]<long_vol[-1]):
            g.matrix[index][5]=True                                                                          
    else: 	
            g.matrix[index][5]=False                                                                       
def request4(value,stock,index):
    close_macd = value[stock]['close'].as_matrix()
    	
    DIF, DEA, MACD = talib.MACD(close_macd, fastperiod=12, slowperiod=26, signalperiod=9)
    	
    if DIF[-1] > DEA[-1] and DIF[-1] > 0 and DEA[-1] > 0:
        	
         g.matrix[index][6]=True
         log.info("MACD成功")
    else:	
         g.matrix[index][6]=False
         	
def add_request(value,stock):
    high = value['high'].as_matrix()
    low = value['low'].as_matrix()
    close = value['close'].as_matrix()
    atr = talib.ATR(high, low, close, 20)[-1]
   	
    if (record[record['symbol'] == stock]['last_buy_price'].empty):
        return False
    else:	
        last_price = float(record[record['symbol'] == stock]['last_buy_price'])  
        add_price = last_price + 0.5 * atr  
        add_unit = float(record[record['symbol'] == stock]['add_time'])  
        if close[-1] > add_price and add_unit <2:  
            return True
        else:	
            return False
def calcUnit(portfolio_value,ATR):
   	
    value = portfolio_value * 0.01     
    return int((value/ATR)/100)*100
