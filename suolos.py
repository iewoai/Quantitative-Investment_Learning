import pandas as pd
import numpy as np
import datetime, time
import statsmodels.api as sm
import talib
import math

def init(context):
	set_log_level('warn')
	#智能选股
	get_iwencai('沪深 300')
	#设置交易周期
	context.pp=5
	#设置账户总资金
	context.cash=10000000
	#设置最大持股数
	context.max_stocks=10
	#设置筛选因子个数
	context.max_need=5
	#设置因子测试期数
	context.fpn=9
	context.st=0
	#设置检验因子所用的股票数
	context.samt=5
	#设置子账户
	set_subportfolios([{'cash':7000000,'type':'stock'},{'cash':3000000,'type':'future'}])
	log.warn(['context.cash:',context.cash])
	#设置计数起点
	context.cday=0
	#设置因子池
	context.need=['factor.pe',
	'factor.pb',
	'factor.pcf_cash_flow_ttm',
	'factor.ps',
	'factor.dividend_rate',
	'factor.market_cap',
	'factor.current_market_cap',
	'factor.capitalization',
	'factor.circulating_cap',
	'factor.current_ratio',
	'factor.equity_ratio',
	'factor.quick_ratio',
	'factor.tangible_assets_liabilities',
	'factor.tangible_assets_int_liabilities',
	'factor.net_debt_equity',
	'factor.long_term_debt_to_opt_capital_ratio',
	'factor.tangible_assets_net_liabilities',
	'factor.overall_income_growth_ratio',
	'factor.net_cashflow_psg_rowth_ratio',
	'factor.opt_profit_grow_ratio',
	'factor.total_profit_growth_ratio',
	'factor.diluted_net_asset_growth_ratio',
	'factor.net_cashflow_from_opt_act_growth_ratio',
	'factor.net_profit_growth_ratio',
	'factor.basic_pey_ear_growth_ratio',
	'factor.turnover_of_overall_assets',
	'factor.turnover_ratio_of_context_payable',
	'factor.turnover_of_current_assets',
	'factor.turnover_of_fixed_assets',
	'factor.cash_cycle',
	'factor.inventory_turnover_ratio',
	'factor.turnover_ratio_of_receivable',
	'factor.weighted_roe',
	'factor.overall_assets_net_income_ratio',
	'factor.net_profit_margin_on_sales',
	'factor.before_tax_profit_div_income',
	'factor.sale_cost_div_income',
	'factor.roa',
	'factor.ratio_of_sales_to_cost',
	'factor.net_profit_div_income',
	'factor.opt_profit_div_income',
	'factor.opt_cost_div_income',
	'factor.administration_cost_div_income',
	'factor.financing_cost_div_income',
	'factor.vr_rate',
	'factor.vstd',
	'factor.arbr',
	'factor.srdm',
	'factor.vroc',
	'factor.vrsi',
	'factor.cr',
	'factor.mfi',
	'factor.vr',
	'factor.mass',
	'factor.obv',
	'factor.pvt',
	'factor.wad',
	'factor.bbi',
	'factor.mtm',
	'factor.dma',
	'factor.ma',
	'factor.macd',
	'factor.expma',
	'factor.priceosc',
	'factor.trix',
	'factor.dbcd',
	'factor.dpo',
	'factor.psy',
	'factor.vma',
	'factor.vmacd',
	'factor.vosc',
	'factor.tapi',
	'factor.micd',
	'factor.rccd']
	#设置股票因子及其方向
	context.mfactors={'factor.pe':-1,
	'factor.pb':-1,
	'factor.pcf_cash_flow_ttm':-1,
	'factor.ps':-1,
	'factor.dividend_rate':1,
	'factor.market_cap':-1,
	'factor.current_market_cap':-1,
	'factor.capitalization':-1,
	'factor.circulating_cap':-1,
	'factor.current_ratio':1,
	'factor.equity_ratio':-1,
	'factor.quick_ratio':1,
	'factor.tangible_assets_liabilities':1,
	'factor.tangible_assets_int_liabilities':1,
	'factor.net_debt_equity':-1,
	'factor.long_term_debt_to_opt_capital_ratio':-1,
	'factor.tangible_assets_net_liabilities':1,
	'factor.overall_income_growth_ratio':1,
	'factor.net_cashflow_psg_rowth_ratio':1,
	'factor.opt_profit_grow_ratio':1,
	'factor.total_profit_growth_ratio':1,
	'factor.diluted_net_asset_growth_ratio':1,
	'factor.net_cashflow_from_opt_act_growth_ratio':1,
	'factor.net_profit_growth_ratio':1,
	'factor.basic_pey_ear_growth_ratio':1,
	'factor.turnover_of_overall_assets':1,
	'factor.turnover_ratio_of_context_payable':1,
	'factor.turnover_of_current_assets':1,
	'factor.turnover_of_fixed_assets':1,
	'factor.cash_cycle':-1,
	'factor.inventory_turnover_ratio':1,
	'factor.turnover_ratio_of_receivable':1,
	'factor.weighted_roe':1,
	'factor.overall_assets_net_income_ratio':1,
	'factor.net_profit_margin_on_sales':1,
	'factor.before_tax_profit_div_income':1,
	'factor.sale_cost_div_income':-1,
	'factor.roa':1,
	'factor.ratio_of_sales_to_cost':-1,
	'factor.net_profit_div_income':1,
	'factor.opt_profit_div_income':1,
	'factor.opt_cost_div_income':-1,
	'factor.administration_cost_div_income':-1,
	'factor.financing_cost_div_income':-1,
	'factor.vr_rate':1,
	'factor.vstd':-1,
	'factor.arbr':1,
	'factor.srdm':1,
	'factor.vroc':1,
	'factor.vrsi':1,
	'factor.cr':1,
	'factor.mfi':1,
	'factor.vr':1,
	'factor.mass':1,
	'factor.obv':1,
	'factor.pvt':1,
	'factor.wad':1,
	'factor.bbi':1,
	'factor.mtm':1,
	'factor.dma':1,
	'factor.ma':1,
	'factor.macd':1,
	'factor.expma':1,
	'factor.priceosc':1,
	'factor.trix':1,
	'factor.dbcd':1,
	'factor.dpo':1,
	'factor.psy':1,
	'factor.vma':1,
	'factor.vmacd':1,
	'factor.vosc':1,
	'factor.tapi':1,
	'factor.micd':1,
	'factor.rccd':1}
	context.lastlong={}.fromkeys(context.need) #超配股票池
	context.lastshort={}.fromkeys(context.need) #低配股票池
	context.winr={}
	context.ir={}
	context.incomer={}
	#设置默认因子
	context.realneed=['factor.pe','factor.weighted_roe','factor.net_cashflow_from_opt_act_growth_ratio',
	'factor.overall_income_growth_ratio','factor.turnover_ratio_of_context_payable']
	#设置打分机制
	context.score=5
	#定期运行函数
	run_monthly(func=reallocate,date_rule=8)

def reallocate(context,bar_dict):
	log.warn('……每月第 8 交易日/起始日……')
	tempstocks=give_me_stocks(context,bar_dict,is_re=1)
	 #前maxstocks只股票带综合分数
	#log.info('This is the stocks for choosing')
	#log.info(tempstocks)
	nstocks=[]
	for i in range(len(tempstocks)):
		nstocks.append(tempstocks[i][0])
	#起手均分现金，之后均分持仓规模
	log.warn('刷新持仓股票')
	if(context.cday==0):
		cash=context.cash/context.max_stocks
		log.warn(['context.cash:',context.cash])
	else:
		# 分配可用资金
		cash=context.portfolio.portfolio_value/context.max_stocks
	# 原股票池中的股票不在新股票池中则清0
	for i in list(context.portfolio.positions.keys()):
		if(i not in nstocks[:context.max_stocks]):
			order_target(i,0)
	for i in nstocks:
		order_target_value(i,cash)
		if(len(context.portfolio.positions.keys())==context.max_stocks):
			break
	log.warn('刷新完毕')
	log.warn('期货操作')
	Astocks=[] #用于计算期货份数的列表，存放持股分数
	tstocks=dict(tempstocks)
	ctr=0
	for i in list(context.portfolio.positions.keys()):
		if i not in tstocks:
			continue
		Astocks.append((i,tstocks[i]))
		ctr+=1
	#log.info(ctr)
	Amount=give_me_amount(context,bar_dict,*Astocks)
	#log.info(['The number of Amount is',Amount])
	code=get_future_code('IF','next_month') #下月期货
	if(len(context.portfolio.stock_account.positions)==0): #若子账户期货为空，说明当月期货已经交割，入手下月
		order(code,Amount,pindex=1,type='short')
		log.warn('下月期货入手')
	else:
		current_code=list(context.portfolio.stock_account.positions)[0] #手头期货
		number=context.portfolio.stock_account.positions[current_code].total_amount #手头期货份数
		if(code==current_code): #如果手头的是下月期货，直接调仓
			if Amount>number:
				order(current_code,-number+Amount,pindex=1,type='short') #做空期货，卖出
				log.warn('做空期货')
			else:
				order(current_code,number-Amount,pindex=1,type='long') #做多期货，买回
				log.warn('做多期货')
		else:
			order(current_code,-number,pindex=1,type='long') #平仓买回
			order(code,Amount,pindex=1,type='short') #做空下月期货
			log.warn('平仓买回 AND 做空下月期货')
	log.warn('期货操作完毕')

def give_me_amount(context,bar_dict,*tempstocks):
	log.warn('获取期货买卖份数')
	#计算 Beta 值
	beta=[]
	percent=[]
	nstocks=[]
	rstocks=[]
	sum1=0.0
	for i in range(len(tempstocks)):
		sum1+=tempstocks[i][1]
		rstocks.append(tempstocks[i][1])
		nstocks.append(tempstocks[i][0])
	#计算权重
	for i in range(len(rstocks)):
		percent.append(rstocks[i]/sum1)
	for i in nstocks: #迭代股票池
		iclose=get_price(['000300.SH',i],None,get_datetime().strftime('%Y%m%d'),'1d',['close'],False,None,220) #证券
		xclose=iclose['000300.SH']['close']
		yclose=iclose[i]['close']
		if(len(yclose)==0):
			#log.info('糟糕,yclose 长度是 0')
			beta.append(0)
			continue
		if(len(xclose)!=len(yclose)):
			#log.info(['iclose 是',iclose])
			l=min(len(xclose),len(yclose))
			xclose=xclose[-l:]
			yclose=yclose[-l:]
		x=np.array(xclose)
		y=np.array(yclose)
		rm=(x[1:]-x[0:-1])/x[0:-1]
		r1=(y[1:]-y[0:-1])/y[0:-1]
		est=sm.OLS(r1,sm.add_constant(rm))
		est=est.fit() #单只股票的 beta
		beta.append(est.params[1])
	Beta=0.0
	for i in range(len(beta)):
		Beta+=beta[i]*percent[i] #Beat 等于因子打分权重之和
	#log.info(['Beta 是',Beta])
	close=history('000300.SH',['close'],1,'1d') #期货开盘价
	#log.info(['沪深 300 股指期货的开盘价是',close])
	cash=context.portfolio.portfolio_value*1.0/context.max_stocks
	log.warn(['Beta and cash and 期货价：',Beta,cash,close['close'][0]])
	Amount=math.ceil((Beta*(cash*context.max_stocks))/(300*close['close'][0]))
	log.warn(['Amount 是',Amount])
	log.warn('期货买卖份数获取完毕')
	return int(Amount*1)

def give_me_need(context,bar_dict):
	log.warn('获取新建因子')
	n=len(context.mfactors)
	querc=','.join(context.need)
	#log.info(querc)
	q=query(querc).filter(factor.symbol.in_(context.iwencai_securities),factor.date==get_datetime().strftime('%Y-%m-%d'))
	df=get_factors(q).fillna(0)
	#log.info(df)
	df['factor_symbol']=context.iwencai_securities[:len(df)]
	n=len(df)
	m=len(df.columns)-1
	for k in range(m):
		tempdf=df.sort_values(df.columns[k])
		name=list(tempdf['factor_symbol'])
		#log.info('排序后的 df')
		#log.info(tempdf)
		#log.info(name)
		f='factor.'+df.columns[k]
		#log.info(f)
		if context.mfactors[f]<0:
			context.lastlong[f]=name[:context.samt*2]
			context.lastshort[f]=name[-2*context.samt:]
		else:
			context.lastlong[f]=name[-2*context.samt:]
			context.lastshort[f]=name[:2*context.samt]
		#log.info(['long',len(context.lastlong[f])])
		#log.info(['short',len(context.lastshort[f])])
		value=get_price(context.lastlong[f],None,get_datetime().strftime("%Y%m%d"),'22d',['close'],True,None,context.fpn)
		rsum1=np.zeros((context.fpn-1,1))
		ctr=0
		for stk in context.lastlong[f]:
			#log.info(['这是 value[stk]',value[stk]])
			x=np.array(value[stk])
			#log.info(x)
			if(len(x)!=context.fpn):
				#log.info('超配之力不从心')
				continue
			if(ctr==context.samt):
				break
			rsum1+=(x[1:]-x[:-1])/x[:-1]
			ctr=ctr+1
		#log.info([rsum1,ctr])
		value=get_price(context.lastshort[f],None,get_datetime().strftime("%Y%m%d"),'22d',['close'],True,None,context.fpn)
		rsum2=np.zeros((context.fpn-1,1))
		ctr=0
		for stk in context.lastshort[f]:
			x=np.array(value[stk])
			#log.info(x)
			if(len(x)!=context.fpn):
				#log.info('低配之力不从心')
				continue
			if(ctr==context.samt):
				break
			rsum2+=(x[1:]-x[:-1])/x[:-1]
			ctr=ctr+1
		#log.info([rsum2,ctr])
		context.winr[f]=((np.sum(rsum1-rsum2>0)/(context.fpn-1)))
		context.ir[f]=(np.mean(rsum1-rsum2)/np.std(rsum1-rsum2))
		context.incomer[f]=(np.mean(rsum1))
		#log.info(context.winr)
		#log.info(context.ir)
	#log.info(context.winr)
	#log.info(context.ir)
	#log.info(context.incomer)
	score={}
	score=score.fromkeys(list(context.need),0.0)
	#log.info(score)
	temp=sorted(context.winr.items(),key=lambda item:item[1])
	for i in range(len(temp)):
		score[temp[i][0]]+=0.4*i
	temp=sorted(context.ir.items(),key=lambda item:item[1])
	for i in range(len(temp)):
		score[temp[i][0]]+=0.4*i
	#log.info(score)
	temp=sorted(context.incomer.items(),key=lambda item:item[1])
	for i in range(len(temp)):
		score[temp[i][0]]+=0.2*i
	#log.info(score)
	tempscore=sorted(score.items(),key=lambda item:item[1],reverse=True)
	#log.info(tempscore)
	realneed=[]
	for i in range(context.max_need):
		realneed.append(tempscore[i][0])
	log.info(realneed)
	log.warn('因子获取完毕')
	return realneed

def give_me_stocks(context,bar_dict,is_re):
	log.warn('获取新建股票池')
	samp=context.iwencai_securities
	if(is_re==1):
		context.realneed=give_me_need(context,bar_dict)
		#从数据库查询因子,获取 q
	#log.info(context.realneed)
	need=','.join(context.realneed)
	#log.info(['这是 need',need])
	q=query(need).filter(factor.symbol.in_(samp),factor.date==get_datetime().strftime('%Y-%m-%d') )
	df=get_factors(q).fillna(0) #结合当天时间，形成 dateframe,缺失值填充为 0
	samp=samp[:len(df)]
	df['factor_symbol']=samp
	n=len(df) #获取行数，即总共获取的股票数目
	m=len(df.columns)-1 #获取列数(因子数目)=总列数-1
	stocks={}
	stocks=stocks.fromkeys(samp,0.0) #预设打分股票池
	for i in range(m): #先对因子一列从小到大进行排序，方便打分 ,逐列展开
		f='factor.'+df.columns[i]
		if(context.mfactors[f]<0):
			tempdf=df.sort_values(df.columns[i]) #从小到大
		else:
			tempdf=df.sort_values(df.columns[i],ascending=False)
		security=list(tempdf['factor_symbol']) #排序后的股票代码，重新构建序列
		for j in range(n): #确定 n，即确定的股票代码
			name=security[j]#字符串格式，股票的代码
			rank=int((j/(n/context.score))) #确定等级
			stocks[name]+=(5-rank)
	tempstocks=sorted(stocks.items(),key=lambda item:item[1],reverse=True) #给已经拥有分数的股票池从大到小排序
	log.info(tempstocks)
	log.warn('新建股票池获取完毕')
	return tempstocks[:2*context.max_stocks]

def boll(context,bar_dict):
	log.warn('布林线检测')
	flag=0 #设置触发变量
	up=[[],[]]
	mid=[[],[]]
	low=[[],[]]
	cprice=[[],[]]
	last=-6
	now=-1
	for i in list(context.portfolio.positions.keys()):
		# 获取历史收盘价
		values=history(i,['close'],21,'1d',False,None).fillna(0)
	upper,middle,lower=talib.BBANDS(values['close'].values,timeperiod=15,nbdevup=2,nbdevdn=2,matype=0)
	up[0].append(upper[last])
	up[1].append(upper[now])
	mid[0].append(middle[last])
	mid[1].append(middle[now])
	low[0].append(lower[last])
	low[1].append(lower[now])
	cprice[0].append(values['close'][last])
	cprice[1].append(values['close'][now])
	up=np.mean(up,1)
	mid=np.mean(mid,1)
	low=np.mean(low,1)
	cprice=np.mean(cprice,1)
	#log.info(up)
	if(up[1]>up[0] and low[1]<low[0]): #开轨
		if(cprice[1]<cprice[0] and cprice[1]>mid[1]):
			flag=1
			log.warn('开轨')
	elif(up[1]<up[0] and low[1]>low[0]): #收轨
		if((cprice[0]<up[0] and cprice[1]>up[1]) or (cprice[0]>low[0] and
			cprice[1]<low[1])):
			flag=1
			log.warn('收轨')
	elif((up[1]-up[0])*(low[1]-low[0])>0): #三轨同向
		if(cprice[0]>mid[0] and cprice[1]<mid[1]):
			flag=1
			log.warn('三轨同向')
	log.warn('布林线检测完毕')
	if(flag==1):
		log.warn('布林线动作触发')
	else:
		log.warn('无动作')
	return flag

def handle_bar_dict(context,bar_dict):
	ctime=get_datetime() #今日时间
	if(context.cday==0):
		context.st=ctime
	dd=ctime-context.st
	if(int(dd.days)>=context.cday): #如果超过计时点，就进行判断
		if(context.cday==0):
			reallocate(context,bar_dict) #月初处理
			context.cday+=context.pp #调整计时起点
		else: #非初始日
			#布林线通道
			log.warn([context.pp,'/天一检测'])
			if(boll(context,bar_dict)==1 or int(dd.days)%(15)==0): #布林线判断出要构建新的股票池或者距离上一次构建达到达到 15 天
				if(int(dd.days)%(15)==0):
					log.warn('15 天/构建')
				tempstocks=give_me_stocks(context,bar_dict,is_re=0)
				nstocks=[]
				for i in range(len(tempstocks)):
					nstocks.append(tempstocks[i][0])
				log.info(nstocks)
				cash=context.portfolio.portfolio_value/context.max_stocks
				log.warn('刷新持仓股票')
				ctr=0
				for i in list(context.portfolio.positions.keys()):
					if(i not in nstocks[:context.max_stocks]):#股票不存在现有账户中，此回合需卖出
						order_target(i,0) #全部卖出
						ctr+=1
				log.warn(['卖出股票数:',ctr])
				for i in nstocks:
					order_target_value(i,cash) #调整股票使满足目标金额
					if(len(context.portfolio.positions.keys())==context.max_stocks):
						break
				log.warn('进行期货操作')
				Astocks=[]
				tstocks=dict(tempstocks)
				for i in list(context.portfolio.positions.keys()):
					if i not in tstocks:
						continue
					Astocks.append((i,tstocks[i]))
				Amount=give_me_amount(context,bar_dict,*Astocks)
				if(len(context.portfolio.stock_account.positions)==0): #若子账户期货为空，说明当月期货已经提前交割，入手下月
					code=get_future_code('IF','next_month')
					order(code,Amount,pindex=1,type='short')
					log.warn('下月期货入手')
				else:
					current_code=list(context.portfolio.stock_account.positions)[0] #手头期货
					number=context.portfolio.stock_account.positions[current_code].total_amount
					if Amount>number:
						log.warn('做空期货')
						order(current_code,-number+Amount,pindex=1,type='short')
					else:
						log.warn('做多期货')
						order(current_code,number-Amount,pindex=1,type='long')
	context.cday+=context.pp #调整计时起点
	log.warn([context.pp,'/天一检测(15 天检测)完毕'])

