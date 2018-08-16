import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import copy
import random
import functools as ft

ONE_MON = relativedelta(months = 1, days = -1)# for minute data
# ONE_MON = relativedelta(months = 1) # for daily data
ONE_DAY = relativedelta(days= 1)
ANNUAL_TRADING_DAY = 252
SELL_COMIT = 0.00125
BUY_COMIT = 0.00025
COMIT_LIMIT = 3
SLIPPAGE = 0.0008
YYYYMMDD = "%Y%m%d"
Y_M_D_H_M_S = "%Y-%m-%d %H:%M:%S"

COVERED_VALUE = 1000000
R_F = 0.00125

class Stock(object):
    """"
    price_df:
    Time        Price
0   2017-06-01  66.43
1   2017-06-02  64.81
2   2017-06-05  64.09
    """""
    def __init__(self, price_df, stock_id, minute = False):
        self.price_df = price_df.set_index(['Time']).sort_index()
        self.trade_time = [x for x in self.price_df.index]
        self.stock_id = stock_id
        self.stock_hold = 0
        # TO DO: calc_vol
        self.vol = self.calc_vol(minute)

    # TO DO
    def __repr__(self):
        pass

    # TO DO
    def __str__(self):
        pass

    # len(Stock) will show the length of the data we have
    def __len__(self):
        return self.trade_time[0].date(), self.trade_date[-1].date()

    # Just becuase of the limited data
    # There are other ways to estimate vol
    def calc_vol(self,minute = False):
        if not minute:
            time_length = ANNUAL_TRADING_DAY
        else:
            time_length = ANNUAL_TRADING_DAY * 240
        # if we have enough data
        if len(self.price_df) >= (time_length + 1):
            per_return = [x/y - 1 for x, y in zip(self.price_df['Price'].iloc[-time_length:,],
                                                     self.price_df['Price'].iloc[-(time_length+1):,])]
        # if not enough data is given
        else:
            length = len(self.price_df)
            per_return = [x / y - 1 for x, y in zip(self.price_df['Price'].iloc[-(length - 1):, ],
                                                      self.price_df['Price'].iloc[-length:, ])]
        return np.std(per_return) * np.sqrt(time_length)

    def sell(self, number, time):
        self.stock_hold = self.stock_hold - number
        return self.price_df.loc[time,'Price']*(1-SLIPPAGE)*number*(1-SELL_COMIT)

    def buy(self,number, time):
        self.stock_hold = self.stock_hold + number
        return self.price_df.loc[time,'Price']*(1+SLIPPAGE)*number*(1+ BUY_COMIT)

    @staticmethod
    def calc_stocknumber(number):
        return number // 100 * 100

class CallOption(Stock):
    # option_dict: {stock:[expiries]}
    option_dict = {}
    def __init__(self, price_df, stock_id, expir, strike, now, minute, write_option = True):
        super().__init__(price_df, stock_id,minute=minute)
        self.prem = 0
        # datetime with only day to be none-zero
        self.expir = expir
        self.option_sell = 0
        # set strike price from outside
        self.strike = strike
        # now is the position of price_df index, the time we are now at
        self.now = now
        self.set_option_trade_time()
        self.length = len(self.option_trade_time)
        self.calc_c_delta(now)
        # option return = return from selling option - cost in hedging(buying/selling stocks)
        self.option_return = 0
        self.changebound = False
        self.dev_factor = 1
        if write_option:
            # Default way when option is initialized
            self.write_call(covered_value= COVERED_VALUE)

        if stock_id not in CallOption.option_dict:
            CallOption.option_dict[stock_id] = [expir]
        else:
            CallOption.option_dict[stock_id].append(expir)

    def set_option_trade_time(self):
        self.option_trade_time = [ x for x in self.trade_time if ( self.now.date()<=x.date()<=self.expir.date())]
        self.price_df = self.price_df.loc[self.option_trade_time]

    def __repr__(self):
        pass

    def __str__(self):
        pass

    # time/now is in the same format of price_df index
    # change the  c, delta
    def calc_c_delta(self, time):
        self.__T = (self.expir - time).days/365
        temp = self.vol*np.sqrt(self.__T)
        self.__d1 = (np.log(self.price_df.loc[time,'Price']/self.strike) + (R_F+ self.vol**2/2)*self.__T)/temp
        self.__d2 = self.__d1 - temp
        N_d1 = norm.cdf(self.__d1)
        self.__delta = N_d1
        N_d2 = norm.cdf(self.__d2)
        self.__c = self.price_df.loc[time,'Price']*N_d1 - self.strike*np.exp(-R_F*self.__T)*N_d2



    def write_call(self, covered_value=None,option_num=None,option_return = None):
        assert(covered_value or option_num or option_return)
        if covered_value:
            plus_option_sell = super().calc_stocknumber(covered_value/self.strike)
        elif option_num:
            plus_option_sell = super().calc_stocknumber(option_num)
        else:
            plus_option_sell = super().calc_stocknumber(option_return/self.__c)
        self.option_sell = self.option_sell + plus_option_sell
        self.prem = self.prem + plus_option_sell*self.__c*1.1

    def total_delta_hedge(self):
        stock_def = super().calc_stocknumber(self.stock_hold - self.__delta * self.option_sell)
        if stock_def > 0:
            outflow = self.sell(stock_def, self.now)
            self.option_return = self.option_return + outflow
        elif stock_def < 0:
            inflow = self.buy(-stock_def,self.now)
            self.option_return = self.option_return - inflow

    def reset_dev_factor(self):
        self.dev_factor = 1
        self.changebound = False

     # dev is always a positive number
    def dev_hedge(self,dev):
        curr_dev = self.__delta -self.stock_hold/self.option_sell
        # print (curr_dev)
        if curr_dev > dev:
            if self.changebound:
                self.reset_dev_factor()
            stock_o = self.__delta*self.option_sell - dev*self.option_sell
            stock_def =super().calc_stocknumber(stock_o - self.stock_hold)
            assert (stock_def >= 0)
            if stock_def >= COMIT_LIMIT/(BUY_COMIT* self.price_df.loc[self.now,'Price']*(1+SLIPPAGE)):
                outflow = self.buy(stock_def,self.now)
                self.option_return = self.option_return - outflow
        elif curr_dev < -1*dev:
            if self.changebound:
                self.reset_dev_factor()
            stock_o = self.__delta * self.option_sell+ dev * self.option_sell
            stock_def = super().calc_stocknumber(self.stock_hold-stock_o)
            assert(stock_def >= 0 )
            if stock_def >= COMIT_LIMIT/(SELL_COMIT* self.price_df.loc[self.now,'Price']*(1-SLIPPAGE)):
                inflow = self.sell(stock_def, self.now)
                self.option_return = self.option_return + inflow
        elif self.__delta >= 0.99 or self.__delta <= 0.01:
            if not self.changebound:
                self.changebound = True
            else:
                self.dev_factor = self.dev_factor/2
                curr_bound = dev*self.dev_factor
                if curr_dev > curr_bound:
                    stock_o = self.__delta * self.option_sell - curr_bound* self.option_sell
                    stock_def = super().calc_stocknumber(stock_o - self.stock_hold)
                    assert (stock_def >= 0)
                    if stock_def >= COMIT_LIMIT / (BUY_COMIT * self.price_df.loc[self.now, 'Price']*(1+SLIPPAGE)):
                        outflow = self.buy(stock_def, self.now)
                        self.option_return = self.option_return - outflow
                elif curr_dev < -1 * curr_bound:
                    stock_o = self.__delta * self.option_sell + curr_bound * self.option_sell
                    stock_def = super().calc_stocknumber(self.stock_hold - stock_o)
                    assert (stock_def >= 0)
                    if stock_def >= COMIT_LIMIT / (SELL_COMIT * self.price_df.loc[self.now, 'Price']*(1-SLIPPAGE)):
                        inflow = self.sell(stock_def, self.now)
                        self.option_return = self.option_return + inflow

        # print ('Stock_Hold ', self.stock_hold)

    def hold_hedge(self,per):
        stock_def = super().calc_stocknumber(self.stock_hold-self.option_sell*per)
        if stock_def >=100:
            inflow = self.sell(stock_def, self.now)
            self.option_return = self.option_return + inflow
        elif stock_def <= -100:
            outflow = self.buy(-stock_def, self.now)
            self.option_return = self.option_return - outflow


    def check_last_day(self):
        if self.now.day == self.expir.day:
            return True
        assert(self.now.day <= self.expir.day)

    def calc_return(self):
        price = self.price_df.loc[self.now, 'Price']
        if price - self.strike >= 0.01:
            stock_def = self.option_sell - self.stock_hold
            return self.option_return + self.option_sell*self.strike*(1-SELL_COMIT) -\
                   stock_def*price*(1 + BUY_COMIT)
        else:
            return self.option_return + self.stock_hold * price * (1 - SELL_COMIT)

    # a function used to iterate thourg the __option_trade_time
    # check length before using next_time
    def next_time(self, total_delta = False, dev = None, hold = None):
        self.now = self.option_trade_time.pop(0)
        if total_delta:
            self.calc_c_delta(self.now)
            self.total_delta_hedge()
        elif dev:
            self.calc_c_delta(self.now)
            self.dev_hedge(dev)
        elif hold:
            self.hold_hedge(hold)
        else:
            ValueError('Please set one of the hedging flag on')
        self.option_return = self.option_return + self.prem/self.length
        return self.now, self.calc_return()


    def __del__(self):
        # print('Remove {},{} from option_dict'.format(self.stock_id, self.expir))
        CallOption.option_dict[self.stock_id].remove(self.expir)



def get_stock_df(data_df, stock,minute = False):
    stock_df = data_df[data_df.loc[:,'Stock'] == stock]
    stock_df.drop(columns=['Stock'], inplace=True)
    if not minute:
        stock_df.loc[:,'Time'] = pd.Series([dt.datetime.strptime(str(x), YYYYMMDD) for x in stock_df['Time']],
                                     index=stock_df.index)

    stock_df.loc[:,'Time'] = pd.to_datetime(stock_df.loc[:,'Time'])
    return stock_df

def create_call_option(time, input_df, stock_id, minute =False):
    temp_df = input_df[input_df['Time'] >= time.date()]
    now = temp_df['Time'].iloc[0]
    strike = temp_df['Price'].iloc[0]
    expir = now + ONE_MON
    # print ('Creating call option: ({},{})'.format(stock_id,expir.date()))
    return CallOption(input_df, stock_id, expir, strike, now, minute=minute)



def result(total_delta = False, dev = None, hold = None, minute = False):
    if not minute:
        file_address = 'D:/file/SUMMER2018/yieldchain/stock_sim_data.csv'
        data_df = pd.read_csv(file_address, header=None)
        data_df.columns = ['Time', 'Stock', 'Price']
        time_list = pd.Series([dt.datetime.strptime(str(x), YYYYMMDD) for x in data_df['Time'].unique()])
    else:
        file_address = 'C:/Users/Sherry/PycharmProjects/option_hedge_obo/option-yieldchain/minute_data/minute_dt.csv'
        data_df = pd.read_csv(file_address,  index_col = [1])
        data_df.columns = ['Stock','Time','Price']
        time_list = [dt.datetime.strptime(str(x).replace("/", "-"), '%Y-%m-%d %H:%M:%S') for x in data_df['Time'].unique()]
        time_list.sort()
        time_L = copy.copy(time_list)
        time_l = []
        for x in time_L:
            if x.date() in time_l:
                time_list.remove(x)
            else:
                time_l.append(x.date())
        time_list = pd.Series(time_list)
    stock_list = data_df['Stock'].unique().tolist()
    stock_list.sort()
    option_dict = {}

    stock_num = len(stock_list)
    for time in time_list[0:stock_num]:

        stock = stock_list.pop(0)
        print ('Stock: ', stock)

        stock_df = get_stock_df(data_df, stock,minute)
        option = create_call_option(time, stock_df, stock, minute = minute)

        option_dict[(option.stock_id, option.expir)] = []
        time_index = []
        Re_list = []
        is_first_time = True
        while len(option.option_trade_time) > 0:
            if is_first_time:
                time, Re = option.next_time(total_delta = True)
                is_first_time = False
            else:
                time, Re = option.next_time(total_delta, dev, hold)

            time_index.append(time)
            Re_list.append(Re)
        option_dict[(option.stock_id, option.expir)] = pd.Series(Re_list, index=time_index)

        while (time_list.iloc[-1]>(option.expir + ONE_DAY)):
            option = create_call_option((option.expir + ONE_DAY), stock_df, stock, minute)

            option_dict[(option.stock_id, option.expir)] = []
            # print('Option expir: ', option.expir)
            time_index = []
            Re_list = []
            is_first_time = True
            while len(option.option_trade_time) > 0:
                if is_first_time:
                    time, Re = option.next_time(total_delta=True)
                    is_first_time = False
                else:
                    time, Re = option.next_time(total_delta, dev, hold)
                time_index.append(time)
                Re_list.append(Re)
            option_dict[(option.stock_id, option.expir)] = pd.Series(Re_list, index=time_index)
    return option_dict

# unit tests
def test():
    file_address = 'D:/file/SUMMER2018/yieldchain/stock_sim_data.csv'
    data_df = pd.read_csv(file_address, header=None)
    data_df.columns = ['Time', 'Stock', 'Price']
    stock_list = data_df['Stock'].unique()
    stock = stock_list[0]
    stock_df = data_df[data_df['Stock'] == stock]
    stock_df.drop(columns=['Stock'], inplace=True)
    stock_df.loc[:,'Time'] = pd.Series([dt.datetime.strptime(str(x), YYYYMMDD) for x in stock_df['Time']],
                                 index=stock_df.index)
    stock_df['Time'] = pd.to_datetime(stock_df['Time'])
    # stock_df = stock_df.set_index(['Stock', 'Time']).sort_index()
    def test_create_call_option(input_df,stock_id):
        now = input_df['Time'].iloc[0]
        strike = input_df['Price'].iloc[0]
        print ('We start at ', now)
        expir = now + ONE_MON
        return CallOption(input_df, stock_id, expir, strike, now)

    temp_option = test_create_call_option(stock_df, stock)

    def test_next_time(option):
        time, Re = option.next_time(total_delta=True)
        print ("At {}, return from option ({},{}) is {}".format(time,option.stock_id, option.expir.date(),Re ))

    test_next_time(temp_option)

    def test_del(option):
        del option
        print (CallOption.option_dict)

    test_del(temp_option)




def clean_option_dict(option_dict,total_delta=False,dev=None, hold = None, minute =False):
    df = pd.DataFrame(option_dict)
    # group it by stock
    stock_groups = df.groupby(axis = 1, level =0)
    for stock, stock_df in stock_groups:
        if total_delta:
            hedge = '_delta_'
        elif dev:
            hedge = '_dev'+ str(dev) +'_'
        elif hold:
            hedge = '_hold' + str(hold) + '_'
        else:
            ValueError('In side clean_option_dict. Please use the flags.')
        file_name = stock + hedge + '.csv'
        for column in stock_df.columns:
            index = stock_df[column].dropna().index[-1]
            stock_df.loc[index:,column]= stock_df.loc[index,column]
        # stock_df.to_csv(file_name)
        temp_df = stock_df.sum(axis=1)
        if minute:
            # file-name is the folder to store the result
            file_name =  'C:/Users/Sherry/PycharmProjects/option_hedge_obo/option-yieldchain/minute_data/' +file_name
        temp_df.to_csv(file_name)



if __name__ == '__main__':
    option_dict= result(dev=0.05,minute=True)
    clean_option_dict(option_dict, dev=0.05,minute=True)
    option_dict = result(dev=0.1, minute=True)
    clean_option_dict(option_dict, dev=0.1, minute=True)
    option_dict = result(dev=0.15,minute=True)
    clean_option_dict(option_dict, dev=0.15,minute=True)
    # option_dict = result(dev=0.4,minute=True)
    # clean_option_dict(option_dict, dev=0.4,minute=True)
    # option_dict = result(total_delta=True)
    # clean_option_dict(option_dict, total_delta =True)
    # option_dict = result(hold=0.5,minute=True)
    # clean_option_dict(option_dict, hold=0.5,minute=True)
    # prem_mean =  result(hold=0.5)
    # # print (prem_mean)
