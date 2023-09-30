from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,TimeSeriesSplit
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import datetime as dt
import cbpro
import matplotlib.pyplot as plt 
import time
from web3.middleware import geth_poa_middleware
from web3.gas_strategies.time_based import medium_gas_price_strategy
from eth_account.messages import encode_defunct
import pickle
from web3 import Web3, constants
from web3 import middleware
# from flask import Flask, flash, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
import json
import librosa
infura_url = "https://mainnet.infura.io/v3/5c9cb0b35a2742659dec6fc7680c16c4"
web3 = Web3(Web3.HTTPProvider(infura_url))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

'''RELEASE CONTRACT WITHDRAWL TO COINBASE WALLET FOR AI BOT TRADING''' 


# address = '0x1A0F33bBc5c7bA83f490cdB6C13ee50e1C851908'
# abi = '[{"inputs":[{"internalType":"string","name":"_name","type":"string"},{"internalType":"string","name":"_symbol","type":"string"},{"internalType":"string","name":"_initBaseURI","type":"string"},{"internalType":"string","name":"_initNotRevealedUri","type":"string"}],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"","type":"address"},{"indexed":false,"internalType":"uint256","name":"","type":"uint256"}],"name":"AddressCalled","type":"event"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address[]","name":"","type":"address[]"}],"name":"AddressesAdded","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"previousOwner","type":"address"},{"indexed":true,"internalType":"address","name":"newOwner","type":"address"}],"name":"OwnershipTransferred","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Transfer","type":"event"},{"inputs":[],"name":"PRICE","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"_mintSingleNFT1","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"addressMintedBalance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"baseExtension","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"baseURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"_count","type":"uint256"}],"name":"claimTauros","outputs":[],"stateMutability":"payable","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getCurrentToken","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"maxMintAmount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"maxSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"notRevealedUri","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"bool","name":"_state","type":"bool"}],"name":"pause","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"paused","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"renounceOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"reserveNFTs","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"reveal","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"revealed","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"_newBaseExtension","type":"string"}],"name":"setBaseExtension","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"_newBaseURI","type":"string"}],"name":"setBaseURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"string","name":"_notRevealedURI","type":"string"}],"name":"setNotRevealedURI","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_newPRICE","type":"uint256"}],"name":"setPRICE","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"_newmaxMintAmount","type":"uint256"}],"name":"setmaxMintAmount","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"tokenId","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"_Id","type":"uint256"}],"name":"tokenURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"totalSupply","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"transferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_owner","type":"address"}],"name":"walletOfOwner","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"withdraw","outputs":[],"stateMutability":"payable","type":"function"}]'
# abi = json.loads(abi)
# mint_acct ='Enter Address'


# web3.eth.mint_acct = mint_acct
# mint_key = 'Enter Private Key'
# nonce =  web3.eth.getTransactionCount(mint_acct)
# contract = web3.eth.contract(address=address, abi=abi)


# withdraw = contract.functions.withdrawAI().buildTransaction({'chainId': 1,'gas':250000,'gasPrice': web3.toWei('32.875000015', 'gwei'), 'nonce': nonce})
# signed_tx2 = web3.eth.account.signTransaction(withdraw, mint_key)
# latestBlock()
# tx_hash = web3.eth.sendRawTransaction(signed_tx2.rawTransaction)
# print("withdrawing")
# tx_hash2 = web3.toHex(tx_hash)
# print(tx_hash2)
# web3.eth.waitForTransactionReceipt(tx_hash2)

## Transfer to CB-pro Address AS ETH to allow the machine learning model to trade.. will integrate LSTM as staking contract accumulates liquidity overtime , more liquidity ?

# result = web3.eth.getBalance(mint_acct) 
# result = web3.fromWei(result,'ether')
# print(result)

### CBPRO API KEYS
apiKey = "ENTER CBPRO APIKEY"
apiSecret = "ENTER CBPRO SECRET"
passphrase = "ENTER PASSPHRASE"

auth_client = cbpro.AuthenticatedClient(apiKey,apiSecret,passphrase)
auth_client_df = pd.DataFrame(auth_client.get_accounts()) 

def current_price(currency):
    currency = currency
    Period = 60 #[60, 300, 900, 3600, 21600, 86400]
    historicData = auth_client.get_product_historic_rates(currency, granularity=Period)
    #     print(historicData)
            # Make an array of the historic price data from the matrix
    price = np.squeeze(np.asarray(np.matrix(historicData)[:,4]))
            # Wait for 1 second, to avoid API limit
    time.sleep(1)
            # Get latest data and show to the user for reference
    newData = auth_client.get_product_ticker(product_id=currency)
    currentPrice=newData['price']
    print('currency: {}'.format(currency))
    print('current_price {} \n\n'.format(currentPrice))
    return currentPrice




currency = 'ETH-USD'
Period = 60 #[60, 300, 900, 3600, 21600, 86400]
historicData = auth_client.get_product_historic_rates(currency, granularity=Period)
historicData = pd.DataFrame(historicData, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
historicData.to_csv('data/{}.csv'.format(currency), index=False)

def history(currency):
    currency = currency
    Period = 60 #[60, 300, 900, 3600, 21600, 86400]      
    historicData = auth_client.get_product_historic_rates(currency, granularity=Period)
    historicData = pd.DataFrame(historicData,columns=['time','open','high','low','close','volume'])
    price = historicData['high']
            # Wait for 1 second, to avoid API limit
#     pd.to_csv('data/{}.csv'.format(currency),historicData)
    time.sleep(1)
    return historicData
def profit_target(token,current_holdings,target_percentage): 
    token = token
    print('\n\n {} target'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * float(target_percentage)
    total_target = current_holdings+target_percentage
    print('{} profit target {}, == {}'.format(token,target_percentage,total_target))
    return target_percentage 
def loss(token,current_holdings,loss):
    token = token
    print('\n\n {} loss'.format(token))
    current_holdings = current_holdings
    target_percentage = current_holdings * float(loss)
    total_loss = current_holdings-target_percentage
    print('{} stop loss {}, == {}'.format(token,target_percentage,total_loss)) 
    return target_percentage 

currency = input('Enter currency pair (dnt-usd)')
current_price = current_price(currency) 

print('amount to trade is betweet 0.03<=>0.1 ETH')
auth_client_currency = np.random.uniform(0.03,0.1)
print('available {} for trading: {}\n\n'.format(currency,auth_client_currency))
amount = auth_client_currency



from coinbase.wallet.client import Client
import json
import pandas as pd 
# Before implementation, set environmental variables with the names API_KEY and API_SECRET
api_key = 'ENTER CB API KEY'
api_secret = 'ENTER CB API SECRET'
client = Client(api_key, api_secret)
# client.get_primary_account()

current_balance = float(client.get_buy_price(currency_pair='ETH-USD')['amount'])* float(auth_client_currency)
print('current balances: {}\n\n'.format(current_balance))

print('-->PROFIT TARGETS:')
tar = profit_target(currency,current_balance, .3) 
tar2 = profit_target(currency,current_balance, .15) 
tar3 = profit_target(currency,current_balance, .03) 
print('\n\n -->MAX LOSS:')
loss = loss(currency,current_balance, .1)


def latestBlock():
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    web3.eth.getBlock('latest')
    a = web3.eth.getBlock('latest')
    import time 
    time.sleep(3)
    return a


import os
import pandas as pd 
import librosa
def strip(x, frame_length, hop_length):
    # Compute RMSE.
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    # Identify the first frame index where RMSE exceeds a threshold.
    thresh = 0.01
    frame_index = 0
    while rmse[0][frame_index] < thresh:
        frame_index += 1
        
    # Convert units of frames to samples.
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)
    
    # Return the trimmed signal.
    return x[start_sample_index:]


def read_data():
    import pandas as pd 
    import matplotlib.pyplot as plt
    import IPython.display as ipd
    import pandas as pd
    import librosa
    import keras
    import librosa.display
    import time
    # %pylab inline
    import glob
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    import warnings
    import numpy as np
    import plotly.express as px
    from sklearn.decomposition import PCA, FastICA
    import plotly.graph_objects as go
    warnings.filterwarnings('ignore')

    if not os.path.exists("images"):
        os.mkdir("images")
        
    '''call chosen currency via coinbase API'''
#     currency = input('enter currency pairing')
#     iteration=1
#     print('Analyzing and predicting {} \n\n'.format(currency))
    while True:
        '''Call Chosen Currency History'''
        data = pd.read_csv('ETH-USD.csv')  #currency
        data.to_csv('currency_high.csv',index=False)
        a0 = pd.read_csv('currency_high.csv')

        '''Isolate Features From Respective Currency Data'''
        # a0 = a0.drop(['Unnamed: 0'], axis=0 )
        b0 = a0['open']
        c0 = a0['high']
        d0 = a0['low']
        e0 = a0['close']
        f0 = a0['volume']
        i0 = a0['time']
        
#         '''Visualize Interactive Data Via plotly'''
#         fig = go.Figure(data=[go.Candlestick(x=a0, 
#                                              open=b0, 
#                                              high=c0, 
#                                              low=d0, 
#                                              close=e0)])
#         print('Use The Slider to Adjust and Zoom')
#         fig.show()
        
        order_book = auth_client.get_product_order_book('ETH-USD')
        print('Order Book \n\n',order_book)
        
        '''Averaging Isolated Price Data'''
        avg=np.average(b0)   
        avg1=np.average(c0) 
        avg2=np.average(d0)
        avg3=np.average(e0)
        print('avg OPEN : {}, avg High : {}, avg LOW : {}, avg CLOSE : {}\n\n'.format(avg,avg1,avg2,avg3)) 

        '''Display x=Volume , y = Open Signals '''
#         data = dict(
#             number=[b0,c0,d0,e0],
#             stage=[ "Open", "High", "Low", "Close"])
#         fig = px.funnel(a0, x=f0, y=b0)
#         fig.show()

        '''Display x=Time , y = Open Signals '''
#         data = dict(
#             number=[b0,c0,d0,e0],
#             stage=[ "Open", "High", "Low", "Close"])
#         fig = px.funnel(a0, x=i0, y=b0)
#         fig.show()
        
        '''currency volume'''
        background = f0
        '''Time'''
        x = i0
        '''Open'''
        y = b0
        '''Creating isolated datasets'''
        x_df = pd.DataFrame(x)  
        y_df = pd.DataFrame(y) 
        background_df = pd.DataFrame(background) 
        x = x_df 
        y = y_df 

        '''Extract and rejoin volume,time,open data'''
        background = background_df
        extract = x.join(background) 
        extract = extract.join(y)
        extract 
       
        data = extract.to_csv('data/extraction_data.csv') 
        data = pd.read_csv('data/extraction_data.csv')
        data = data.drop(['Unnamed: 0'],axis=1) 
        data 

        X= i0 #time
        y = data['open'] 
        background = data['volume']

#         plt.plot(y) 
#         # plt.plot(x)
#         plt.hist2d(i0,data['open']) 
#         plt.hist2d(i0,data['volume']) 

        '''Restructure data so algorithim can read data and udate sample rates for linear regression '''
        data = np.squeeze(np.asarray(np.matrix(data)[:,1])) 
        sam_rate = np.squeeze(np.asarray(np.matrix(data)[:,0])) 
        D = np.abs(librosa.stft(data))**2
        S = librosa.feature.melspectrogram(data,sr=sam_rate,S=D,n_mels=512)
        log_S1 = librosa.power_to_db(S,ref=np.max)

#         plt.figure(figsize=(12,4))
#         librosa.display.specshow(log_S1,sr=sam_rate,x_axis='time',y_axis='mel')
#         plt.title('MEL POWER SPECTOGRAM')

#         plt.colorbar(format='%+02.0f dB')

#         plt.tight_layout()
#         plt.show()
        librosa.get_duration(data, sam_rate)
        # h_l = 500
        # f_l = 0
        h_l = 256 
        f_l = 512

        #Create Linear regression models for Time Series Cross Validation
        reg = LinearRegression(n_jobs=-1, normalize=True ) 
        reg1 = LinearRegression(n_jobs=-1, normalize=True ) 
        reg2 = LinearRegression(n_jobs=-1, normalize=True ) 
        reg3 = LinearRegression(n_jobs=-1, normalize=True ) 
        reg4 = LinearRegression(n_jobs=-1, normalize=True ) 

        first_iteration = a0
        time = first_iteration['time']
        
        '''Open Prediction Model'''
        y_open= first_iteration['open'] 
        X_open = first_iteration.drop(['open'],axis=1) 
        mini = MinMaxScaler() 
        X_open = mini.fit_transform(X_open) 
        Xo_train,Xo_test,yo_train,yo_test = train_test_split(X_open,y_open,test_size=.45,shuffle=False) 
        reg.fit(Xo_train,yo_train)
        pickle.dump(reg, open('models/open_model.pkl','wb'))
        reg = pickle.load(open('models/open_model.pkl','rb'))
        
        tscv = TimeSeriesSplit(n_splits=5)
#         print(tscv)  
        TimeSeriesSplit(max_train_size=None, n_splits=4)
        for train_index, test_index in tscv.split(X_open):
            print("TRAIN:", train_index, "TEST:", test_index)
            Xo_train, Xo_test = X_open[train_index], X_open[test_index]
            yo_train, yo_test = y_open[train_index], y_open[test_index]
    #     from sklearn.externals import joblib
    #     joblib.dump(reg, 'models/tsco_1.pkl')
        bata =  data
#         bata.shape
        date = i0 
        future_x_open = X_open 
        X_open = X_open[-1:] 
        bata = bata
        date = i0 
        date = date.tail()
        #bata = bata.tail() 
        date = i0
        y_open = reg.predict(future_x_open) 
        print('accuracy {}'.format(reg.score(Xo_test,yo_test)))
        y_open_df = pd.DataFrame(y_open) 
        y_open_df.to_csv('open_pred.csv')

        '''High Prediction Model'''
        y_high= first_iteration['high']
        X_high = first_iteration.drop(['high'],axis=1) 
        mini = MinMaxScaler() 
        X_high = mini.fit_transform(X_high) 
        Xh_train,Xh_test,yh_train,yh_test = train_test_split(X_high,y_high,test_size=.45,shuffle=False) 
        reg1.fit(Xh_train,yh_train)
        pickle.dump(reg1, open('models/high_model.pkl','wb'))
        reg1 = pickle.load(open('models/high_model.pkl','rb'))
                    
        tscv = TimeSeriesSplit(n_splits=5)
#         print(tscv)  
        TimeSeriesSplit(max_train_size=None, n_splits=4)
        for train_index, test_index in tscv.split(X_high):
            print("TRAIN:", train_index, "TEST:", test_index)
            Xh_train, Xh_test = X_high[train_index], X_high[test_index]
            yh_train, yh_test = y_high[train_index], y_high[test_index]
    #     from sklearn.externals import joblib
    #     joblib.dump(reg, 'models/tscv_1.pkl')
        bata =  data
#         bata.shape
        date = i0 
        future_x_high = X_high 
        X_high = X_high[-1:] 
        bata = bata
        date = i0 
        date = date.tail()
        #bata = bata.tail() 
        date = i0
        y_high = reg1.predict(future_x_high) 
        print('accuracy {}'.format(reg1.score(Xh_test,yh_test)))
        y_high_df = pd.DataFrame(y_high) 
        y_high_df.to_csv('high_pred.csv')

        '''Low Prediction Model'''
        y_low= first_iteration['low']
        X_low = first_iteration.drop(['low'],axis=1) 
        mini = MinMaxScaler() 
        X_low = mini.fit_transform(X_low) 
        Xl_train,Xl_test,yl_train,yl_test = train_test_split(X_low,y_low,test_size=.45,shuffle=False) 
        reg2.fit(Xl_train,yl_train)
        pickle.dump(reg2, open('models/low_model.pkl','wb'))
        reg2 = pickle.load(open('models/low_model.pkl','rb'))
                    
        tscv = TimeSeriesSplit(n_splits=5)
#         print(tscv)  
        TimeSeriesSplit(max_train_size=None, n_splits=4)
        for train_index, test_index in tscv.split(X_low):
            print("TRAIN:", train_index, "TEST:", test_index)
            Xl_train, Xl_test = X_low[train_index], X_low[test_index]
            yl_train, yl_test = y_low[train_index], y_low[test_index]
    #     from sklearn.externals import joblib
    #     joblib.dump(reg, 'models/tscv_1.pkl')
        bata =  data
#         bata.shape
        date = i0 
        future_x_low = X_low 
        X_low = X_low[-1:] 
        bata = bata
        date = i0 
        date = date.tail()
        #bata = bata.tail() 
        date = i0
        y_low = reg2.predict(future_x_low) 
        print('accuracy {}'.format(reg2.score(Xl_test,yl_test)))
        y_low_df = pd.DataFrame(y_low) 
        y_low_df.to_csv('low_pred.csv')

        '''Close Prediction Model'''
        y_close= first_iteration['close']
        X_close = first_iteration.drop(['close'],axis=1) 
        mini = MinMaxScaler() 
        X_close = mini.fit_transform(X_close) 
        Xc_train,Xc_test,yc_train,yc_test = train_test_split(X_close,y_close,test_size=.45,shuffle=False) 
        reg3.fit(Xc_train,yc_train)
    
        pickle.dump(reg3, open('models/close_model.pkl','wb'))

        reg3 = pickle.load(open('models/close_model.pkl','rb'))
        tscv = TimeSeriesSplit(n_splits=5)
#         print(tscv)  
        TimeSeriesSplit(max_train_size=None, n_splits=4)
        for train_index, test_index in tscv.split(X_close):
            print("TRAIN:", train_index, "TEST:", test_index)
            Xc_train, Xc_test = X_close[train_index], X_close[test_index]
            yc_train, yc_test = y_close[train_index], y_close[test_index]
    #     from sklearn.externals import joblib
    #     joblib.dump(reg, 'models/tscc_1.pkl')
        bata =  data
#         bata.shape
        date = i0 
        future_x_close = X_close 
        X_close = X_close[-1:] 
        bata = bata
        date = i0 
        date = date.tail()
        #bata = bata.tail() 
        date = i0
        y_close = reg3.predict(future_x_close) 
        print('accuracy {}'.format(reg3.score(Xc_test,yc_test)))
        y_close_df = pd.DataFrame(y_close) 
        y_close_df.to_csv('close_pred.csv')
        
        '''Volume Prediction Model'''
        y_volume= first_iteration['volume']
        X_volume = first_iteration.drop(['volume'],axis=1) 
        Xv_train,Xv_test,yv_train,yv_test = train_test_split(X_volume,y_volume,test_size=.45,shuffle=False) 
        mini = MinMaxScaler() 
        X_volume = mini.fit_transform(X_volume) 
        reg4.fit(Xv_train,yv_train)
                
        pickle.dump(reg4, open('models/volume_model.pkl','wb'))
        reg4 = pickle.load(open('models/close_model.pkl','rb'))

        tscv = TimeSeriesSplit(n_splits=5)
#         print(tscv)  
#         TimeSeriesSplit(max_train_size=None, n_splits=4)
        for train_index, test_index in tscv.split(X_volume):
            print("TRAIN:", train_index, "TEST:", test_index)
            Xv_train, Xv_test = X_volume[train_index], X_volume[test_index]
            yv_train, yv_test = y_volume[train_index], y_volume[test_index]
    #     from sklearn.externals import joblib
    #     joblib.dump(reg, 'models/tscv_1.pkl')
        bata =  data
#         bata.shape
        date = i0 
        future_x_volume = X_volume 
        X_volume = X_volume[-1:] 
        bata = bata
        date = i0 
        date = date.tail()
        #bata = bata.tail() 
        date = i0
        y_volume = reg4.predict(future_x_volume) 
        
        '''Calculate Predicted Energy For Data Features: Open High Low Close Volume 
        to extract, process, and analyze data from multiple sources'''
        energy = np.array([
                sum(abs(data[i:i+f_l]**2))
                for i in range(0, len(data), h_l)
            ]) 
        

        energy_r0 = np.array([
                sum(abs(reg.predict(Xo_test[i:i+f_l])**2))
                for i in range(0, len(reg.predict(Xo_test)), h_l)
            ])  

        energy_r1 = np.array([
                sum(abs(reg1.predict(Xh_test[i:i+f_l])**2))
                for i in range(0, len(reg1.predict(Xh_test)), h_l)
            ])  
        
        energy_r2 = np.array([
                sum(abs(reg2.predict(Xl_test[i:i+f_l])**2))
                for i in range(0, len(reg2.predict(Xl_test)), h_l) 
            ])  

 
        energy_r3 = np.array([
                sum(abs(reg3.predict(Xv_test[i:i+f_l])**2))
                for i in range(0, len(reg3.predict(Xc_test)), h_l)
            ])
    
        energy_r4 = np.array([
                sum(abs(reg4.predict(Xv_test[i:i+f_l])**2))
                for i in range(0, len(reg4.predict(Xv_test)), h_l)
            ])
        

        rmse_o = librosa.feature.rms(reg.predict(Xo_test), frame_length=f_l, hop_length=h_l, center=True)
        rmse_h = librosa.feature.rms(reg1.predict(Xh_test), frame_length=f_l, hop_length=h_l, center=True)
        rmse_l = librosa.feature.rms(reg2.predict(Xl_test), frame_length=f_l, hop_length=h_l, center=True)
        rmse_c = librosa.feature.rms(reg3.predict(Xc_test), frame_length=f_l, hop_length=h_l, center=True)
        rmse_v = librosa.feature.rms(reg4.predict(Xv_test), frame_length=f_l, hop_length=h_l, center=True)
        

        frames = range(len(energy))
        t = librosa.frames_to_time(frames, sr=sam_rate, hop_length=h_l) 
        plt.scatter(frames,t)

        sig = dict( 
            number=[b0,c0,d0,e0,f0],
            stage=[ "Open", "High", "Low", "Close",'volume'])
        fig = px.funnel(data, x=t, y=frames)
        fig.show()
        
        print('predicted open')
        yo = strip(reg.predict(Xo_test), f_l, h_l) #0,500
        plt.plot(y)
        plt.show()
        
        print('predicted high')
        yh = strip(reg1.predict(Xh_test), f_l, h_l) #0,500
        plt.plot(yh)
        plt.show()
        
        print('predicted low')
        yl = strip(reg2.predict(Xl_test), f_l, h_l) #0,500
        plt.plot(yl)
        plt.show()

        print('predicted close')
        yc = strip(reg3.predict(Xc_test), f_l, h_l) #0,500
        plt.plot(yc)
        plt.show()

        print('predicted Volume')
        yv = strip(reg4.predict(Xv_test), f_l, h_l) #0,500
        plt.plot(yv)
        plt.show()

        print('Predicted Open Energy: {}'.format(energy_r0))
        print('Predicted High Energy: {}'.format(energy_r1))
        print('Predicted Low Energy: {}'.format(energy_r2))
        print('Predicted Close Energy: {}'.format(energy_r3))
        print('Predicted Volume Energy: {}'.format(energy_r4))
        
        print('Predicted Open Root mean squared error: {}'.format(rmse_o))
        print('Predicted High Root mean squared error: {}'.format(rmse_h))
        print('Predicted Low Root mean squared error: {}'.format(rmse_l))
        print('Predicted Close Root mean squared error: {}'.format(rmse_c))
        print('Predicted Volume Root mean squared error: {}'.format(rmse_v))
        
        print('predicted Open market cap: {}'.format((reg.predict(X_open[-1:])*background[-1:])))
        print('predicted High market cap: {}'.format((reg1.predict(X_high[-1:])*background[-1:])))
        print('predicted Low market cap: {}'.format((reg2.predict(X_low[-1:])*background[-1:])))
        print('predicted Close market cap: {}'.format((reg3.predict(X_close[-1:])*background[-1:])))
        
        print('predicted Open {} Price: {} \n'.format(currency,yo[-1:]))
        print('predicted High {} Price: {} \n'.format(currency,yh[-1:]))
        print('predicted Low {} Price: {} \n'.format(currency,yl[-1:]))
        print('predicted Close {} Price: {} \n'.format(currency,yc[-1:]))
        print('predicted {} volume: {} \n'.format(currency,yv[-1:]))

        print('current {} balance {} \n'.format(currency,current_balance))
        print('predicted Open {} portfolio balance {} \n'.format(currency,float(yo[-1:])*float(auth_client_currency)))
        print('predicted High {} portfolio balance {} \n'.format(currency,float(yh[-1:])*float(auth_client_currency)))
        print('predicted Low {} portfolio balance {} \n'.format(currency,float(yl[-1:])*float(auth_client_currency)))
        print('predicted Close {} portfolio balance {} \n'.format(currency,float(yc[-1:])*float(auth_client_currency)))
        
        print(yo[-1:])
        print(yo_test[-1:])
        print(y_close[-1:])
        print('30% target ',current_balance+tar)
        print('10% taget',current_balance+tar2) 
        print('10% stop loss', current_balance-loss)                       
        
        if float(yo[-1:]) and float(yo_test[-1:]) < float(y_close[-1:])*.1:
            print('Predicted open is 10% less than actual previous close: buying {}'.format(currency))
            print('predicted price', yo[-1:])
            print('actual price', yo_test[-1:])
            print('current trading balance ',current_balance-1.5)
#             buy = client.buy('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(buy)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock() 
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock() 
            import time
            time.sleep(60)

        if float(yo[-1:]) and float(yo_test[-1:]) < float(y_close[-1:])*.3:
            print('Predicted open is 30% greater than actual previous close: Selling {}'.format(currency))
            print('predicted price', yo[-1:])
            print('actual price', yo_test[-1:])
            print('current trading balance ',current_balance-1.5)
#             buy = client.buy('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(buy)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock() 
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock()
            import time
            time.sleep(60)
        
        if float(yo[-1:])==float(yo_test[-1:]):
            print('Predicted open is equal to current price Holding {}'.format(currency))
            print('predicted price', yo[-1:])
            print('actual price', yo_test[-1:])
            print('current trading balance ',current_balance)
#             buy = client.buy('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(buy)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock() 
            import time
            time.sleep(60)

        if current_balance == tar:
            print('30% profit target hit, selling')
            print('current trading balance ',current_balance)
            print('tar ',tar)
#             sell = client.sell('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(sell)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock() 
            import time
            time.sleep(60)
            
        if current_balance == tar2:
            print('15% profit target hit, selling')
            print('current trading balance ',current_balance)
            print('tar2 ',tar2)
#             sell = client.sell('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(sell)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock()
            import time
            time.sleep(60)
 

        if current_balance == tar3:
            print('3% profit target hit, selling')
            print('current trading balance ',current_balance)
            print('tar3 ',tar3)
#             sell = client.sell('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(sell)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock()
            import time
            time.sleep(60)


        if current_balance == loss:
            print('10% Stop loss hit, selling')
            print('current trading balance ',current_balance)
            print('loss ', loss)
#             sell = client.sell('ede7351f-835f-57e2-a26d-a70c49058833', amount=amount*.35, currency=currency)
#             print(sell)
#             fills1 = pd.DataFrame(client.get_buy('ede7351f-835f-57e2-a26d-a70c49058833', buy.id))   
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock()
            import time
            time.sleep(60)

        else:
            print('Parameters Not Met: Holding')
            print('current trading balance',current_balance)
#             print(client.get_buy_price(currency_pair='ETH-USD')['amount'])
            latestBlock()
            latestBlock()
            latestBlock()
            latestBlock() 
            import time
            time.sleep(60)
        
#         test_size = .8
#         data = dict(
#             number=[first_iteration['time']*test_size,[first_iteration['volume']]],
#             stage=[ "Open", "High", "Low", "Close",'Volume'])
#         fig = px.funnel(first_iteration, x=first_iteration['time'][:50], y=reg4.predict(Xv_test)[:50],title='Sounds of Crypto',labels=())
#         fig.show()
#         fig.write_html("data/signal.html")

    latestBlock()
    latestBlock()
    latestBlock()
    latestBlock() 
    import time
    time.sleep(60)
    iteration += 1 
    return 

read_data()

'''Buy Me A Coffee :)'''
# client.get_primary_account()
account = client.get_accounts(limit=100)
# # account = client.get_accounts()
# # print(account)
acct = pd.DataFrame(account)
acct.to_json('accounts.json')
with open('accounts.json', encoding='utf-8-sig') as f_input:
    df = pd.read_json(f_input)

acct = df.to_csv('accounts.csv', encoding='utf-8', index=False)
# acct = pd.read_csv('accounts.csv')
# acct
