#%%

import pandas as pd
from VAR_Factors import merged_1_2_3_4

#%%

df = merged_1_2_3_4[['Date', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']]

factors = (df['Factor 1'], df['Factor 2'], df['Factor 3'], df['Factor 4'])


#%% Codes to calculate returns from prices (yet to be adjusted to get data for first 5 days)

prices = pd.to_numeric(df['Factor 1'])

pricesdf = pd.DataFrame(data=prices)

pric = pd.DataFrame(pricesdf['Factor 1'])

return_column = []

def calculate_returns():

    i=5
        
    for row in pric.iterrows():
    
        if i <= (len(pric)-1): 
            days5ret = (pric['Factor 1'].iloc[i] - pric['Factor 1'].iloc[i-5]) / pric['Factor 1'].iloc[i-5]
            days5ret = days5ret.tolist()
            return_column.append(days5ret)
            i+=1
    
    returns_to_append = pd.DataFrame(data=return_column)    
    pric['Returns'] = returns_to_append


#pric.head()
#pric.tail()

#%% Show Start and End date of every instrument

import os.path
import pandas as pd

path = '/Users/javierfdz/Documents/GitHub/VaR-Calculation-SPARK/Instruments_data'  
filelist = os.listdir(path)

csv_inst_names = []

for i in filelist:
    csv_inst_names.append(i[:-4])

for i in csv_inst_names:
    
    instru = pd.read_csv('/Users/javierfdz/Documents/GitHub/VaR-Calculation-SPARK/Instruments_data/{}.csv'.format(i))
    
    print(i[6:])
    print('Start date: ', min(instru['timestamp']))
    print('End date: ', max(instru['timestamp']), '\n')

    if min(instru['timestamp']) > '2008-01-01' and max(instru['timestamp']) < '2019-01-31':
        print( i[6:], ' no data for full period 1-Jan-08 - 31-Jan-19 \n')
    

