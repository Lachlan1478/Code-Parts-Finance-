import requests
import pandas as pd
from datetime import datetime
import arrow
import matplotlib.pyplot as plt

Int = '60m'
Rnge = '7d'
Ticker = 'APT.AX'

#This function retrives and stores stock data in data frame
def GatherData(Symbol, Range, Interval):
    Request = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{Symbol}?range={Range}&interval={Interval}'.format(**locals()))
    data = Request.json()
    body = data['chart']['result'][0]
    Time = pd.Series(body['timestamp'])
    Date = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime(EST)')
    Frame = pd.DataFrame(body['indicators']['quote'][0], index=Date)

    #Only retrieve these values of data
    return Frame.loc[:, ('open', 'high', 'low', 'close', 'volume')]

Stock = GatherData(Ticker, Rnge, Int);

'close of the day before'
Past = Stock.iloc[5]['close']

length = len(Stock.index)-1

Present = Stock.iloc[length]['close']

'Calculate Percentage Change'
if (Present > Past):
    PercentageChange = Present/Past
else:
    PercentageChange = -Present/Past

print(PercentageChange)
print(Stock.head())

# Plot the _Close columns for stock
Stock['close'].plot(label='APT', legend=True)


plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for LNG
Stock['close'].pct_change().plot.hist(bins=50)
plt.xlabel('close 1-day percent change')
plt.show()
