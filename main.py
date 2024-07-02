from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import datetime



finviz_url='https://finviz.com/quote.ashx?t='

tickers=['AAPL','META','GOOG','MSFT']

news_tables={}
for ticker in tickers:
    url=finviz_url+ticker

    req=Request(url=url,headers={'user-agent':'my-app'})
    response=urlopen(req)

    html =BeautifulSoup(response,features='html.parser')
    news_table=html.find(id='news-table')
    news_tables[ticker]=news_table

current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%b-%d-%y")


def convert_to_24hr(time_str):
    if time_str[-2:] == 'AM' and time_str[:2] == '12':
        return '00' + time_str[2:-2]
    elif time_str[-2:] == 'AM':
        return time_str[:-2]
    elif time_str[-2:] == 'PM' and time_str[:2] == '12':
        return time_str[:-2]
    else:
        return str(int(time_str[:2]) + 12) + time_str[2:-2]


parsed_data=[]
for ticker,news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        
        title=row.a.text
        date_data=row.td.text.replace('/r','').replace('/n','').replace("Today", formatted_date)
        date_data=date_data.split()

        if len(date_data)==1:
            time=convert_to_24hr(date_data[0])
        else:
            date=date_data[0]
            time=convert_to_24hr(date_data[1])

        parsed_data.append([ticker, date, time, title])


df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])



# print(df.head())



vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
df['date'] = pd.to_datetime(df.date).dt.date

# print(df.head(10))
# print(df.count())

plt.figure(figsize=(10, 8))
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()
mean_df.plot(kind='bar')
plt.title('Mean Sentiment Compound Score by Date')
plt.xlabel('Date')
plt.ylabel('Mean Compound Score')
plt.xticks(rotation=45)
plt.legend(title='Ticker')
plt.show()

