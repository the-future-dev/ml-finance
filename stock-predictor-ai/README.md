# Using advancements in AI to predict stock market movements.

The objective of this repository is to create complete process for predicting stock price movements.

We will use a Generative Adversial Network (**GAN**) with
- **LSTM** (a type of Recurrent Neural Network) as generator
- and a Convolutional Neural Network (**CNN**) as discriminator.

Why we use GAN and specifically CNN as discriminator? Good question: there is a specific section on that later.

We will get into greater detail for each step, but the most difficult part is the GAN: very tricy part of training a GA is gettting the right set of hyperparameters. For that reason we will use a Bayesian optimisation (along with Gaussian processes) and Reinforcement Learning (**RL**) for deciding when and how to change the GAN's hyperparameters (the exploration vs exploitation dilemma). In creating the reinforcement learning we will use the most recent advancements in the field such as Rainbow and PPO.


We will use a lot of different types of input data. Along with the stock's historical trading data and technical indicators, we will use the newest advancements in NLP (using 'Bidirectional Embedding Representations from Transformers', BERT, sort of a transfer learning for NLP) to create sentiment analysis (as a source for fundamental analysis), Fourier transforms for extracting overall trend directions, Stacked autoencoders for identifying other high-level features, Eigen portfolios for finding correlated assets, autoregressive integrated moving average (ARIMA) for the stock function approximation, and many more, in order to capture as much information, patterns, dependencies, etc, as possible about the stock. As we all know, the more (data) the merrier. Predicting stock price movements is an extremely complex task, so the more we know about the stock (from different perspectives) the higher our changes are.


## 1. Introduction
Accurately predicting the stock markets is a complex task as there are millions of events and pre-conditions for a particilar stock to move in a particular direction. So we need to be able to capture as many of these pre-conditions as possible. We also need make several important assumptions: 1) markets are not 100% random, 2) history repeats, 3) markets follow people's rational behavior, and 4) the markets are 'perfect'. 

We will try to predict the price movements of Goldman Sachs (NYSE: GS). For the purpose, we will use daily closing price from January 1st, 2010 to December 31st, 2018 (seven years for training purposes and two years for validation purposes). We will use the terms 'Goldman Sachs' and 'GS' interchangeably.

## Acknowledgement
The structure idea implemented in the following repository is inspired by https://github.com/borisbanushev/stockpredictionai/.

## 3. The Data
What affects whether GS's stock price will move up or down? We will start by incorporating as much information as possible and we will perform feature importance (meaning how indicative it is for the movement of GS) on every feature later on and decide whether we will use it.

We will use daily data: 1585 days to train the various algorithms and redict the next 680 days (test data).


Overview of each data type (*feature*):

1. **Correlated assets** - these are other assets (any type, not necessarily stocks, such as commodities, FX, indices, or even fixed income securities). A big company, such as Goldman Sachs, doesn't live in an isolated world - it depends on, and interacts with, many external factors, including its competitors, clients, the global economy, the geo-political situation, fiscal and monetary policies, access to capital, etc.

2. **Technical indicators** - a lot of investors follow technical indicators. We will include the most popular indicators as independent features. Among them - 7 and 21 days moving average, momentum, Bollinger bands, MACD.

3. **Fundamental analysis** - A verty important feature indicating whether a stock might move up or down. There are two features that can be used in fundamental analysis:
    1. Analysing the company performance using 10-K and 10-Q reports, analysing ROE and P/E, etc.
    2. News - potentially news can idicate upcoming evnts that can potentially move the stock in certain direction. We will read all daily news for Goldman Sachs and extract whether the total sentiment about Goldman Sachs on that day is positive, neutral, or negative (as a score from 0 to 1). As many investors closely read the news and make investment decisions based (partially of course) on news, there is a somewhat high chance that if, say, the news for Goldman Sachs today are extremely positive the stock will surge tomorrow.

For the purpose of creating accurate sentiment prediction we will use Neural Language Processing (NLP). We will use BERT - Google's recently announced NLP approach for transfer learning for sentiment classification stock news sentiment extraction.

4. **Fourier transforms** - Along with the daily closing price, we will create Fourier transforms in order to generalize several long- and short- term trends. Using these transforms we will eliminate a lot of noise (random walks) and create approximations of the real stock movement. Having trend approximations can help the LSTM network pick its prediction trends more accurately.

5. **Autoregressive Integrated Moving Average** (ARIMA) - This was one of the most popular thecniques for predicting future values of time series data in pre-neutal network ages.

6. **Stacked Autoencoders** - most of the before mentioned features were found by people after decades of research. But maybe we have missed something. Maybe there are hidden correlations that people cannot comprehend due to the normous amount of data pounts, events, assets, charts, etc. With stacked autoencoders (type of neural networks) we can use the power of computers and probably find new types of features that affect stock movements. Even though we will not be able to understand these features in human language, we will use them in the GAN.

7. **Deep Unsupervised learning for anomaly detection in options pricing** - We will use one more feature - for every day we will add the price for 90-days call option on Goldman Sachs stock. Options pricing itself combines a lot of data. The price for options contract depends on the future value of the stock (analysts try to also predict the price in order to come up with the most accurate price for the call option). Using deep unsupervised learning (Self-organized Maps) we will try to spot anomalies in every day's pricing. Anomaly (such as a drastic change in pricing) might indicate an event that might be useful for the LSTM to learn the overall stock pattern.


Having so many features we need to perform a couple of important steps:

1. Perform statistical checks for the 'quality' of the data. If the data we create is flawed, then no matter how sophisticated our algorithms are, the results will not be positive. The checks include making sure the data does not suffer from heteroskedasticity, multicollinearity, or serial correlation.

2. Create feature importance. If a feature (e.g. another stock or a technical indicator) has no explanatory power to the stock we want to predict, then there is no need for us to use it in the training of the neural nets. We will using XGBoost (eXtreme Gradient Boosting), a type of boosted tree regression algorithms.

3. As a final step of our data preparation, we will also create **Eigen portfolios** using **Principal Component Analysis** (PCA) in order to reduce the dimensionality of the features created from the autoencoders.


## 3.1. Correlated assets
 As explained earlier we will use other assets as features, not only GS. So what other assets would affect GS's stock movements? Good understanding of the company, its lines of businesses, competitive landscape, dependencies, suppliers and client type, etc is very important for picking the right set of correlated assets:

1. Peer Companies:
    PMorgan Chase & Co. (JPM)
    Morgan Stanley (MS)
    Bank of America Corp (BAC)
    Citigroup Inc. (C)
    Wells Fargo & Co (WFC)
    Barclays PLC (BARC)
    Deutsche Bank AG (DB)
    Credit Suisse Group AG (CS)
    UBS Group AG (UBS)
    HSBC Holdings plc (HSBC)

TODO:
2. Global economy indicators:
    USD LIBOR
    GBP LIBOR
    VIX (Volatility Index)
    US Consumer Price Index (CPI)
    US Unemployment Rate
    US Gross Domestic Product (GDP)
    Euro Area GDP
    China GDP
    Japan GDP
    UK GDP

TODO
3. Daily volatility index (VIX) - for the reason described in the previous point.

TODO
4. Major Stock Indices:
    NASDAQ Composite Index
    NYSE Composite Index
    S&P 500 Index
    Dow Jones Industrial Average
    FTSE 100 Index
    DAX Index (Germany)
    CAC 40 Index (France)
    Nikkei 225 Index
    Hang Seng Index
    Shanghai Composite Index

5. Currencies:
    USD/EUR
    USD/JPY
    GBP/USD
    USD/CHF
    USD/CAD
    AUD/USD
    NZD/USD
    USD/CNY
    EUR/JPY
    EUR/GBP

6. Commodities:
    Gold
    Silver
    Crude Oil (WTI)
    Brent Crude
    Natural Gas
    Copper
    Aluminum
    Iron Ore
    Platinum
    Palladium

7. Fixed Income:
    US 10-Year Treasury Yield
    US 2-Year Treasury Yield
    German 10-Year Bund Yield
    UK 10-Year Gilt Yield
    Japan 10-Year Bond Yield
    Corporate Bond Yields
    Municipal Bond Yields
    High-Yield Bond Index
    Emerging Market Bond Index
    Mortgage Backed Securities Index

TODO
8. Real Estate and Insurance-Related Indices
    Commercial Real Estate Prices Index
    Residential Real Estate Prices Index
    Catastrophe Bond Index
    Insurance-Linked Securities Index

TODO
9. Global Policy Rates:
    US Federal Funds Rate
    European Central Bank Main Refinancing Rate
    Bank of England Base Rate
    Bank of Japan Policy Rate
    People's Bank of China Base Rate
    Reserve Bank of Australia Cash Rate
    Bank of Canada Overnight Rate
    Reserve Bank of India Repo Rate
    Central Bank of Brazil Selic Rate
    Bank of Russia Key Rate

TODO
10. Credit Indices:
    iTraxx Europe Index (European corporate credit risk)
    CDX North America Index (North American corporate credit risk)
    Markit iBoxx USD Liquid Investment Grade Index
    Markit iBoxx USD Liquid High Yield Index
    FRED US Bank Prime Loan Rate
    US Corporate BBB Effective Yield
    US Corporate Aaa Effective Yield
    US High Yield Master II Effective Yield

TODO
11. Global Economic Indicators:
    Global Manufacturing Purchasing Managers Index (PMI)
    US Retail Sales
    Eurozone Consumer Confidence
    China Industrial Production
    Japan Retail Sales
    UK Retail Sales
    Global Trade Volume
    US Business Inventories

TODO
12. Alternative Investments:
    Bitcoin
    Ethereum
    Other major cryptocurrencies
    Venture Capital Index
    Private Equity Index
    Hedge Fund Index
