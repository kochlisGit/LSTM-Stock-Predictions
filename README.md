# Stocks-Prediction
Prediction of Stock price using Recurrent Neural Network (RNN) models. In this project, I compare how different well-optimized RNN models perform at stocks prediction. 

# Models
I have used some of the most popular RNN models that are used in the today's industry:
1. LSTM
2. GRU
3. Bidirectional LSTM
4. Combination of Bidirection GRU with LSTM

# Dataset
The source of my datasets is Yahoo's finance website: https://finance.yahoo.com/

The datasets include Google's, Tesla & Greek's Alpha-Bank stocks. Specifically, each dataset contains training data about the stocks from 01/01/2017 to 01/01/2019. Then, a small dataset from
01/01/2019 to 01/01/2020 is used to make the predictions. The dataset contains the following data for each stock:

Data | Open | High | Low | Close | Adj Close | Volume

* Open: The inital price of the stock at the beginning of the day.
* High: The highest price of the stock at that particular day.
* Low: The lowest price of the stock at that particular day.
* Close: The final price of the stock at that particular.


# Libraries
The RNN were implemented using Python. The libraries that were used are the following:
1. Numpy
2. Pandas
3. Matplotlib
4. Keras
5. Tensorflow
6. Tensorflow Addons

# !..Important..!
Don't trust prediction models for stock prediction. The predictions that are made are based on patterns that are found in the dataset. However, It's impossible to know the exact "Open" value of a stock for the next day. What You should be interested in is the behavior of the price (e.g. If the price is rising up or falling down).

Google's Prediction
![](https://github.com/kochlisGit/Stocks-Prediction/blob/main/google/plots/google_bgru_lstm_plot.png)

Tesla's Prediciton
![](https://github.com/kochlisGit/Stocks-Prediction/blob/main/tesla/plots/tesla_bgru_lstm_plot.png)

Alpha Bank's Prediction
![](https://github.com/kochlisGit/Stocks-Prediction/blob/main/alpha-bank/alpha_predict_plot.png)


# Additional Analysis
It's important to feed the RNNs only with data that are related to the prediction of the targets. Unecessary data might confuse the network, slow down the training process
and even reduce the prediction performance.

Aside from Open, Close, High, Low, there are other factors that are related to the price of a stock. For example, we could also feed the RNN the pointer's data.
Here is a plot of the importance of the pointers data:

![](https://github.com/kochlisGit/Stocks-Prediction/blob/main/alpha-bank/open_pointers_correlation_plot.png)

You can also see how related are 2 variables by calculating Pearson's correlation. More about correlations can be found here: https://github.com/kochlisGit/ML-Statistics/tree/main/correlation
