# tradingstuff
Implementations of various trading algorithms. An abstract base class is created from which the agents are implemented. 
The main purpose is to generate trading signals. 
The agents can perform the signal generation on a dataframe consiting of multiple stocks. 
It utilizies a multilevel index on the column where level 0 is the stock ticker and the second level is Close,Open, High, Low and Volume.



### [Agents](https://github.com/asmirprepic/tradingstuff/blob/main/agents.py)
[Base class (abstract)](https://github.com/asmirprepic/tradingstuff/blob/main/agents/trading_agent.py)

1. [Bollinger Bands](https://github.com/asmirprepic/tradingstuff/blob/main/agents/bollinger_bands_agent.py)
2. [Moving Average](https://github.com/asmirprepic/tradingstuff/blob/main/agents/moving_average_agent.py)
3. [MACD](https://github.com/asmirprepic/tradingstuff/blob/main/agents/macd_agent.py)
4. [Momentum](https://github.com/asmirprepic/tradingstuff/blob/main/agents/momentum_agent.py)
5. [RSI](https://github.com/asmirprepic/tradingstuff/blob/main/agents/rsi_agent.py)
6. [On balance volume](https://github.com/asmirprepic/tradingstuff/blob/main/agents/on_balance_volume_agent.py)
7. [Price volume trend](https://github.com/asmirprepic/tradingstuff/blob/main/agents/price_volume_trend_agent.py)
8. [Volume weighted average price](https://github.com/asmirprepic/tradingstuff/blob/main/agents/volume_weighted_average_price.py)
9. [KNN](https://github.com/asmirprepic/tradingstuff/blob/main/agents/knn_agent.py)
10. [Logistic regression](https://github.com/asmirprepic/tradingstuff/blob/main/agents/logistic_reg_agent.py)
11. [Neural network](https://github.com/asmirprepic/tradingstuff/blob/main/agents/nn_classification_agent.py)
12. [SVM](https://github.com/asmirprepic/tradingstuff/blob/main/agents/svm_agent.py)
