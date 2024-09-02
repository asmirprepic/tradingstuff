# tradingstuff
Implementations of various trading algorithms. An abstract base class is created from which the agents are implemented. 
The main purpose is to generate trading signals. 
The agents can perform the signal generation on a dataframe consiting of multiple stocks. 
It utilizies a multilevel index on the column where level 0 is the stock ticker and the second level is Close,Open, High, Low and Volume.



### [Agents](https://github.com/asmirprepic/tradingstuff/blob/main/agents)
[Base class (abstract)](https://github.com/asmirprepic/tradingstuff/blob/main/agents/trading_agent.py)

1. [Bollinger Bands](https://github.com/asmirprepic/tradingstuff/blob/main/agents/techincal/bollinger_bands_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/BollingerBandsAgent_AZN_ST.png" width="600">
2. [Moving Average](https://github.com/asmirprepic/tradingstuff/blob/main/agents/moving_average_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/MovingAverageAgent_AZN_ST.png" width="600">
   
3. [MACD](https://github.com/asmirprepic/tradingstuff/blob/main/agents/macd_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/MACDAgent_AZN_ST.png" width="600">
   
4. [Momentum](https://github.com/asmirprepic/tradingstuff/blob/main/agents/momentum_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/MomentumAgent_AZN_ST.png" width="600">
   
5. [RSI](https://github.com/asmirprepic/tradingstuff/blob/main/agents/rsi_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/RSIAgent_AZN_ST.png" width="600">
  
6. [On balance volume](https://github.com/asmirprepic/tradingstuff/blob/main/agents/on_balance_volume_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/OBVAgent_AZN_ST.png" width="600">
  
7. [Price volume trend](https://github.com/asmirprepic/tradingstuff/blob/main/agents/price_volume_trend_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/PVTAgent_AZN_ST.png" width="600">
  
8. [Volume weighted average price](https://github.com/asmirprepic/tradingstuff/blob/main/agents/volume_weighted_average_price.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/VWAPAgent_AZN_ST.png" width="600">
  
8. [KNN](https://github.com/asmirprepic/tradingstuff/blob/main/agents/knn_agent.py)<br><br>
   <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/KNNAgent_AZN_ST.png" width="600">
  
10. [Logistic regression](https://github.com/asmirprepic/tradingstuff/blob/main/agents/logistic_reg_agent.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/LRAgent_AZN_ST.png" width="600">
  
11. [Neural network](https://github.com/asmirprepic/tradingstuff/blob/main/agents/nn_classification_agent.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/NNClassification_AZN_ST.png" width="600">
  
12. [SVM](https://github.com/asmirprepic/tradingstuff/blob/main/agents/svm_agent.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/SVMAgent_AZN_ST.png" width="600">
  
13. [Q-learning](https://github.com/asmirprepic/tradingstuff/blob/main/reinforcement_learning/reinforcement_agent.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/AZN_ST_plot.png" width="600">

14. [HighLow](https://github.com/asmirprepic/tradingstuff/blob/main/agents/high_low.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/HighLowAgent.png" width="600">

15. [Naive Bayes](https://github.com/asmirprepic/tradingstuff/blob/main/agents/naive_bayes_agent.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/naive_bayes_AZNST.png" width="600">

16. [Bayesian NN](https://github.com/asmirprepic/tradingstuff/blob/main/agents/bayesian_nn.py)<br><br>
    <img src="https://github.com/asmirprepic/tradingstuff/blob/main/plots/bayesian_nn_AZNST.png" width="600">
    
