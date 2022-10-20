MC-Dropout can approximate a BNN using one multiple stochastic method to achieve uncertainty quantification for machine learning applications. This method averages the prediction results of several different networks to obtain the final result of the model, which is the same idea as that of ensemble learning methods.

Also, the variance, the volatility of the prediction results, is used as an indicator to evaluate the prediction uncertainty of the model.

The mian program is "TEST", which contains different network structures. The pre - and post-processing procedures is "process-last". The remaining four are datasets, which are inputs and outputs of training dataset and testing dataset, respectively.