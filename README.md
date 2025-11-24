# Bayesian-Neural-Network
This repository includes my M.Sc Research Project. My research project's objective was to develop an accurate and reliable Bayesian Neural Network predictive model for identifying adolescents at risk of developing depression and identify significant risk factors associated with depression in adolescents.

eda.py fetchs data from url, conducts data cleaning, preparation and exploratory data analysis.

bnn.py builds the Bayesian Neural Network predictive model and uses Variational Inference (Bayes by Backprop). ReLU was used within the hidden layers while sigmoid activation for the output layer.  ReLU leads to scaling symmetry which is an unidentifiability problem, so using a Gaussian prior minimizes the scaling symmetry as it favors weights with the same Frobenius norm on each layer. The weights of the network will be treated as stochastic variables. Bayes by Backprop ensures model balances accuracy, scalability and computational efficiency while capturing critical uncertainty estimates needed for predicting susceptibility to depression among adolescents.

rf.py builds a Random Forest Classifier using the variables used in building the BNN. The Random Forest predictive model is benchmark model to which its performance will be compared to the BNN.
