# Experiments on Neural Networks

This repository documents my experiments involving neural networks.

## Neural Network for Regression from Scratch

I constructed a neural network for (linear) regression to noisy, synthetic data using only Numpy because I wanted to gain a deeper understanding of the mathematics behind neural networks. This involved implementing a simple neural network architecture and algorithms for forward propagation, backpropagation, and training (using the gradient descent algorithm).

Jupyter notebook: [neural_network_fundamentals.ipynb](neural_network_fundamentals.ipynb)

### Simulating Pendulum Motion using a Physics Informed Neural Network

I used a Physics Informed Neural Network (PINN) to numerically simulate solutions to the nonlinear pendulum ODE. Physics was incorporated into the training process by rewarding solutions whose behaviour satisfied the ODE at a set of randomly generated collocation points. The solutions computed using the PINN were benchmarked against numerical solutions computed using the RK4 algorithm. This technique can be generalised to numerically solve ODEs and PDEs.

Jupyter notebook: [pendulum/pendulum.ipynb](pendulum/pendulum.ipynb)


