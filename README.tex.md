# Neural Networks

## Introduction

### Perceptrons

The basic unit of work in a neural network is the perceptron. A perceptron has an associated potential to emit a signal. For convenience the value of the potential is kept between $0$ and $1$. If the potential $p = 1$ the neuron is active, if $p < 1$ the neuron is inactive. We can implement the perceptron as a function $P$ with an array of activation values $[a_{1}, a_{2}, a_{3}, ..., a_{n}]$ i.e. $a_{1...n}$  in it's internal scope. The function parameters are an array of weight values $[w_{1}, w_{2}, w_{3}, ..., w_{n}]$ or $w_{1...n}$. The output then is the signal $p$. The values $a_{1...n}$ and $w_{1...n}$ are defined as tensors because the types of operations or functions that will be used to manipulate the perceptrons come from a branch of mathematics called [Tensor Analysis](https://en.wikipedia.org/wiki/Tensor_calculus). The internal implementation of $P$ will then consist of the following steps:

* We define $\hat{A} = a_{1...n}$ and $\hat{W} = w_{1...n}$
* Take the tensor product of $\hat{A}$ and $\hat{W}$ i.e. $\hat{C} = \hat{A} \cdot \hat{W}$
* Reduce the resulting vector $\hat{C}$ to a scalar by adding it's components.
* The tensor product $\hat{C} = [a_{1}w_{1}, a_{2}w_{2}, a_{3}w_{3}, ..., a_{n}w_{n}]$
* The sum of $\hat{C}$ components is $a_{1}w_{1} + a_{2}w_{2} + a_{3}w_{3}, ..., a_{n-1}w_{n-1} + a_{n}w_{n}$
* This sum is called a weighted sum  and is represented by $\sum_{n=1}^{k} a_{n}w_{n}$ where $k$ is the number of elements in $\hat{C}$.
* The weighted sum will control when the perceptron will emit a signal.
* Capping the weighted sum adds additional control to signal emission and this is done by introducing a bias $b$ term to the sum. 
* $(\sum_{n=1}^{k} a_{n}w_{n}) + b$ could have a value outside the desire signal i.e. $0 \geq p \leq 1$. For this reason an activation function is used to bring this value into the desired range.
* One of the  commonly used activation functions is the sigmoid $h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^- \theta^Tx }$.
