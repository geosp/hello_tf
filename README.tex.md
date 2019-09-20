# Neural Networks

## Introduction

### Perceptrons

The basic unit of work in a neural network is the perceptron. A perceptron has an associated potential to emit a signal. For convenience the value of the potential is kept between $0$ and $1$. If the potential $p = 1$ the neuron is active, if $p = 0$ the neuron is inactive. We can implement the perceptron as a function $P$ with an array of activation values $[a_{1}, a_{2}, a_{3}, ..., a_{n}]$ i.e. $a_{1...n}$  in it's internal scope. The function parameters are an array of weight values $[w_{1}, w_{2}, w_{3}, ..., w_{n}]$ or $w_{1...n}$. The output then is the signal $p$. The values $a_{1...n}$ and $w_{1...n}$ are defined as tensors because the types of operations or functions that will be used to manipulate the perceptrons come from a branch of mathematics called [Tensor Analysis](https://en.wikipedia.org/wiki/Tensor_calculus). Consider the implementation of $P$ based on the following:

* We define tensors $\hat{A} = a_{1...n}$ and $\hat{W} = w_{1...n}$.
* Multiply tensors $\hat{A}$ and $\hat{W}$ i.e. $\hat{C} = \hat{A} \cdot \hat{W}$.
* The tensor product will be $\hat{C} = [a_{1}w_{1}, a_{2}w_{2}, a_{3}w_{3}, ..., a_{n}w_{n}]$.
* Reduce the resulting tensor $\hat{C}$ to a scalar value by adding it's components.
* The sum of the tensor $\hat{C}$ components is $S_{w} = a_{1}w_{1} + a_{2}w_{2} + a_{3}w_{3}, ..., a_{n-1}w_{n-1} + a_{n}w_{n}$.
* ${S_{w}}$ is called a weighted sum  and is represented by $\sum_{n=1}^{k} a_{n}w_{n}$ where $k$ is the number of elements in $\hat{C}$.
* The weighted sum determines the strength of the signal emitted by the perceptron.
* Capping $S_{w}$ adds additional control over signal emission and is done by substracting a bias $b$ from sum.
* It is possible for $S_{w} - b$ to have a value outside the desire signal strength $0 \geq p \leq 1$. For this reason an activation function is used to bring this value into the desired range.
* One of the  commonly used `activation functions` is the `sigmoid` $\sigma (x) =  \frac {\mathrm{1} }{\mathrm{1} + e^-r }$.
* In conclusion $P(\hat{W}) = \sigma (S_{w} - b)$.
