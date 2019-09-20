# Neural Networks

## Introduction

### Perceptrons

The basic unit of work in a neural network is the perceptron. A perceptron has an associated potential to emit a signal. For convenience the value of the potential is kept between <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> and <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>. If the potential <img src="/tex/012b36279aac832bdad672ff18d4243a.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/> the neuron is active, if <img src="/tex/c9bb81328f293e18596280dfb95dd631.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/> the neuron is inactive. We can implement the perceptron as a function <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> with an array of activation values <img src="/tex/8a3231c98df6eafb822b8b2c14dc3f97.svg?invert_in_darkmode&sanitize=true" align=middle width=117.88258679999998pt height=24.65753399999998pt/> i.e. <img src="/tex/46ba15cae98b59904b6ca6b625b68e28.svg?invert_in_darkmode&sanitize=true" align=middle width=35.08012364999999pt height=14.15524440000002pt/>  in it's internal scope. The function parameters are an array of weight values <img src="/tex/31291bd296fcc0ad119638b272e094e7.svg?invert_in_darkmode&sanitize=true" align=middle width=130.20000345pt height=24.65753399999998pt/> or <img src="/tex/9f0bfc20948e9822b9fad9c7c0101985.svg?invert_in_darkmode&sanitize=true" align=middle width=38.159477399999986pt height=14.15524440000002pt/>. The output then is the signal <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/>. The values <img src="/tex/46ba15cae98b59904b6ca6b625b68e28.svg?invert_in_darkmode&sanitize=true" align=middle width=35.08012364999999pt height=14.15524440000002pt/> and <img src="/tex/9f0bfc20948e9822b9fad9c7c0101985.svg?invert_in_darkmode&sanitize=true" align=middle width=38.159477399999986pt height=14.15524440000002pt/> are defined as tensors because the types of operations or functions that will be used to manipulate the perceptrons come from a branch of mathematics called [Tensor Analysis](https://en.wikipedia.org/wiki/Tensor_calculus). Consider the implementation of <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> based on the following:

* We define tensors <img src="/tex/8bdd0d211d83d7197dcb2bf67dae4386.svg?invert_in_darkmode&sanitize=true" align=middle width=69.32654684999999pt height=31.141535699999984pt/> and <img src="/tex/a609488241ae292f72e220eb1be3c4c2.svg?invert_in_darkmode&sanitize=true" align=middle width=77.88534435pt height=31.141535699999984pt/>.
* Multiply tensors <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> and <img src="/tex/b92ac9c04c031ed7cddd215260ac9b30.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=31.141535699999984pt/> i.e. <img src="/tex/c04c69c1055e74987278fa4398254dd6.svg?invert_in_darkmode&sanitize=true" align=middle width=76.85130254999999pt height=31.141535699999984pt/>.
* The tensor product will be <img src="/tex/568fe2dced1a73a3f74d872bf21d3be0.svg?invert_in_darkmode&sanitize=true" align=middle width=230.87020439999998pt height=31.141535699999984pt/>.
* Reduce the resulting tensor <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/> to a scalar value by adding it's components.
* The sum of the tensor <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/> components is <img src="/tex/5187d90814d670a3ed3dee4f4b418807.svg?invert_in_darkmode&sanitize=true" align=middle width=346.3807660499999pt height=22.465723500000017pt/>.
* <img src="/tex/551ec4bb1977e368d324ef42c4ef766e.svg?invert_in_darkmode&sanitize=true" align=middle width=19.899247499999987pt height=22.465723500000017pt/> is called a weighted sum  and is represented by <img src="/tex/eb4a66cd60fcb7fd335345c273da12ab.svg?invert_in_darkmode&sanitize=true" align=middle width=83.21478164999999pt height=32.51169900000002pt/> where <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is the number of elements in <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/>.
* The weighted sum determines the strength of the signal emitted by the perceptron.
* Capping <img src="/tex/72a8c3544b60e70e9af83b6202f1d1f7.svg?invert_in_darkmode&sanitize=true" align=middle width=19.899247499999987pt height=22.465723500000017pt/> adds additional control over signal emission and is done by substracting a bias <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> from sum.
* It is possible for <img src="/tex/a15ed02692768e127f845e4d76374049.svg?invert_in_darkmode&sanitize=true" align=middle width=47.86712369999999pt height=22.831056599999986pt/> to have a value outside the desire signal strength <img src="/tex/a05f09f794b3ed2f38eb678dc04b450a.svg?invert_in_darkmode&sanitize=true" align=middle width=68.54424719999999pt height=21.18721440000001pt/>. For this reason an activation function is used to bring this value into the desired range.
* One of the  commonly used `activation functions` is the `sigmoid` <img src="/tex/c2d299c45e9ff1983a754f75c9a14704.svg?invert_in_darkmode&sanitize=true" align=middle width=95.11795424999998pt height=27.77565449999998pt/>.
* In conclusion <img src="/tex/6d136b281bb2acf64aa7b8f22fe0ba14.svg?invert_in_darkmode&sanitize=true" align=middle width=135.98351085pt height=31.141535699999984pt/>.
