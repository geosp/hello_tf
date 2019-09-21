# Machine Learning with Neural Networks

## Perceptrons

The basic unit of work in a neural network is the `perceptron`. A `perceptron` has an associated potential to emit a signal. For convenience the value of the potential is kept between <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> and <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>. If the potential <img src="/tex/012b36279aac832bdad672ff18d4243a.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/> the neuron is active, if <img src="/tex/c9bb81328f293e18596280dfb95dd631.svg?invert_in_darkmode&sanitize=true" align=middle width=38.40740639999999pt height=21.18721440000001pt/> the neuron is inactive. We can implement the `perceptron` as a function <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> with an array of `activation values` <img src="/tex/8a3231c98df6eafb822b8b2c14dc3f97.svg?invert_in_darkmode&sanitize=true" align=middle width=117.88258679999998pt height=24.65753399999998pt/> i.e. <img src="/tex/46ba15cae98b59904b6ca6b625b68e28.svg?invert_in_darkmode&sanitize=true" align=middle width=35.08012364999999pt height=14.15524440000002pt/>  in it's internal scope. The function parameters are an array of `weight values` <img src="/tex/31291bd296fcc0ad119638b272e094e7.svg?invert_in_darkmode&sanitize=true" align=middle width=130.20000345pt height=24.65753399999998pt/> or <img src="/tex/9f0bfc20948e9822b9fad9c7c0101985.svg?invert_in_darkmode&sanitize=true" align=middle width=38.159477399999986pt height=14.15524440000002pt/>. The output then is the signal <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/>. The values <img src="/tex/46ba15cae98b59904b6ca6b625b68e28.svg?invert_in_darkmode&sanitize=true" align=middle width=35.08012364999999pt height=14.15524440000002pt/> and <img src="/tex/9f0bfc20948e9822b9fad9c7c0101985.svg?invert_in_darkmode&sanitize=true" align=middle width=38.159477399999986pt height=14.15524440000002pt/> are defined as tensors because the types of operations or functions that will be used to manipulate the `perceptrons` comes from a branch of mathematics called [Tensor Analysis](https://en.wikipedia.org/wiki/Tensor_calculus). Consider the implementation of <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> based on the following:

* We define tensors <img src="/tex/8bdd0d211d83d7197dcb2bf67dae4386.svg?invert_in_darkmode&sanitize=true" align=middle width=69.32654684999999pt height=31.141535699999984pt/> and <img src="/tex/a609488241ae292f72e220eb1be3c4c2.svg?invert_in_darkmode&sanitize=true" align=middle width=77.88534435pt height=31.141535699999984pt/>.
* Multiply tensors <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> and <img src="/tex/b92ac9c04c031ed7cddd215260ac9b30.svg?invert_in_darkmode&sanitize=true" align=middle width=17.80826024999999pt height=31.141535699999984pt/> i.e. <img src="/tex/c04c69c1055e74987278fa4398254dd6.svg?invert_in_darkmode&sanitize=true" align=middle width=76.85130254999999pt height=31.141535699999984pt/>.
* The tensor product will be <img src="/tex/568fe2dced1a73a3f74d872bf21d3be0.svg?invert_in_darkmode&sanitize=true" align=middle width=230.87020439999998pt height=31.141535699999984pt/>.
* Reduce <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/> to a scalar value by adding it's components.
* The sum of the <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/> components is <img src="/tex/5187d90814d670a3ed3dee4f4b418807.svg?invert_in_darkmode&sanitize=true" align=middle width=346.3807660499999pt height=22.465723500000017pt/>.
* <img src="/tex/551ec4bb1977e368d324ef42c4ef766e.svg?invert_in_darkmode&sanitize=true" align=middle width=19.899247499999987pt height=22.465723500000017pt/> is called a weighted sum  and is represented by <img src="/tex/eb4a66cd60fcb7fd335345c273da12ab.svg?invert_in_darkmode&sanitize=true" align=middle width=83.21478164999999pt height=32.51169900000002pt/> where <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> is the number of elements in <img src="/tex/108c8d1c66f6974b9e54c8e9674ca238.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=31.141535699999984pt/>.
* <img src="/tex/72a8c3544b60e70e9af83b6202f1d1f7.svg?invert_in_darkmode&sanitize=true" align=middle width=19.899247499999987pt height=22.465723500000017pt/> determines the strength of the signal emitted by the `perceptron`.
* Capping <img src="/tex/72a8c3544b60e70e9af83b6202f1d1f7.svg?invert_in_darkmode&sanitize=true" align=middle width=19.899247499999987pt height=22.465723500000017pt/> adds additional control over signal emission and is done by subtracting a bias <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> from the sum.
* It is possible for <img src="/tex/a15ed02692768e127f845e4d76374049.svg?invert_in_darkmode&sanitize=true" align=middle width=47.86712369999999pt height=22.831056599999986pt/> to have a value outside the desire signal strength <img src="/tex/a05f09f794b3ed2f38eb678dc04b450a.svg?invert_in_darkmode&sanitize=true" align=middle width=68.54424719999999pt height=21.18721440000001pt/>. For this reason an `activation function` is used to bring <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/> into the desired range.
* One of the  commonly used `activation functions` is the `sigmoid` <img src="/tex/23c2e593354a49d6159a7b521a41f2ff.svg?invert_in_darkmode&sanitize=true" align=middle width=95.16736844999998pt height=27.77565449999998pt/>.

In conclusion the implementation of a perceptron is the function <img src="/tex/6d136b281bb2acf64aa7b8f22fe0ba14.svg?invert_in_darkmode&sanitize=true" align=middle width=135.98351085pt height=31.141535699999984pt/>.

## Neural Networks

A `neural network` is a graph of associated `perceptrons`. `Neural networks` are composed of `neural network layers`. A `neural network layer` is a tensor of `perceptrons`. The `perceptrons` in a `neural network layer` are connected to each other because they are components of a tensor. We can define layer n as <img src="/tex/21ec0f7c7be033a755c8d561b9ab2842.svg?invert_in_darkmode&sanitize=true" align=middle width=167.3933712pt height=31.141535699999984pt/>. Neural networks have three `layer types input, hidden, and output`. A neural network  may have multiple hidden layers but only one input and output layers. Consider a neural network consisting of the fallowing layers:

<div align="center">
<div>
<img with=160 height=160 src="img/simple_nn.svg"/>
</div>
<div>

<img src="/tex/9b10f91822afbf5be4cefeacc35b17ae.svg?invert_in_darkmode&sanitize=true" align=middle width=70.28910404999999pt height=31.141535699999984pt/> &nbsp;&nbsp; <img src="/tex/5ef91ad474a3f576e1b87884e30b3cb0.svg?invert_in_darkmode&sanitize=true" align=middle width=115.86194895pt height=31.141535699999984pt/> &nbsp;&nbsp; <img src="/tex/fb44c96eb90a025b1d144ea7f7fc42ae.svg?invert_in_darkmode&sanitize=true" align=middle width=73.96450709999998pt height=31.141535699999984pt/>

</div>
</div>

Neural network themselves are tensors. In this case neural network <img src="/tex/7db884798ecc73c9bba043881409781b.svg?invert_in_darkmode&sanitize=true" align=middle width=122.07730094999997pt height=31.141535699999984pt/>. `Perceptrons` in a neural network are associated to each other via `function composition`. Consider <img src="/tex/6abb365fad186c6e2bbfa9783072fe89.svg?invert_in_darkmode&sanitize=true" align=middle width=21.75709304999999pt height=22.465723500000017pt/> it has an internal tensor of `activation values` <img src="/tex/d3c8ec4c3897e5a12784aab039c196c2.svg?invert_in_darkmode&sanitize=true" align=middle width=76.11870255pt height=31.141535699999984pt/>. The number of components in <img src="/tex/a0638f9c12d15dc1c92d324151064311.svg?invert_in_darkmode&sanitize=true" align=middle width=23.53224389999999pt height=31.141535699999984pt/> is one. The output of <img src="/tex/6abb365fad186c6e2bbfa9783072fe89.svg?invert_in_darkmode&sanitize=true" align=middle width=21.75709304999999pt height=22.465723500000017pt/> is a potential <img src="/tex/e18e054e2d963f4a4d6225355036b639.svg?invert_in_darkmode&sanitize=true" align=middle width=19.474012799999993pt height=14.15524440000002pt/>. The key question one must ask at this point is, how are the number of `activation values` in <img src="/tex/4e9bc937eb9b385a1891ea5866bb9463.svg?invert_in_darkmode&sanitize=true" align=middle width=25.43581589999999pt height=31.141535699999984pt/> associated to the number of `activation values` in <img src="/tex/533c4a66a96bb16af3cb7da1a9f9b598.svg?invert_in_darkmode&sanitize=true" align=middle width=15.838142099999992pt height=31.141535699999984pt/>? Here is where the magic happens <img src="/tex/e18e054e2d963f4a4d6225355036b639.svg?invert_in_darkmode&sanitize=true" align=middle width=19.474012799999993pt height=14.15524440000002pt/> becomes the input weight for <img src="/tex/0ef7292e4c2074cde496f21400c2fb71.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> and <img src="/tex/7db08e494b0537210daf4ebb9fc5dffb.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/>. This means that <img src="/tex/e18e054e2d963f4a4d6225355036b639.svg?invert_in_darkmode&sanitize=true" align=middle width=19.474012799999993pt height=14.15524440000002pt/> becomes <img src="/tex/0c2385783fe94439987bb7421f6e8dc4.svg?invert_in_darkmode&sanitize=true" align=middle width=78.89643464999999pt height=31.141535699999984pt/> a weight value tensor and the input for <img src="/tex/0ef7292e4c2074cde496f21400c2fb71.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> and <img src="/tex/7db08e494b0537210daf4ebb9fc5dffb.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/>. This means that the activation values tensor for <img src="/tex/0ef7292e4c2074cde496f21400c2fb71.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> is <img src="/tex/779dea4eab1f92036f9f664a299c4283.svg?invert_in_darkmode&sanitize=true" align=middle width=82.20899114999999pt height=31.141535699999984pt/> a tensor with one component because the input layer consist of only one component <img src="/tex/6abb365fad186c6e2bbfa9783072fe89.svg?invert_in_darkmode&sanitize=true" align=middle width=21.75709304999999pt height=22.465723500000017pt/>. Is important to notice that `the number of activation values in a layer's perceptrons are determined by the number of perceptrons in the previous layer`.

For completeness let's consider the output layer `perceptron` <img src="/tex/6d91e89f6ae4bcd1e7056e64d3e3cb61.svg?invert_in_darkmode&sanitize=true" align=middle width=23.59479704999999pt height=22.465723500000017pt/>. Based on our current understanding <img src="/tex/6d91e89f6ae4bcd1e7056e64d3e3cb61.svg?invert_in_darkmode&sanitize=true" align=middle width=23.59479704999999pt height=22.465723500000017pt/> has an internal tensor of `activation values` <img src="/tex/242ea210b6817042e03c419d7d81f000.svg?invert_in_darkmode&sanitize=true" align=middle width=109.65218384999999pt height=31.141535699999984pt/> because <img src="/tex/4e9bc937eb9b385a1891ea5866bb9463.svg?invert_in_darkmode&sanitize=true" align=middle width=25.43581589999999pt height=31.141535699999984pt/> has two components <img src="/tex/0ef7292e4c2074cde496f21400c2fb71.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> and <img src="/tex/7db08e494b0537210daf4ebb9fc5dffb.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/>. The output for <img src="/tex/0ef7292e4c2074cde496f21400c2fb71.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> is a potential <img src="/tex/f2c41641f126a8d3b7a9b24b89185346.svg?invert_in_darkmode&sanitize=true" align=middle width=22.51914059999999pt height=14.15524440000002pt/> and the output for <img src="/tex/7db08e494b0537210daf4ebb9fc5dffb.svg?invert_in_darkmode&sanitize=true" align=middle width=24.80222084999999pt height=22.465723500000017pt/> is a potential <img src="/tex/42d5fd37ed216df9ac993061a1972185.svg?invert_in_darkmode&sanitize=true" align=middle width=22.51914059999999pt height=14.15524440000002pt/> therefore the weight value tensor is <img src="/tex/4b4a9dab3fc7ce7366f50e4e4c3b875b.svg?invert_in_darkmode&sanitize=true" align=middle width=115.63365659999998pt height=31.141535699999984pt/>. The weighted sum for <img src="/tex/6d91e89f6ae4bcd1e7056e64d3e3cb61.svg?invert_in_darkmode&sanitize=true" align=middle width=23.59479704999999pt height=22.465723500000017pt/> is <img src="/tex/d607a841b6980dc90f4755055c478778.svg?invert_in_darkmode&sanitize=true" align=middle width=166.73567789999998pt height=22.465723500000017pt/> and its potential is <img src="/tex/8f189c431a85e64fff451db30183bff2.svg?invert_in_darkmode&sanitize=true" align=middle width=127.72780019999999pt height=24.65753399999998pt/>. Notice how all perceptrons in every layer of the neural network are relaying information directly or indarectly to each other in a forward direction. The type of neural network where all perceptrons are connected to each other is called dense neural network.


## Neural Network Activation

### Introduction

We define a `neural network algorithm` as a function <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> that produces an `output` <img src="/tex/85e71836f40f6c1cb7a2aa1db55ae948.svg?invert_in_darkmode&sanitize=true" align=middle width=17.675844449999992pt height=31.141535699999984pt/> in response to an `input` <img src="/tex/533c4a66a96bb16af3cb7da1a9f9b598.svg?invert_in_darkmode&sanitize=true" align=middle width=15.838142099999992pt height=31.141535699999984pt/> and n number of hidden layers <img src="/tex/7a39f6d0d08d0074f8ed3f1614a10b83.svg?invert_in_darkmode&sanitize=true" align=middle width=49.066130849999986pt height=31.141535699999984pt/> i.e. <img src="/tex/9d4b7ecc2b051cba9574b7bf36271d13.svg?invert_in_darkmode&sanitize=true" align=middle width=141.23278784999997pt height=31.141535699999984pt/>. A neural network is a system defined by the following tensor <img src="/tex/0aee37ed4c8b8524c599452054bc5721.svg?invert_in_darkmode&sanitize=true" align=middle width=238.04027610000003pt height=31.141535699999984pt/>.
In our daily experience we go through time and we have a `state` at each moment in time. Our reality is a series of moments in time. At each moment we can `assess our state` and map any number of metrics to an exact moment in time and persist the resulting information representing our `state`. Our memories are our `state` and we derive knowledge from them. Compare to you or me <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/> is a very simple system, a moment of time for <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/> is represented by evaluating <img src="/tex/46b900f2a449005c3693598536a2adc5.svg?invert_in_darkmode&sanitize=true" align=middle width=141.3987894pt height=31.141535699999984pt/> and <img src="/tex/85e71836f40f6c1cb7a2aa1db55ae948.svg?invert_in_darkmode&sanitize=true" align=middle width=17.675844449999992pt height=31.141535699999984pt/> at a given value of <img src="/tex/533c4a66a96bb16af3cb7da1a9f9b598.svg?invert_in_darkmode&sanitize=true" align=middle width=15.838142099999992pt height=31.141535699999984pt/>. We bring <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/> to life by feeding it `input` and evaluating the `output` of every `perceptron` <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> in each neural network layer <img src="/tex/dc8dc5a2f03a5937263a8b1b75664767.svg?invert_in_darkmode&sanitize=true" align=middle width=11.18724254999999pt height=31.141535699999984pt/>. A neural network is active when <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/> is modified by given algorithm <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>.

### Back Propagation Algorithm

Back propagation is an algorithm or strategy to feed or activate <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/>. The first step to feed <img src="/tex/4e9e77702a1c8278864c977f3ab48980.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=31.141535699999984pt/> is to prepare the information for `input`. The process of preparing the `input` is challenging and can vary depending on what you are trying to accomplish. The last step in the process for preparing input is data normalization, and is usually accomplished by [min-max feature scaling](https://en.wikipedia.org/wiki/Feature_scaling). The function for `min-max feature scaling` is <img src="/tex/d46e2bec7575830fcd5c7bb8e57731ab.svg?invert_in_darkmode&sanitize=true" align=middle width=229.09847399999995pt height=27.7259796pt/> where <img src="/tex/57ca67c2e2bd8e7ca683975c3b79ca5b.svg?invert_in_darkmode&sanitize=true" align=middle width=21.466075949999986pt height=14.15524440000002pt/> is any, <img src="/tex/efb309e17d08a0db7d6167a7037b83c9.svg?invert_in_darkmode&sanitize=true" align=middle width=39.58965779999999pt height=14.15524440000002pt/> is the maximum, and <img src="/tex/240e8d6c2484916597174366d8c77a56.svg?invert_in_darkmode&sanitize=true" align=middle width=37.78182539999999pt height=14.15524440000002pt/> is the minimum of the components in <img src="/tex/533c4a66a96bb16af3cb7da1a9f9b598.svg?invert_in_darkmode&sanitize=true" align=middle width=15.838142099999992pt height=31.141535699999984pt/>

### Error Function

### Gradient Descent
