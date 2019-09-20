import * as tf from '@tensorflow/tfjs-node'

let a = tf.tensor2d([[1,2], [3,4]])
let b = tf.tensor([1, 1])
let c = tf.mul(a,b)
c.print()