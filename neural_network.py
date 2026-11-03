from base import Value
import random
from graph import draw_dot


class Neuron:
    def __init__(self ,num_inputs ):
        self.w = [Value(random.uniform(-1,1)) for i in range(num_inputs)]
        self.b = Value(random.uniform(-1,1))


    def __call__(self, x):
       # print(list(zip(self.w , x))) -> [(Value(data=-0.8176276481255216), 2.0), (Value(data=-0.2572480197797884), 3.0)]

       # w * x + b (forward pass)
       act = sum((wi*xi for wi ,xi in zip(self.w,x)),self.b)
       out = act.tanh()
      #  print(out)
       return out 

# x =[2.0 , 3.0]
# n = Neuron(2)
# n(x)
# return random output Value(data=0.3628976512390918)

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for i in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

  
# x =[2.0 , 3.0]
# n = Layer(2,3)
# n(x)
#[Value(data=-0.9669850169537446), Value(data=0.6958752842093446), Value(data=0.6487042565405848)]

class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
x =[2.0 , 3.0 , -1.0]
n = MLP(3, [4,4,1])
n(x)

# dot = draw_dot(n(x))

# dot.render("mlp", view=True , directory="/Users/ayushjha/Desktop/lab/ml/neural-network-from-scratch/Micrograd/")  # writes graph.svg and opens it

