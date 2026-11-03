# lets learn what derivative is
# increasing a function output by changing value of an input by very small amount
#  f'(x) ≈ (f(x + h) - f(x)) / h
# where h is small amount of change in rate of value


# lets start with small micrograd

from graph import draw_dot
import math 
class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda:None #by default this func doesnt do anything 
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    # to update the gradients of the two input objects when backpropagation happens.
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out

 # used for activation in neural network settings
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

# print(L)
# L.grad=1.0
# L._backward()
# d._backward()
# f._backward()
# e._backward()
# c._backward()
# a._backward() #leaf nodes calling lambda: None.
# b._backward() #leaf nodes calling lambda: None.


# -------------------or just do this 

L.backward()
dot = draw_dot(L)

dot.render("graph", view=True , directory="/Users/ayushjha/Desktop/lab/ml/neural-network-from-scratch/Micrograd/")  # writes graph.svg and opens it

