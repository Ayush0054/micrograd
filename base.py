# lets learn what derivative is
# increasing a function output by changing value of an input by very small amount
#  f'(x) ≈ (f(x + h) - f(x)) / h
# where h is small amount of change in rate of value


# lets start with small micrograd

from graph import draw_dot

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    return out


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

print(L)

dot = draw_dot(L)
dot.render("graph", view=True , directory="/Users/ayushjha/Desktop/lab/ml/neural-network-from-scratch/Micrograd/")  # writes graph.svg and opens it

