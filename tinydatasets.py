# binary classifiers neural net
from neural_network import Layer , n


xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# lets just predict them
ypred = [n(x) for x in xs]
print(ypred) #[Value(data=-0.8443431044236114), Value(data=0.09848708066320733), Value(data=-0.6971375106656454), Value(data=-0.7200555014821302)]