# binary classifiers neural net
from neural_network import Layer , n
from graph import draw_dot

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# lets just predict them
# ypred = [n(x) for x in xs] # using same neural net with new sample
# print(ypred) #[Value(data=-0.8443431044236114), Value(data=0.09848708066320733), Value(data=-0.6971375106656454), Value(data=-0.7200555014821302)]

# # as you can see above value is not accurate or near target , to get to the target we need to calculate the loss , the less the loss the better the output
# loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# loss 
# print(loss)
# loss.backward()

# print(n.layers[0].neurons[0].w[0].grad)
# print(n.layers[0].neurons[0].w[0].data) 

# dot = draw_dot(loss)

# dot.render("tiny-data-set-loss", view=True , directory="/Users/ayushjha/Desktop/lab/ml/neural-network-from-scratch/Micrograd/")  # writes graph.svg and opens it


# training loop 
for k in range(20):
  
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update 
  for p in n.parameters():
    p.data += -0.1 * p.grad # the -0.1 is small step size also called as learning rate
  
  print(k, loss.data)


print(ypred)