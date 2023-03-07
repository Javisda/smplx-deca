# Un modelo básico en el que se entrena una red de solo dos parámetros para aprender a convertir grados celsius en kelvin

import torch
from torch import nn
from torch.nn import Parameter

x = torch.tensor([1, 2, 3, 4, 5])
y = x + torch.tensor([273.5], dtype=torch.float)

# Model
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
model = [Parameter(a), Parameter(b)]
lr = 0.05
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model, lr=lr)

for epoch in range(500):
    optimizer.zero_grad()

    y_predicted = model[0] + model[1] * x

    loss = criterion(y_predicted, y)
    loss.backward()
    optimizer.step()

    print("----------------------------------------------------------------")
    print(loss)
    print(model[0], model[1])
    print(x, y, y_predicted)