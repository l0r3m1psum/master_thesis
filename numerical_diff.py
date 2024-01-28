import torch
At = torch.ones(2,2, requires_grad=True)
Bt = torch.ones(2,2)*2
lt = torch.sum((At@Bt).reshape(-1))
lt.backward()
print(At.grad)

import numpy.typing
An = numpy.ones((2,2))

def f(x1: float, x2: float, x3: float, x4: float) -> float:
	An = numpy.array(((x1, x2), (x3, x4)))
	Bn = numpy.ones((2,2))*2
	return numpy.sum((An@Bn).reshape(-1)).item()

# NOTE: Is there any way to reduce the number of f invocations???
x1 = x2 = x3 = x4 = 1
h = 0.000001
fX = f(x1, x2, x3, x4)
grad1 = (f(x1 + h, x2, x3, x4) - fX)/h
grad2 = (f(x1, x2 + h, x3, x4) - fX)/h
grad3 = (f(x1, x2, x3 + h, x4) - fX)/h
grad4 = (f(x1, x2, x3, x4 + h) - fX)/h
print(grad1, grad2, grad3, grad4)
