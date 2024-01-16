// clang++ -std=c++17 -I ~/local/include/torch/csrc/api/include/ -I ~/local/include -L ~/local/lib pytorch.cpp -lc10 -ltorch_cpu -rpath ~/local/lib -o pytorch
// https://pytorch.org/cppdocs/
#include <torch/torch.h>
#include <iostream>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

// This will be very helpful for some automated testing!
int main() {
	int seed = 42;
	srand(seed);
	torch::manual_seed(seed);
	at::globalContext().setDeterministicAlgorithms(true, true);

	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << '\n';
	
	torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
	torch::Tensor b = torch::randn({2, 2});
	auto c = (a + b).sum();
	c.backward();
	std::cout << a.grad() << '\n';
	
	return 0;
}
