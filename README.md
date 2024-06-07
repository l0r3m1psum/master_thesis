# My master thesis
 My original idea was to implement a simple deep learnign library with the
 ability to generate on demand fused GPU kernels, like Jittor or `tinygrad`...
 This didn't go as planned due to how hard it was to find information about
 dering vJps for back-proapgating the gradient through matrix operatoins. In
 return I have managed to find a very simple way to derive this results by hand
 using a matrix calculus from two econometricians `:)`.

I have also implemented some simple code that can train a neural network on GPU
(using Apple's Metal.)
