
Find the capacity of a channel between *X* and *Y* by maximizing the mutual information

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/channel/README//eq_no_01.png" alt="" height="60">


The gradient based approaches use the Lagrangian

<img src="https://raw.githubusercontent.com/pbenner/autodiff/master/demo/channel/README//eq_no_02.png" alt="" height="60">

where
- Newton's method is used to find the roots of the gradient
- Rprop minimized the squared norm of the gradient

![Optimization](channel.png)
