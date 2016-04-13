## Examples

### Differentiation

Compute the derivative of a function *f* at *x = 9*

```go
  f := func(x Scalar) Scalar {
    return Add(Mul(NewReal(2), Pow(x, NewBareReal(3))), NewReal(4))
  }
  x := NewReal(9)
  Variables(1, x)
  y := f(x)
```
where *y.Value()* returns the function value and *y.Derivative()* the derivative at *x = 9*.

### Gradient descent

Compare vanilla gradient descent with resilient backpropagation
```go
  f := func(x Vector) Scalar {
    // x^4 - 3x^3 + 2
    return Add(Sub(Pow(x[0], NewBareReal(4)), Mul(NewReal(3), Pow(x[0], NewBareReal(3)))), NewReal(2))
  }
  x0 := NewVector(RealType, []float64{8})
  // vanilla gradient descent
  xn1, err1 := gradientDescent.Run(f, x0, 0.0001, gradientDescent.Epsilon{1e-8})
  // resilient backpropagation
  xn2, err2 := rprop.Run(f, x0, 0.0001, 0.4, rprop.Epsilon{1e-8})
```
![Gradient descent](demo/example1/example1.png)


### Matrix inversion

Compute the inverse *r* of a matrix *m* by minimizing the Frobenius norm *||mb - I||*
```go
  m := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})

  I := IdentityMatrix(matrix.Dims()[0])
  r := matrix.Clone()
  // objective function
  f := func(x Vector) Scalar {
    r.SetValues(x)
    s := MNorm(MSub(MMul(matrix, r), I))
    return s
  }
  x, _ := rprop.Run(f, r.Values(), 0.01, 0.1, rprop.Epsilon{1e-12})
  r.SetValues(x)
```

### Newton's method

Find the root of a function *f* with initial value *x0 = (1,1)*

```go
  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := MakeVector(2)
    // y1 = x1^2 + x2^2 - 6
    // y2 = x1^3 - x2^2
    y[0] = Sub(Add(Pow(x[0], NewBareReal(2)), Pow(x[1], NewBareReal(2))), NewReal(6))
    y[1] = Sub(Pow(x[0], NewBareReal(3)), Pow(x[1], NewBareReal(2)))

    return y
  }

  x0    := NewVector(RealType, []float64{1,1})
  xn, _ := newton.Run(f, x0, newton.Epsilon{1e-8})
```
