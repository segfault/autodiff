## Examples

### Differentiation

Compute the derivative of a function *f* at *x = 9*

```go
  f := func(x Scalar) Scalar {
    return Add(Mul(NewScalar(2), Pow(x, 3)), NewScalar(4))
  }
  x := NewVariable(9)
  y := f(x)
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
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewScalar(6))
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))

    return y
  }

  x0    := NewVector([]float64{1,1})
  xn, _ := Newton(f, x0, 1e-8)
```
