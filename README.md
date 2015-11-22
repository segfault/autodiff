## Examples

### Newton's method

```go

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := MakeVector(2)
    // x1^2 + y^2 - 6
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewScalar(6))
    // x^3 - y^2
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))

    return y
  }

  v1    := NewVector([]float64{1,1})
  v2, _ := Newton(f, v1, 1e-8)
  v3    := NewVector([]float64{1.537656, 1.906728})
```
