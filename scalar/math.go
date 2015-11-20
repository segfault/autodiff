
package scalar

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

func Add(a *Scalar, b *Scalar) *Scalar {
  return &Scalar{a.Value() + b.Value(), a.Derivative() + b.Derivative()}
}

func Sub(a *Scalar, b *Scalar) *Scalar {
  return &Scalar{a.Value() - b.Value(), a.Derivative() - b.Derivative()}
}

func Mul(a *Scalar, b *Scalar) *Scalar {
  return &Scalar{a.Value()*b.Value(), a.Value()*b.Derivative() + a.Derivative()*b.Value()}
}

func Div(a *Scalar, b *Scalar) *Scalar {
  return &Scalar{a.Value()/b.Value(), (a.Derivative()*b.Value() - a.Value()*b.Derivative())/(b.Value()*b.Value())}
}

/* -------------------------------------------------------------------------- */

func Sin(a *Scalar) *Scalar {
  return &Scalar{math.Sin(a.Value()), a.Derivative()*math.Cos(a.Value())}
}

func Cos(a *Scalar) *Scalar {
  return &Scalar{math.Cos(a.Value()), -a.Derivative()*math.Sin(a.Value())}
}

func Exp(a *Scalar) *Scalar {
  return &Scalar{math.Exp(a.Value()), a.Derivative()*math.Exp(a.Value())}
}

func Log(a *Scalar) *Scalar {
  return &Scalar{math.Log(a.Value()), a.Derivative()/a.Value()}
}

func Pow(a *Scalar, k float64) *Scalar {
  return &Scalar{math.Pow(a.Value(), k), k*math.Pow(a.Value(), k-1)*a.Derivative()}
}
