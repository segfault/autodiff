/* Copyright (C) 2015 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package autodiff

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func (a *Real) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

func (a *Real) Greater(b Scalar) bool {
  return a.Value() > b.Value()
}

func (a *Real) Smaller(b Scalar) bool {
  return a.Value() < b.Value()
}

func (a *Real) Neg() Scalar {
  c := NewReal(-a.Value(), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i))
    }
  }
  return c
}

func (a *Real) Add(b Scalar) Scalar {
  n := iMax(a.N(), b.N())
  c := NewReal(a.Value() + b.Value(), n)
  c.order = iMax(a.Order(), b.Order())
  if c.order >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) + b.Derivative(1, i))
    }
  }
  if c.order >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) + b.Derivative(2, i))
    }
  }
  return c
}

func (a *Real) Sub(b Scalar) Scalar {
  n := iMax(a.N(), b.N())
  c := NewReal(a.Value() - b.Value(), n)
  c.order = iMax(a.Order(), b.Order())
  if c.order >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) - b.Derivative(1, i))
    }
  }
  if c.order >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) - b.Derivative(2, i))
    }
  }
  return c
}

func (a *Real) Mul(b Scalar) Scalar {
  n := iMax(a.N(), b.N())
  c := NewReal(a.Value()*b.Value(), n)
  c.order = iMax(a.Order(), b.Order())
  if c.order >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Value()*b.Derivative(1, i) + a.Derivative(1, i)*b.Value())
    }
  }
  if c.order >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Value()*b.Derivative(2, i) + a.Derivative(2, i)*b.Value() + 2*a.Derivative(1, i)*b.Derivative(1, i))
    }
  }
  return c
}

func (a *Real) Div(b Scalar) Scalar {
  n := iMax(a.N(), b.N())
  c := NewReal(a.Value()/b.Value(), n)
  c.order = iMax(a.Order(), b.Order())
  if c.order >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, (a.Derivative(1, i)*b.Value() - a.Value()*b.Derivative(1, i))/(b.Value()*b.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, (2*a.Value()*math.Pow(b.Derivative(1, i), 2) + math.Pow(b.Value(), 2)*a.Derivative(2, i) - b.Value()*(2*a.Derivative(1, i)*b.Derivative(1, i) + a.Value()*b.Derivative(2, i)))/math.Pow(b.Value(), 3))
    }
  }
  return c
}

func (a *Real) Pow(k Scalar) Scalar {
  n := iMax(a.N(), k.N())
  c := NewReal(math.Pow(a.Value(), k.Value()), n)
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < n; i++ {
      if k.Order() >= 1 && k.Derivative(1, i) != 0.0 {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*(
          k.Value()*a.Derivative(1, i) + a.Value()*math.Log(a.Value())*k.Derivative(1, i)))
      } else {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*k.Value()*a.Derivative(1, i))
      }
    }
  }
  if c.order >= 2 {
    for i := 0; i < n; i++ {
      if k.Order() >= 1 && k.Derivative(1, i) != 0.0 {
        c.SetDerivative(2, i,
          math.Pow(a.Value(), k.Value())*(
            (k.Value()-1.0)*k.Value()*math.Pow(a.Derivative(1, i), 2)/math.Pow(a.Value(), 2) +
              (2.0*(1.0 + k.Value()*math.Log(a.Value()))*a.Derivative(1, i)*k.Derivative(1, i) + k.Value()*a.Derivative(2, i))/a.Value() +
              math.Log(a.Value())*(math.Log(a.Value())*math.Pow(k.Derivative(1, i), 2.0) + k.Derivative(2, i))))
      } else {
        c.SetDerivative(2, i, k.Value()*math.Pow(a.Value(), k.Value()-1)*a.Derivative(2, i) + k.Value()*(k.Value()-1)*math.Pow(a.Value(), k.Value()-2)*math.Pow(a.Derivative(1, i), 2))
      }
    }
  }
  return c
}

func (a *Real) Sqrt() Scalar {
  return a.Pow(NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (a *Real) Sin() Scalar {
  c := NewReal(math.Sin(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cos(a.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cos(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Sin(a.Value()))
    }
  }
  return c
}

func (a *Real) Sinh() Scalar {
  c := NewReal(math.Sinh(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cosh(a.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Sinh(a.Value()))
    }
  }
  return c
}

func (a *Real) Cos() Scalar {
  c := NewReal(math.Cos(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i)*math.Sin(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (a *Real) Cosh() Scalar {
  c := NewReal(math.Cosh(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Sin(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (a *Real) Tan() Scalar {
  c := NewReal(math.Tan(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0+math.Pow(math.Tan(a.Value()), 2)))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2, i) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (a *Real) Tanh() Scalar {
  c := NewReal(math.Tanh(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0-math.Pow(math.Tanh(a.Value()), 2)))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2, i) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (a *Real) Exp() Scalar {
  c := NewReal(math.Exp(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Exp(a.Value()))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i) + math.Pow(a.Derivative(1, i), 2))*math.Exp(a.Value()))
    }
  }
  return c
}

func (a *Real) Log() Scalar {
  c := NewReal(math.Log(a.Value()), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)/a.Value())
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i)*a.Value() - a.Derivative(1, i)*a.Derivative(1, i))/(a.Value()*a.Value()))
    }
  }
  return c
}

func (a *Real) Gamma() Scalar {
  c := NewReal(math.Gamma(a.Value()), a.N())
  c.order = a.Order()
  // preevaluate some expressions
  v1 := c.Value()
  if c.order >= 1 {
    v2 := special.Digamma(a.Value())
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, v1*v2*a.Derivative(1, i))
    }
    if c.order >= 2 {
      v3 := special.Trigamma(a.Value())
      for i := 0; i < a.N(); i++ {
        c.SetDerivative(2, i, v1*((v2*v2 + v3)*math.Pow(a.Derivative(1, i), 2.0) + v1*a.Derivative(2, i)))
      }
    }
  }
  return c
}

func (a *Real) Mlgamma(k int) Scalar {
  c := NewReal(special.Mlgamma(a.Value(), k), a.N())
  c.order = a.Order()
  // preevaluate some expressions
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Digamma(a.Value() + float64(1-j)/2.0)
      }
      c.SetDerivative(1, i, sum)
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Trigamma(a.Value() + float64(1-j)/2.0)
      }
      c.SetDerivative(2, i, sum)
    }
  }
  return c
}
