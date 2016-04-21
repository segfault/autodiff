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

import "math"

import "github.com/pbenner/autodiff/special"

/* -------------------------------------------------------------------------- */

func (a *Probability) Equals(b Scalar) bool {
  epsilon := 1e-12
  if math.IsNaN(a.LogValue()) && math.IsNaN(b.LogValue()) {
    return true
  }
  if math.IsInf(a.LogValue(), -1) && math.IsInf(b.LogValue(), -1) {
    return true
  }
  if !math.IsInf(a.LogValue(), -1) && !math.IsInf(b.LogValue(), -1) {
    return math.Abs(a.LogValue() - b.LogValue()) < epsilon
  }
  return false
}

func (a *Probability) Greater(b Scalar) bool {
  return a.LogValue() > b.LogValue()
}

func (a *Probability) Smaller(b Scalar) bool {
  return a.LogValue() < b.LogValue()
}

func (c *Probability) Neg(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(-a.Value())
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

func (c *Probability) Add(a, b Scalar) Scalar {
  c.AllocFor(a, b)
  c.value = logAdd(a.LogValue(), b.LogValue())
  if c.Order() >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) + b.Derivative(1, i))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) + b.Derivative(2, i))
    }
  }
  return c
}

func (c *Probability) Sub(a, b Scalar) Scalar {
  c.AllocFor(a, b)
  c.value = logSub(a.LogValue(), b.LogValue())
  if c.Order() >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) - b.Derivative(1, i))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) - b.Derivative(2, i))
    }
  }
  return c
}

func (c *Probability) Mul(a, b Scalar) Scalar {
  c.AllocFor(a, b)
  c.value = a.LogValue() + b.LogValue()
  if c.Order() >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Value()*b.Derivative(1, i) + a.Derivative(1, i)*b.Value())
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Value()*b.Derivative(2, i) + a.Derivative(2, i)*b.Value() + 2*a.Derivative(1, i)*b.Derivative(1, i))
    }
  }
  return c
}

func (c *Probability) Div(a, b Scalar) Scalar {
  c.AllocFor(a, b)
  c.value = a.LogValue() - b.LogValue()
  if c.Order() >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, (a.Derivative(1, i)*b.Value() - a.Value()*b.Derivative(1, i))/(b.Value()*b.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, (2*a.Value()*math.Pow(b.Derivative(1, i), 2) + math.Pow(b.Value(), 2)*a.Derivative(2, i) - b.Value()*(2*a.Derivative(1, i)*b.Derivative(1, i) + a.Value()*b.Derivative(2, i)))/math.Pow(b.Value(), 3))
    }
  }
  return c
}

func (c *Probability) Pow(a, k Scalar) Scalar {
  c.AllocFor(a, k)
  c.value = k.Value()*a.LogValue()
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      if k.Order() >= 1 && k.Derivative(1, i) != 0.0 {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*(
          k.Value()*a.Derivative(1, i) + a.Value()*math.Log(a.Value())*k.Derivative(1, i)))
      } else {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*k.Value()*a.Derivative(1, i))
      }
    }
  }
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
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

func (c *Probability) Sqrt(a Scalar) Scalar {
  return c.Pow(a, NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *Probability) Sin(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Sin(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cos(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cos(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Sin(a.Value()))
    }
  }
  return c
}

func (c *Probability) Sinh(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Sinh(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cosh(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Sinh(a.Value()))
    }
  }
  return c
}

func (c *Probability) Cos(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Cos(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i)*math.Sin(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (c *Probability) Cosh(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Cosh(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Sin(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (c *Probability) Tan(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Tan(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0+math.Pow(math.Tan(a.Value()), 2)))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2, i) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (c *Probability) Tanh(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Tanh(a.Value()))
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0-math.Pow(math.Tanh(a.Value()), 2)))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2, i) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (c *Probability) Exp(a Scalar) Scalar {
  c.AllocFor(a)
  c.value = a.Value()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Exp(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i) + math.Pow(a.Derivative(1, i), 2))*math.Exp(a.Value()))
    }
  }
  return c
}

func (c *Probability) Log(a Scalar) Scalar {
  c.AllocFor(a)
  c.value = math.Log(a.LogValue())
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)/a.Value())
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i)*a.Value() - a.Derivative(1, i)*a.Derivative(1, i))/(a.Value()*a.Value()))
    }
  }
  return c
}

func (c *Probability) Erf(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Erf(a.Value()))
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, 2.0*a.Derivative(1, i)/(math.Exp(a.Value()*a.Value())*math.Sqrt(math.Pi)))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (2.0*a.Derivative(2, i) - 4.0*a.Value()*a.Derivative(1, i)*a.Derivative(1, i))/(math.Exp(a.Value()*a.Value())*math.Sqrt(math.Pi)))
    }
  }
  return c
}

func (c *Probability) Erfc(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Erfc(a.Value()))
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -2.0*a.Derivative(1, i)/(math.Exp(a.Value()*a.Value())*math.Sqrt(math.Pi)))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -(2.0*a.Derivative(2, i) - 4.0*a.Value()*a.Derivative(1, i)*a.Derivative(1, i))/(math.Exp(a.Value()*a.Value())*math.Sqrt(math.Pi)))
    }
  }
  return c
}

func (c *Probability) Gamma(a Scalar) Scalar {
  c.AllocFor(a)
  c.SetValue(math.Gamma(a.Value()))
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

func (c *Probability) Lgamma(a Scalar) Scalar {
  v1, s := math.Lgamma(a.Value())
  if s == -1 {
    v1 = math.NaN()
  }
  c.AllocFor(a)
  c.SetValue(v1)
  if c.order >= 1 {
    v2 := special.Digamma(a.Value())
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, v2*a.Derivative(1, i))
    }
    if c.order >= 2 {
      v3 := special.Trigamma(a.Value())
      for i := 0; i < a.N(); i++ {
        c.SetDerivative(2, i, v3*math.Pow(a.Derivative(1, i), 2.0) + v2*a.Derivative(2, i))
      }
    }
  }
  return c
}

func (c *Probability) Mlgamma(a Scalar, k int) Scalar {
  c.AllocFor(a)
  c.SetValue(special.Mlgamma(a.Value(), k))
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
