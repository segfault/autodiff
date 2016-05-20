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

func (a *Real) RealEquals(b *Real) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *Real) Greater(b Scalar) bool {
  return a.Value() > b.Value()
}

func (a *Real) RealGreater(b *Real) bool {
  return a.Value() > b.Value()
}

/* -------------------------------------------------------------------------- */

func (a *Real) Smaller(b Scalar) bool {
  return a.Value() < b.Value()
}

func (a *Real) RealSmaller(b *Real) bool {
  return a.Value() < b.Value()
}

/* -------------------------------------------------------------------------- */

func (c *Real) Neg(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i))
    }
  }
  c.SetValue(-a.Value())
  return c
}

func (c *Real) RealNeg(a *Real) *Real {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i))
    }
  }
  c.SetValue(-a.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Add(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) + b.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) + b.Derivative(1, i))
    }
  }
  c.SetValue(a.Value() + b.Value())
  return c
}

func (c *Real) RealAdd(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) + b.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) + b.Derivative(1, i))
    }
  }
  c.SetValue(a.Value() + b.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sub(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) - b.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) - b.Derivative(1, i))
    }
  }
  c.SetValue(a.Value() - b.Value())
  return c
}

func (c *Real) RealSub(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) - b.Derivative(2, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) - b.Derivative(1, i))
    }
  }
  c.SetValue(a.Value() - b.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Mul(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Value()*b.Derivative(2, i) + a.Derivative(2, i)*b.Value() + 2*a.Derivative(1, i)*b.Derivative(1, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Value()*b.Derivative(1, i) + a.Derivative(1, i)*b.Value())
    }
  }
  c.SetValue(a.Value() * b.Value())
  return c
}

func (c *Real) RealMul(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, a.Value()*b.Derivative(2, i) + a.Derivative(2, i)*b.Value() + 2*a.Derivative(1, i)*b.Derivative(1, i))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, a.Value()*b.Derivative(1, i) + a.Derivative(1, i)*b.Value())
    }
  }
  c.SetValue(a.Value() * b.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Div(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, (2*a.Value()*math.Pow(b.Derivative(1, i), 2) + math.Pow(b.Value(), 2)*a.Derivative(2, i) - b.Value()*(2*a.Derivative(1, i)*b.Derivative(1, i) + a.Value()*b.Derivative(2, i)))/math.Pow(b.Value(), 3))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, (a.Derivative(1, i)*b.Value() - a.Value()*b.Derivative(1, i))/(b.Value()*b.Value()))
    }
  }
  c.SetValue(a.Value() / b.Value())
  return c
}

func (c *Real) RealDiv(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.order >= 2 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(2, i, (2*a.Value()*math.Pow(b.Derivative(1, i), 2) + math.Pow(b.Value(), 2)*a.Derivative(2, i) - b.Value()*(2*a.Derivative(1, i)*b.Derivative(1, i) + a.Value()*b.Derivative(2, i)))/math.Pow(b.Value(), 3))
    }
  }
  if c.order >= 1 {
    for i := 0; i < c.N(); i++ {
      c.SetDerivative(1, i, (a.Derivative(1, i)*b.Value() - a.Value()*b.Derivative(1, i))/(b.Value()*b.Value()))
    }
  }
  c.SetValue(a.Value() / b.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Pow(a, k Scalar) Scalar {
  c.AllocForTwo(a, k)
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
  c.SetValue(math.Pow(a.Value(), k.Value()))
  return c
}

func (c *Real) RealPow(a, k *Real) *Real {
  c.AllocForTwo(a, k)
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
  c.SetValue(math.Pow(a.Value(), k.Value()))
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sqrt(a Scalar) Scalar {
  return c.Pow(a, NewBareReal(1.0/2.0))
}

func (c *Real) RealSqrt(a *Real) *Real {
  return c.RealPow(a, NewReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sin(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cos(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Sin(a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cos(a.Value()))
    }
  }
  c.SetValue(math.Sin(a.Value()))
  return c
}

func (c *Real) Sinh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Sinh(a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cosh(a.Value()))
    }
  }
  c.SetValue(math.Sinh(a.Value()))
  return c
}

func (c *Real) Cos(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i)*math.Sin(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  c.SetValue(math.Cos(a.Value()))
  return c
}

func (c *Real) Cosh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Sin(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  c.SetValue(math.Cosh(a.Value()))
  return c
}

func (c *Real) Tan(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2, i) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0+math.Pow(math.Tan(a.Value()), 2)))
    }
  }
  c.SetValue(math.Tan(a.Value()))
  return c
}

func (c *Real) Tanh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2, i) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0-math.Pow(math.Tanh(a.Value()), 2)))
    }
  }
  c.SetValue(math.Tanh(a.Value()))
  return c
}

func (c *Real) Exp(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i) + math.Pow(a.Derivative(1, i), 2))*math.Exp(a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Exp(a.Value()))
    }
  }
  c.SetValue(math.Exp(a.Value()))
  return c
}

func (c *Real) Log(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i)*a.Value() - a.Derivative(1, i)*a.Derivative(1, i))/(a.Value()*a.Value()))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)/a.Value())
    }
  }
  c.SetValue(a.LogValue())
  return c
}

func (c *Real) Erf(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (2.0*a.Derivative(2, i) - 4.0*a.Value()*a.Derivative(1, i)*a.Derivative(1, i))/(math.Exp(a.Value()*a.Value())*special.M_SQRTPI))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, 2.0*a.Derivative(1, i)/(math.Exp(a.Value()*a.Value())*special.M_SQRTPI))
    }
  }
  c.SetValue(math.Erf(a.Value()))
  return c
}

func (c *Real) Erfc(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -(2.0*a.Derivative(2, i) - 4.0*a.Value()*a.Derivative(1, i)*a.Derivative(1, i))/(math.Exp(a.Value()*a.Value())*special.M_SQRTPI))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -2.0*a.Derivative(1, i)/(math.Exp(a.Value()*a.Value())*special.M_SQRTPI))
    }
  }
  c.SetValue(math.Erfc(a.Value()))
  return c
}

func (c *Real) LogErfc(a Scalar) Scalar {
  c.AllocForOne(a)
  t := math.Erfc(a.Value())
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (math.Exp(a.Value()*a.Value())*special.M_SQRTPI*t*(4*a.Value()*a.Derivative(1, i)*a.Derivative(1, i) - 2*a.Derivative(2, i)) - 4*a.Derivative(1, i)*a.Derivative(1, i))/(math.Exp(2*a.Value()*a.Value())*math.Pi*t*t))
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -2.0*a.Derivative(1, i)/(math.Exp(a.Value()*a.Value())*special.M_SQRTPI*t))
    }
  }
  c.SetValue(special.LogErfc(a.Value()))
  return c
}

func (c *Real) Gamma(a Scalar) Scalar {
  c.AllocForOne(a)
  // preevaluate some expressions
  v1 := math.Gamma(a.Value())
  if c.order >= 1 {
    v2 := special.Digamma(a.Value())
    if c.order >= 2 {
      v3 := special.Trigamma(a.Value())
      for i := 0; i < a.N(); i++ {
        c.SetDerivative(2, i, v1*((v2*v2 + v3)*math.Pow(a.Derivative(1, i), 2.0) + v1*a.Derivative(2, i)))
      }
    }
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, v1*v2*a.Derivative(1, i))
    }
  }
  c.SetValue(v1)
  return c
}

func (c *Real) Lgamma(a Scalar) Scalar {
  v1, s := math.Lgamma(a.Value())
  if s == -1 {
    v1 = math.NaN()
  }
  c.AllocForOne(a)
  if c.order >= 1 {
    v2 := special.Digamma(a.Value())
    if c.order >= 2 {
      v3 := special.Trigamma(a.Value())
      for i := 0; i < a.N(); i++ {
        c.SetDerivative(2, i, v3*math.Pow(a.Derivative(1, i), 2.0) + v2*a.Derivative(2, i))
      }
    }
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, v2*a.Derivative(1, i))
    }
  }
  c.SetValue(v1)
  return c
}

func (c *Real) Mlgamma(a Scalar, k int) Scalar {
  c.AllocForOne(a)
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Trigamma(a.Value() + float64(1-j)/2.0)
      }
      c.SetDerivative(2, i, sum)
    }
  }
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Digamma(a.Value() + float64(1-j)/2.0)
      }
      c.SetDerivative(1, i, sum)
    }
  }
  c.SetValue(special.Mlgamma(a.Value(), k))
  return c
}
