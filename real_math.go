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
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

func (a *Real) RealEquals(b *Real) bool {
  epsilon := 1e-12
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *Real) Greater(b Scalar) bool {
  return a.GetValue() > b.GetValue()
}

func (a *Real) RealGreater(b *Real) bool {
  return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *Real) Smaller(b Scalar) bool {
  return a.GetValue() < b.GetValue()
}

func (a *Real) RealSmaller(b *Real) bool {
  return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (c *Real) Neg(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -a.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, -a.GetDerivative(1, i))
    }
  }
  c.SetValue(-a.GetValue())
  return c
}

func (c *Real) RealNeg(a *Real) *Real {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -a.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, -a.GetDerivative(1, i))
    }
  }
  c.SetValue(-a.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Add(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) + b.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) + b.GetDerivative(1, i))
    }
  }
  c.SetValue(a.GetValue() + b.GetValue())
  return c
}

func (c *Real) RealAdd(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) + b.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) + b.GetDerivative(1, i))
    }
  }
  c.SetValue(a.GetValue() + b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sub(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) - b.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) - b.GetDerivative(1, i))
    }
  }
  c.SetValue(a.GetValue() - b.GetValue())
  return c
}

func (c *Real) RealSub(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) - b.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) - b.GetDerivative(1, i))
    }
  }
  c.SetValue(a.GetValue() - b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Mul(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetValue()*b.GetDerivative(2, i) + a.GetDerivative(2, i)*b.GetValue() + 2*a.GetDerivative(1, i)*b.GetDerivative(1, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetValue()*b.GetDerivative(1, i) + a.GetDerivative(1, i)*b.GetValue())
    }
  }
  c.SetValue(a.GetValue() * b.GetValue())
  return c
}

func (c *Real) RealMul(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetValue()*b.GetDerivative(2, i) + a.GetDerivative(2, i)*b.GetValue() + 2*a.GetDerivative(1, i)*b.GetDerivative(1, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetValue()*b.GetDerivative(1, i) + a.GetDerivative(1, i)*b.GetValue())
    }
  }
  c.SetValue(a.GetValue() * b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Div(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, (2*a.GetValue()*math.Pow(b.GetDerivative(1, i), 2) + math.Pow(b.GetValue(), 2)*a.GetDerivative(2, i) - b.GetValue()*(2*a.GetDerivative(1, i)*b.GetDerivative(1, i) + a.GetValue()*b.GetDerivative(2, i)))/math.Pow(b.GetValue(), 3))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, (a.GetDerivative(1, i)*b.GetValue() - a.GetValue()*b.GetDerivative(1, i))/(b.GetValue()*b.GetValue()))
    }
  }
  c.SetValue(a.GetValue() / b.GetValue())
  return c
}

func (c *Real) RealDiv(a, b *Real) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, (2*a.GetValue()*math.Pow(b.GetDerivative(1, i), 2) + math.Pow(b.GetValue(), 2)*a.GetDerivative(2, i) - b.GetValue()*(2*a.GetDerivative(1, i)*b.GetDerivative(1, i) + a.GetValue()*b.GetDerivative(2, i)))/math.Pow(b.GetValue(), 3))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, (a.GetDerivative(1, i)*b.GetValue() - a.GetValue()*b.GetDerivative(1, i))/(b.GetValue()*b.GetValue()))
    }
  }
  c.SetValue(a.GetValue() / b.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *Real) Pow(a, k Scalar) Scalar {
  c.AllocForTwo(a, k)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      if k.GetOrder() >= 1 && k.GetDerivative(1, i) != 0.0 {
        c.SetDerivative(2, i,
          math.Pow(a.GetValue(), k.GetValue())*(
            (k.GetValue()-1.0)*k.GetValue()*math.Pow(a.GetDerivative(1, i), 2)/math.Pow(a.GetValue(), 2) +
              (2.0*(1.0 + k.GetValue()*math.Log(a.GetValue()))*a.GetDerivative(1, i)*k.GetDerivative(1, i) + k.GetValue()*a.GetDerivative(2, i))/a.GetValue() +
              math.Log(a.GetValue())*(math.Log(a.GetValue())*math.Pow(k.GetDerivative(1, i), 2.0) + k.GetDerivative(2, i))))
      } else {
        c.SetDerivative(2, i, k.GetValue()*math.Pow(a.GetValue(), k.GetValue()-1)*a.GetDerivative(2, i) + k.GetValue()*(k.GetValue()-1)*math.Pow(a.GetValue(), k.GetValue()-2)*math.Pow(a.GetDerivative(1, i), 2))
      }
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      if k.GetOrder() >= 1 && k.GetDerivative(1, i) != 0.0 {
        c.SetDerivative(1, i, math.Pow(a.GetValue(), k.GetValue()-1)*(
          k.GetValue()*a.GetDerivative(1, i) + a.GetValue()*math.Log(a.GetValue())*k.GetDerivative(1, i)))
      } else {
        c.SetDerivative(1, i, math.Pow(a.GetValue(), k.GetValue()-1)*k.GetValue()*a.GetDerivative(1, i))
      }
    }
  }
  c.SetValue(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

func (c *Real) RealPow(a, k *Real) *Real {
  c.AllocForTwo(a, k)
  if c.Order >= 2 {
    for i := 0; i < c.GetN(); i++ {
      if k.GetOrder() >= 1 && k.GetDerivative(1, i) != 0.0 {
        c.SetDerivative(2, i,
          math.Pow(a.GetValue(), k.GetValue())*(
            (k.GetValue()-1.0)*k.GetValue()*math.Pow(a.GetDerivative(1, i), 2)/math.Pow(a.GetValue(), 2) +
              (2.0*(1.0 + k.GetValue()*math.Log(a.GetValue()))*a.GetDerivative(1, i)*k.GetDerivative(1, i) + k.GetValue()*a.GetDerivative(2, i))/a.GetValue() +
              math.Log(a.GetValue())*(math.Log(a.GetValue())*math.Pow(k.GetDerivative(1, i), 2.0) + k.GetDerivative(2, i))))
      } else {
        c.SetDerivative(2, i, k.GetValue()*math.Pow(a.GetValue(), k.GetValue()-1)*a.GetDerivative(2, i) + k.GetValue()*(k.GetValue()-1)*math.Pow(a.GetValue(), k.GetValue()-2)*math.Pow(a.GetDerivative(1, i), 2))
      }
    }
  }
  if c.Order >= 1 {
    for i := 0; i < c.GetN(); i++ {
      if k.GetOrder() >= 1 && k.GetDerivative(1, i) != 0.0 {
        c.SetDerivative(1, i, math.Pow(a.GetValue(), k.GetValue()-1)*(
          k.GetValue()*a.GetDerivative(1, i) + a.GetValue()*math.Log(a.GetValue())*k.GetDerivative(1, i)))
      } else {
        c.SetDerivative(1, i, math.Pow(a.GetValue(), k.GetValue()-1)*k.GetValue()*a.GetDerivative(1, i))
      }
    }
  }
  c.SetValue(math.Pow(a.GetValue(), k.GetValue()))
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
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Cos(a.GetValue()) - math.Pow(a.GetDerivative(1, i), 2)*math.Sin(a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Cos(a.GetValue()))
    }
  }
  c.SetValue(math.Sin(a.GetValue()))
  return c
}

func (c *Real) Sinh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Cosh(a.GetValue()) + math.Pow(a.GetDerivative(1, i), 2)*math.Sinh(a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Cosh(a.GetValue()))
    }
  }
  c.SetValue(math.Sinh(a.GetValue()))
  return c
}

func (c *Real) Cos(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -a.GetDerivative(2, i)*math.Sin(a.GetValue()) - math.Pow(a.GetDerivative(1, i), 2)*math.Cos(a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, -a.GetDerivative(1, i)*math.Sin(a.GetValue()))
    }
  }
  c.SetValue(math.Cos(a.GetValue()))
  return c
}

func (c *Real) Cosh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Sin(a.GetValue()) + math.Pow(a.GetDerivative(1, i), 2)*math.Cos(a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Sin(a.GetValue()))
    }
  }
  c.SetValue(math.Cosh(a.GetValue()))
  return c
}

func (c *Real) Tan(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.GetValue()), 2))*(a.GetDerivative(2, i) + 2*math.Tan(a.GetValue())*math.Pow(a.GetDerivative(1, i), 2)))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*(1.0+math.Pow(math.Tan(a.GetValue()), 2)))
    }
  }
  c.SetValue(math.Tan(a.GetValue()))
  return c
}

func (c *Real) Tanh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.GetValue()), 2))*(a.GetDerivative(2, i) - 2*math.Tanh(a.GetValue())*math.Pow(a.GetDerivative(1, i), 2)))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*(1.0-math.Pow(math.Tanh(a.GetValue()), 2)))
    }
  }
  c.SetValue(math.Tanh(a.GetValue()))
  return c
}

func (c *Real) Exp(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (a.GetDerivative(2, i) + math.Pow(a.GetDerivative(1, i), 2))*math.Exp(a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Exp(a.GetValue()))
    }
  }
  c.SetValue(math.Exp(a.GetValue()))
  return c
}

func (c *Real) Log(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (a.GetDerivative(2, i)*a.GetValue() - a.GetDerivative(1, i)*a.GetDerivative(1, i))/(a.GetValue()*a.GetValue()))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)/a.GetValue())
    }
  }
  c.SetValue(a.GetLogValue())
  return c
}

func (c *Real) Erf(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (2.0*a.GetDerivative(2, i) - 4.0*a.GetValue()*a.GetDerivative(1, i)*a.GetDerivative(1, i))/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, 2.0*a.GetDerivative(1, i)/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI))
    }
  }
  c.SetValue(math.Erf(a.GetValue()))
  return c
}

func (c *Real) Erfc(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -(2.0*a.GetDerivative(2, i) - 4.0*a.GetValue()*a.GetDerivative(1, i)*a.GetDerivative(1, i))/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, -2.0*a.GetDerivative(1, i)/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI))
    }
  }
  c.SetValue(math.Erfc(a.GetValue()))
  return c
}

func (c *Real) LogErfc(a Scalar) Scalar {
  c.AllocForOne(a)
  t := math.Erfc(a.GetValue())
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI*t*(4*a.GetValue()*a.GetDerivative(1, i)*a.GetDerivative(1, i) - 2*a.GetDerivative(2, i)) - 4*a.GetDerivative(1, i)*a.GetDerivative(1, i))/(math.Exp(2*a.GetValue()*a.GetValue())*math.Pi*t*t))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, -2.0*a.GetDerivative(1, i)/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI*t))
    }
  }
  c.SetValue(special.LogErfc(a.GetValue()))
  return c
}

func (c *Real) Gamma(a Scalar) Scalar {
  c.AllocForOne(a)
  // preevaluate some expressions
  v1 := math.Gamma(a.GetValue())
  if c.Order >= 1 {
    v2 := special.Digamma(a.GetValue())
    if c.Order >= 2 {
      v3 := special.Trigamma(a.GetValue())
      for i := 0; i < a.GetN(); i++ {
        c.SetDerivative(2, i, v1*((v2*v2 + v3)*math.Pow(a.GetDerivative(1, i), 2.0) + v1*a.GetDerivative(2, i)))
      }
    }
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, v1*v2*a.GetDerivative(1, i))
    }
  }
  c.SetValue(v1)
  return c
}

func (c *Real) Lgamma(a Scalar) Scalar {
  v1, s := math.Lgamma(a.GetValue())
  if s == -1 {
    v1 = math.NaN()
  }
  c.AllocForOne(a)
  if c.Order >= 1 {
    v2 := special.Digamma(a.GetValue())
    if c.Order >= 2 {
      v3 := special.Trigamma(a.GetValue())
      for i := 0; i < a.GetN(); i++ {
        c.SetDerivative(2, i, v3*math.Pow(a.GetDerivative(1, i), 2.0) + v2*a.GetDerivative(2, i))
      }
    }
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, v2*a.GetDerivative(1, i))
    }
  }
  c.SetValue(v1)
  return c
}

func (c *Real) Mlgamma(a Scalar, k int) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Trigamma(a.GetValue() + float64(1-j)/2.0)
      }
      c.SetDerivative(2, i, sum)
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      sum := 0.0
      for j := 1; j <= k; j++ {
        sum += special.Digamma(a.GetValue() + float64(1-j)/2.0)
      }
      c.SetDerivative(1, i, sum)
    }
  }
  c.SetValue(special.Mlgamma(a.GetValue(), k))
  return c
}

/* -------------------------------------------------------------------------- */

func (r *Real) VdotV(a, b Vector) Scalar {
  if len(a) != len(b) {
    panic("vector dimensions do not match")
  }
  t := NullReal()
  for i := 0; i < len(a); i++ {
    t.Mul(a[i], b[i])
    r.Add(r, t)
  }
  return r
}
