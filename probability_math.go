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
  if math.IsNaN(a.GetLogValue()) && math.IsNaN(b.GetLogValue()) {
    return true
  }
  if math.IsInf(a.GetLogValue(), -1) && math.IsInf(b.GetLogValue(), -1) {
    return true
  }
  if !math.IsInf(a.GetLogValue(), -1) && !math.IsInf(b.GetLogValue(), -1) {
    return math.Abs(a.GetLogValue() - b.GetLogValue()) < epsilon
  }
  return false
}

func (a *Probability) Greater(b Scalar) bool {
  return a.GetLogValue() > b.GetLogValue()
}

func (a *Probability) Smaller(b Scalar) bool {
  return a.GetLogValue() < b.GetLogValue()
}

func (a *Probability) Min(b Scalar) Scalar {
  if a.GetLogValue() < b.GetLogValue() {
    return a
  }
  return b
}

func (a *Probability) Max(b Scalar) Scalar {
  if a.GetLogValue() > b.GetLogValue() {
    return a
  }
  return b
}

func (c *Probability) Abs(a Scalar) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  }
  return c
}

func (a *Probability) Sign() int {
  if math.IsInf(a.Value, -1) {
    return 0
  }
  return 1
}

func (c *Probability) Neg(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -a.GetDerivative(2, i))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, -a.GetDerivative(1, i))
    }
  }
  c.SetValue(-a.GetValue())
  return c
}

func (c *Probability) Add(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.GetOrder() >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) + b.GetDerivative(2, i))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) + b.GetDerivative(1, i))
    }
  }
  c.Value = logAdd(a.GetLogValue(), b.GetLogValue())
  return c
}

func (c *Probability) Sub(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.GetOrder() >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i) - b.GetDerivative(2, i))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i) - b.GetDerivative(1, i))
    }
  }
  c.Value = logSub(a.GetLogValue(), b.GetLogValue())
  return c
}

func (c *Probability) Mul(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.GetOrder() >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, a.GetValue()*b.GetDerivative(2, i) + a.GetDerivative(2, i)*b.GetValue() + 2*a.GetDerivative(1, i)*b.GetDerivative(1, i))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetValue()*b.GetDerivative(1, i) + a.GetDerivative(1, i)*b.GetValue())
    }
  }
  c.Value = a.GetLogValue() + b.GetLogValue()
  return c
}

func (c *Probability) Div(a, b Scalar) Scalar {
  c.AllocForTwo(a, b)
  if c.GetOrder() >= 2 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(2, i, (2*a.GetValue()*math.Pow(b.GetDerivative(1, i), 2) + math.Pow(b.GetValue(), 2)*a.GetDerivative(2, i) - b.GetValue()*(2*a.GetDerivative(1, i)*b.GetDerivative(1, i) + a.GetValue()*b.GetDerivative(2, i)))/math.Pow(b.GetValue(), 3))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, (a.GetDerivative(1, i)*b.GetValue() - a.GetValue()*b.GetDerivative(1, i))/(b.GetValue()*b.GetValue()))
    }
  }
  c.Value = a.GetLogValue() - b.GetLogValue()
  return c
}

func (c *Probability) Pow(a, k Scalar) Scalar {
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
  c.Value = k.GetValue()*a.GetLogValue()
  return c
}

func (c *Probability) Sqrt(a Scalar) Scalar {
  return c.Pow(a, NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *Probability) Sin(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Cos(a.GetValue()) - math.Pow(a.GetDerivative(1, i), 2)*math.Sin(a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Cos(a.GetValue()))
    }
  }
  c.SetValue(math.Sin(a.GetValue()))
  return c
}

func (c *Probability) Sinh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Cosh(a.GetValue()) + math.Pow(a.GetDerivative(1, i), 2)*math.Sinh(a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Cosh(a.GetValue()))
    }
  }
  c.SetValue(math.Sinh(a.GetValue()))
  return c
}

func (c *Probability) Cos(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, -a.GetDerivative(2, i)*math.Sin(a.GetValue()) - math.Pow(a.GetDerivative(1, i), 2)*math.Cos(a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, -a.GetDerivative(1, i)*math.Sin(a.GetValue()))
    }
  }
  c.SetValue(math.Cos(a.GetValue()))
  return c
}

func (c *Probability) Cosh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)*math.Sin(a.GetValue()) + math.Pow(a.GetDerivative(1, i), 2)*math.Cos(a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Sin(a.GetValue()))
    }
  }
  c.SetValue(math.Cosh(a.GetValue()))
  return c
}

func (c *Probability) Tan(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.GetValue()), 2))*(a.GetDerivative(2, i) + 2*math.Tan(a.GetValue())*math.Pow(a.GetDerivative(1, i), 2)))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*(1.0+math.Pow(math.Tan(a.GetValue()), 2)))
    }
  }
  c.SetValue(math.Tan(a.GetValue()))
  return c
}

func (c *Probability) Tanh(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.GetValue()), 2))*(a.GetDerivative(2, i) - 2*math.Tanh(a.GetValue())*math.Pow(a.GetDerivative(1, i), 2)))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*(1.0-math.Pow(math.Tanh(a.GetValue()), 2)))
    }
  }
  c.SetValue(math.Tanh(a.GetValue()))
  return c
}

func (c *Probability) Exp(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, (a.GetDerivative(2, i) + math.Pow(a.GetDerivative(1, i), 2))*math.Exp(a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*math.Exp(a.GetValue()))
    }
  }
  c.Value = a.GetValue()
  return c
}

func (c *Probability) Log(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.GetOrder() >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)/a.GetValue() - a.GetDerivative(1, i)*a.GetDerivative(1, i)/(a.GetValue()*a.GetValue()))
    }
  }
  if c.GetOrder() >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)/a.GetValue())
    }
  }
  c.Value = math.Log(a.GetLogValue())
  return c
}

func (c *Probability) Log1p(a Scalar) Scalar {
  c.AllocForOne(a)
  if c.Order >= 2 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(2, i, a.GetDerivative(2, i)/(1.0 + a.GetValue()) - a.GetDerivative(1, i)*a.GetDerivative(1, i)/((1.0 + a.GetValue())*(1.0 + a.GetValue())))
    }
  }
  if c.Order >= 1 {
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)/(1.0 + a.GetValue()))
    }
  }
  c.SetValue(math.Log1p(a.GetValue()))
  return c
}

func (c *Probability) Erf(a Scalar) Scalar {
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

func (c *Probability) Erfc(a Scalar) Scalar {
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

func (c *Probability) LogErfc(a Scalar) Scalar {
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

func (c *Probability) Gamma(a Scalar) Scalar {
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

func (c *Probability) Lgamma(a Scalar) Scalar {
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

func (c *Probability) Mlgamma(a Scalar, k int) Scalar {
  c.AllocForOne(a)
  v1 := special.Mlgamma(a.GetValue(), k)
  if c.Order >= 1 {
    v2 := 0.0
    for j := 1; j <= k; j++ {
      v2 += special.Digamma(a.GetValue() + float64(1-j)/2.0)
    }
    if c.Order >= 2 {
      v3 := 0.0
      for j := 1; j <= k; j++ {
        v3 += special.Trigamma(a.GetValue() + float64(1-j)/2.0)
      }
      for i := 0; i < a.GetN(); i++ {
        c.SetDerivative(2, i, a.GetDerivative(2, i)*v2 + a.GetDerivative(1, i)*a.GetDerivative(1, i)*v3)
      }
    }
    for i := 0; i < a.GetN(); i++ {
      c.SetDerivative(1, i, v2*c.GetDerivative(1, i))
    }
  }
  c.SetValue(v1)
  return c
}

func (c *Probability) GammaP(a float64, x Scalar) Scalar {
  c.AllocForOne(x)
  // preevaluate some expressions
  v1 := special.GammaP(a, x.GetValue())
  if c.Order >= 1 {
    v2 := special.GammaPfirstDerivative(a, x.GetValue())
    if c.Order >= 2 {
      v3 := special.GammaPsecondDerivative(a, x.GetValue())
      for i := 0; i < x.GetN(); i++ {
        c.SetDerivative(2, i, x.GetDerivative(2, i)*v2 + x.GetDerivative(1, i)*x.GetDerivative(1, i)*v3)
      }
    }
    for i := 0; i < x.GetN(); i++ {
      c.SetDerivative(1, i, x.GetDerivative(1, i)*v2)
    }
  }
  c.SetValue(v1)
  return c
}

/* -------------------------------------------------------------------------- */

func (r *Probability) VdotV(a, b Vector) Scalar {
  if len(a) != len(b) {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullProbability()
  for i := 0; i < len(a); i++ {
    t.Mul(a[i], b[i])
    r.Add(r, t)
  }
  return r
}

func (r *Probability) Vnorm(a Vector) Scalar {
  r.Reset()
  c := NewBareReal(2.0)
  t := NullScalar(a.ElementType())
  for i := 0; i < len(a); i++ {
    t.Pow(a[i], c)
    r.Add(r, t)
  }
  r.Sqrt(r)
  return r
}
