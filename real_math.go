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

/* derivatives of a generic function
 * -------------------------------------------------------------------------- */

func (c *Real) generic(a, b Scalar, f00, f10, f01, f11, f20, f02 float64) Scalar {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*f10 +
            b.GetDerivative(2, i)*f01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*f20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*f02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*f11*2)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*f10 + b.GetDerivative(1, i)*f01)
    }
  }
  // compute new value
  c.SetValue(f00)
  return c
}

func (c *Real) realGeneric(a, b Scalar, f00, f10, f01, f11, f20, f02 float64) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*f10 +
            b.GetDerivative(2, i)*f01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*f20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*f02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*f11*2)
      }
      // compute first derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(1, i, a.GetDerivative(1, i)*f10 + b.GetDerivative(1, i)*f01)
      }
    }
  }
  // compute new value
  c.SetValue(f00)
  return c
}

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

func (a *Real) Min(b Scalar) Scalar {
  if a.GetValue() < b.GetValue() {
    return a
  }
  return b
}

func (a *Real) RealMin(b *Real) Scalar {
  if a.GetValue() < b.GetValue() {
    return a
  }
  return b
}

/* -------------------------------------------------------------------------- */

func (a *Real) Max(b Scalar) Scalar {
  if a.GetValue() > b.GetValue() {
    return a
  }
  return b
}

func (a *Real) RealMax(b *Real) Scalar {
  if a.GetValue() > b.GetValue() {
    return a
  }
  return b
}

/* -------------------------------------------------------------------------- */

func (c *Real) Abs(a Scalar) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  }
  return c
}

func (c *Real) RealAbs(a *Real) Scalar {
  if c.Sign() == -1 {
    c.Neg(a)
  }
  return c
}

/* -------------------------------------------------------------------------- */

func (a *Real) Sign() int {
  if a.GetValue() < 0.0 {
    return -1
  }
  if a.GetValue() > 0.0 {
    return  1
  }
  return 0
}

func (a *Real) RealSign() int {
  if a.GetValue() < 0.0 {
    return -1
  }
  if a.GetValue() > 0.0 {
    return  1
  }
  return 0
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
  x := a.GetValue()
  y := b.GetValue()
  return c.generic(a, b, x+y, 1, 1, 0, 0, 0)
}

func (c *Real) RealAdd(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realGeneric(a, b, x+y, 1, 1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sub(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.generic(a, b, x-y, 1, -1, 0, 0, 0)
}

func (c *Real) RealSub(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realGeneric(a, b, x-y, 1, -1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Mul(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.generic(a, b, x*y, y, x, 1, 0, 0)
}

func (c *Real) RealMul(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realGeneric(a, b, x*y, y, x, 1, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Div(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.generic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}

func (c *Real) RealDiv(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realGeneric(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}

/* -------------------------------------------------------------------------- */

func (c *Real) Pow(a, k Scalar) Scalar {
  x := a.GetValue()
  y := k.GetValue()
  f00 := math.Pow(x, y)
  f10 := math.Pow(x, y-1)*y
  f01 := math.Pow(x, y-0)*math.Log(x)
  f11 := math.Pow(x, y-1)*(1 + y*math.Log(x))
  f20 := math.Pow(x, y-2)*(y - 1)*y
  f02 := math.Pow(x, y-0)*math.Log(x)*math.Log(x)
  return c.generic(a, k, f00, f10, f01, f11, f20, f02)
}

func (c *Real) RealPow(a, k *Real) *Real {
  x := a.GetValue()
  y := k.GetValue()
  f00 := math.Pow(x, y)
  f10 := math.Pow(x, y-1)*y
  f01 := math.Pow(x, y-0)*math.Log(x)
  f11 := math.Pow(x, y-1)*(1 + y*math.Log(x))
  f20 := math.Pow(x, y-2)*(y - 1)*y
  f02 := math.Pow(x, y-0)*math.Log(x)*math.Log(x)
  return c.realGeneric(a, k, f00, f10, f01, f11, f20, f02)
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
      c.SetDerivative(2, i, a.GetDerivative(2, i)/a.GetValue() - a.GetDerivative(1, i)*a.GetDerivative(1, i)/(a.GetValue()*a.GetValue()))
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

func (c *Real) Log1p(a Scalar) Scalar {
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

func (c *Real) GammaP(a float64, x Scalar) Scalar {
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

func (r *Real) VdotV(a, b Vector) Scalar {
  if len(a) != len(b) {
    panic("vector dimensions do not match")
  }
  r.Reset()
  t := NullReal()
  for i := 0; i < len(a); i++ {
    t.Mul(a[i], b[i])
    r.Add(r, t)
  }
  return r
}

func (r *Real) Vnorm(a Vector) Scalar {
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
