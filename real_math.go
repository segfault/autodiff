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

/* derivatives of a generic functions
 * -------------------------------------------------------------------------- */

func (c *Real) monadic(a Scalar, f0, f1, f2 float64) Scalar {
  c.AllocForOne(a)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*f2 +
            a.GetDerivative(2, i)*f1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*f1)
    }
  }
  // compute new value
  c.SetValue(f0)
  return c
}

func (c *Real) realMonadic(a *Real, f0, f1, f2 float64) *Real {
  c.AllocForOne(a)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*f2 +
            a.GetDerivative(2, i)*f1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*f1)
    }
  }
  // compute new value
  c.SetValue(f0)
  return c
}

func (c *Real) dyadic(a, b Scalar, f00, f10, f01, f11, f20, f02 float64) Scalar {
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

func (c *Real) realDyadic(a, b Scalar, f00, f10, f01, f11, f20, f02 float64) *Real {
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
  x := a.GetValue()
  return c.monadic(a, -x, -1, 0)
}

func (c *Real) RealNeg(a *Real) *Real {
  x := a.GetValue()
  return c.realMonadic(a, -x, -1, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Add(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.dyadic(a, b, x+y, 1, 1, 0, 0, 0)
}

func (c *Real) RealAdd(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realDyadic(a, b, x+y, 1, 1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Sub(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.dyadic(a, b, x-y, 1, -1, 0, 0, 0)
}

func (c *Real) RealSub(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realDyadic(a, b, x-y, 1, -1, 0, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Mul(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.dyadic(a, b, x*y, y, x, 1, 0, 0)
}

func (c *Real) RealMul(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realDyadic(a, b, x*y, y, x, 1, 0, 0)
}

/* -------------------------------------------------------------------------- */

func (c *Real) Div(a, b Scalar) Scalar {
  x := a.GetValue()
  y := b.GetValue()
  return c.dyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
}

func (c *Real) RealDiv(a, b *Real) *Real {
  x := a.GetValue()
  y := b.GetValue()
  return c.realDyadic(a, b, x/y, 1/y, -x/(y*y), -1/(y*y), 0, 2*x/(y*y*y))
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
  return c.dyadic(a, k, f00, f10, f01, f11, f20, f02)
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
  return c.realDyadic(a, k, f00, f10, f01, f11, f20, f02)
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
  x := a.GetValue()
  f0 :=  math.Sin(x)
  f1 :=  math.Cos(x)
  f2 := -math.Sin(x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Sinh(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Sinh(x)
  f1 :=  math.Cosh(x)
  f2 :=  math.Sinh(x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Cos(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Cos(x)
  f1 := -math.Sin(x)
  f2 := -math.Cos(x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Cosh(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Cosh(x)
  f1 :=  math.Sinh(x)
  f2 :=  math.Cosh(x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Tan(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Tan(x)
  f1 :=  1.0+math.Pow(math.Tan(x), 2)
  f2 :=  2.0*math.Tan(x)*f1
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Tanh(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Tanh(x)
  f1 :=  1.0-math.Pow(math.Tanh(x), 2)
  f2 := -2.0*math.Tanh(x)*f1
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Exp(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Exp(x)
  f1 :=  f0
  f2 :=  f0
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Log(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Log(x)
  f1 :=  1/x
  f2 := -1/(x*x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Log1p(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Log1p(x)
  f1 :=  1/ (1+x)
  f2 := -1/((1+x)*(1+x))
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Erf(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Erf(x)
  f1 :=  2.0/(math.Exp(x*x)*special.M_SQRTPI)
  f2 := -4.0/(math.Exp(x*x)*special.M_SQRTPI)*x
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Erfc(a Scalar) Scalar {
  x := a.GetValue()
  f0 :=  math.Erf(x)
  f1 := -2.0/(math.Exp(x*x)*special.M_SQRTPI)
  f2 :=  4.0/(math.Exp(x*x)*special.M_SQRTPI)*x
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) LogErfc(a Scalar) Scalar {
  x := a.GetValue()
  t := math.Erfc(x)
  f0 :=  special.LogErfc(x)
  f1 := -2.0/(math.Exp(a.GetValue()*a.GetValue())*special.M_SQRTPI*t)
  f2 :=  4.0*(math.Exp(x*x)*special.M_SQRTPI*t*x - 1)/(math.Exp(2*x*x)*math.Pi*t*t)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Gamma(a Scalar) Scalar {
  x := a.GetValue()
  v1 := math.Gamma(x)
  v2 := special.Digamma(x)
  v3 := special.Trigamma(x)
  f0 := v1
  f1 := v1*v2
  f2 := v1*(v2*v2 + v3)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Lgamma(a Scalar) Scalar {
  x := a.GetValue()
  f0, s := math.Lgamma(a.GetValue())
  if s == -1 {
    f0 = math.NaN()
  }
  f1 := special.Digamma(x)
  f2 := special.Trigamma(x)
  return c.monadic(a, f0, f1, f2)
}

func (c *Real) Mlgamma(a Scalar, k int) Scalar {
  x := a.GetValue()
  f0 := special.Mlgamma(x, k)
  f1 := 0.0
  f2 := 0.0
  for j := 1; j <= k; j++ {
    f1 += special.Digamma(x + float64(1-j)/2.0)
  }
  for j := 1; j <= k; j++ {
    f2 += special.Trigamma(x + float64(1-j)/2.0)
  }
  return c.monadic(a, f0, f1, f2)
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
