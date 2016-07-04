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

func checkBare(b Scalar) {
  if b.GetOrder() > 0 {
    panic("BareReal cannot carry any derivates!")
  }
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

func (a *BareReal) BareRealEquals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.GetValue() - b.GetValue()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Greater(b Scalar) bool {
  return a.GetValue() > b.GetValue()
}

func (a *BareReal) BareRealGreater(b *BareReal) bool {
  return a.GetValue() > b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Smaller(b Scalar) bool {
  return a.GetValue() < b.GetValue()
}

func (a *BareReal) BareRealSmaller(b *BareReal) bool {
  return a.GetValue() < b.GetValue()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Negative() bool {
  return a.GetValue() < 0.0
}

func (a *BareReal) RealNegative() bool {
  return a.GetValue() < 0.0
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Neg(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(-a.GetValue())
  return c
}

func (c *BareReal) BareRealNeg(a *BareReal) *BareReal {
  *c = BareReal(-a.GetValue())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Add(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() + b.GetValue())
  return c
}

func (c *BareReal) BareRealAdd(a, b *BareReal) *BareReal {
  *c = *a + *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sub(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() - b.GetValue())
  return c
}

func (c *BareReal) BareRealSub(a, b *BareReal) *BareReal {
  *c = *a - *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Mul(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() * b.GetValue())
  return c
}

func (c *BareReal) BareRealMul(a, b *BareReal) *BareReal {
  *c = *a * *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Div(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  *c = BareReal(a.GetValue() / b.GetValue())
  return c
}

func (c *BareReal) BareRealDiv(a, b *BareReal) *BareReal {
  *c = *a / *b
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Pow(a, k Scalar) Scalar {
  checkBare(a)
  checkBare(k)
  *c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

func (c *BareReal) BareRealPow(a, k *BareReal) *BareReal {
  *c = BareReal(math.Pow(a.GetValue(), k.GetValue()))
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sqrt(a Scalar) Scalar {
  checkBare(a)
  return c.Pow(a, NewBareReal(1.0/2.0))
}

func (c *BareReal) BareRealSqrt(a *BareReal) *BareReal {
  return c.BareRealPow(a, NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sin(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sin(a.GetValue()))
  return c
}

func (c *BareReal) Sinh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Sinh(a.GetValue()))
  return c
}

func (c *BareReal) Cos(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cos(a.GetValue()))
  return c
}

func (c *BareReal) Cosh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Cosh(a.GetValue()))
  return c
}

func (c *BareReal) Tan(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tan(a.GetValue()))
  return c
}

func (c *BareReal) Tanh(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Tanh(a.GetValue()))
  return c
}

func (c *BareReal) Exp(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Exp(a.GetValue()))
  return c
}

func (c *BareReal) Log(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Log(a.GetValue()))
  return c
}

func (c *BareReal) Erf(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erf(a.GetValue()))
  return c
}

func (c *BareReal) Erfc(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Erfc(a.GetValue()))
  return c
}

func (c *BareReal) LogErfc(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(special.LogErfc(a.GetValue()))
  return c
}

func (c *BareReal) Gamma(a Scalar) Scalar {
  checkBare(a)
  *c = BareReal(math.Gamma(a.GetValue()))
  return c
}

func (c *BareReal) Lgamma(a Scalar) Scalar {
  checkBare(a)
  v, s := math.Lgamma(a.GetValue())
  if s == -1 {
    v = math.NaN()
  }
  *c = BareReal(v)
  return c
}

func (c *BareReal) Mlgamma(a Scalar, k int) Scalar {
  checkBare(a)
  *c = BareReal(special.Mlgamma(a.GetValue(), k))
  return c
}

/* -------------------------------------------------------------------------- */

func (r *BareReal) VdotV(a, b Vector) Scalar {
  if len(a) != len(b) {
    panic("vector dimensions do not match")
  }
  t := NullBareReal()
  for i := 0; i < len(a); i++ {
    t.Mul(a[i], b[i])
    r.Add(r, t)
  }
  return r
}
