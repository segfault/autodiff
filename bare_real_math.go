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
  if b.Order() > 0 {
    panic("BareReal cannot carry any derivates!")
  }
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

func (a *BareReal) BareRealEquals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Greater(b Scalar) bool {
  return a.Value() > b.Value()
}

func (a *BareReal) BareRealGreater(b *BareReal) bool {
  return a.Value() > b.Value()
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Smaller(b Scalar) bool {
  return a.Value() < b.Value()
}

func (a *BareReal) BareRealSmaller(b *BareReal) bool {
  return a.Value() < b.Value()
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Neg(a Scalar) Scalar {
  checkBare(a)
  c.value = -a.Value()
  return c
}

func (c *BareReal) BareRealNeg(a *BareReal) *BareReal {
  checkBare(a)
  c.value = -a.Value()
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Add(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() + b.Value()
  return c
}

func (c *BareReal) BareRealAdd(a, b *BareReal) *BareReal {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() + b.Value()
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sub(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() - b.Value()
  return c
}

func (c *BareReal) BareRealSub(a, b *BareReal) *BareReal {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() - b.Value()
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Mul(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() * b.Value()
  return c
}

func (c *BareReal) BareRealMul(a, b *BareReal) *BareReal {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() * b.Value()
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Div(a, b Scalar) Scalar {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() / b.Value()
  return c
}

func (c *BareReal) BareRealDiv(a, b *BareReal) *BareReal {
  checkBare(a)
  checkBare(b)
  c.value = a.Value() / b.Value()
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Pow(a, k Scalar) Scalar {
  checkBare(a)
  checkBare(k)
  c.value = math.Pow(a.Value(), k.Value())
  return c
}

func (c *BareReal) BareRealPow(a, k *BareReal) *BareReal {
  checkBare(a)
  checkBare(k)
  c.value = math.Pow(a.Value(), k.Value())
  return c
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sqrt(a Scalar) Scalar {
  checkBare(a)
  return c.Pow(a, NewBareReal(1.0/2.0))
}

func (c *BareReal) BareRealSqrt(a *BareReal) *BareReal {
  checkBare(a)
  return c.BareRealPow(a, NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (c *BareReal) Sin(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Sin(a.Value())
  return c
}

func (c *BareReal) Sinh(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Sinh(a.Value())
  return c
}

func (c *BareReal) Cos(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Cos(a.Value())
  return c
}

func (c *BareReal) Cosh(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Cosh(a.Value())
  return c
}

func (c *BareReal) Tan(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Tan(a.Value())
  return c
}

func (c *BareReal) Tanh(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Tanh(a.Value())
  return c
}

func (c *BareReal) Exp(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Exp(a.Value())
  return c
}

func (c *BareReal) Log(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Log(a.Value())
  return c
}

func (c *BareReal) Erf(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Erf(a.Value())
  return c
}

func (c *BareReal) Erfc(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Erfc(a.Value())
  return c
}

func (c *BareReal) Gamma(a Scalar) Scalar {
  checkBare(a)
  c.value = math.Gamma(a.Value())
  return c
}

func (c *BareReal) Lgamma(a Scalar) Scalar {
  checkBare(a)
  v, s := math.Lgamma(a.Value())
  if s == -1 {
    v = math.NaN()
  }
  c.value = v
  return c
}

func (c *BareReal) Mlgamma(a Scalar, k int) Scalar {
  checkBare(a)
  c.value = special.Mlgamma(a.Value(), k)
  return c
}
