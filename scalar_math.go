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

func Equal(a, b Scalar) bool {
  return a.Equals(b)
}

func Greater(a, b Scalar) bool {
  return a.Greater(b)
}

func Smaller(a, b Scalar) bool {
  return a.Smaller(b)
}

func Neg(a Scalar) Scalar {
  c := a.Clone()
  return c.Neg(a)
}

func Abs(a Scalar) Scalar {
  c := a.Clone()
  c.Pow(a, NewBareReal(2.0))
  return c.Sqrt(c)
}

func Add(a, b Scalar) Scalar {
  c := a.Clone()
  return c.Add(a, b)
}

func Sub(a, b Scalar) Scalar {
  c := a.Clone()
  return c.Sub(a, b)
}

func Mul(a, b Scalar) Scalar {
  c := a.Clone()
  return c.Mul(a, b)
}

func Div(a, b Scalar) Scalar {
  c := a.Clone()
  return c.Div(a, b)
}

func Pow(a Scalar, k Scalar) Scalar {
  c := a.Clone()
  return c.Pow(a, k)
}

func Sqrt(a Scalar) Scalar {
  c := a.Clone()
  return c.Sqrt(a)
}

func Sin(a Scalar) Scalar {
  c := a.Clone()
  return c.Sin(a)
}

func Sinh(a Scalar) Scalar {
  c := a.Clone()
  return c.Sinh(a)
}

func Cos(a Scalar) Scalar {
  c := a.Clone()
  return c.Cos(a)
}

func Cosh(a Scalar) Scalar {
  c := a.Clone()
  return c.Cosh(a)
}

func Tan(a Scalar) Scalar {
  c := a.Clone()
  return c.Tan(a)
}

func Tanh(a Scalar) Scalar {
  c := a.Clone()
  return c.Tanh(a)
}

func Exp(a Scalar) Scalar {
  c := a.Clone()
  return c.Exp(a)
}

func Log(a Scalar) Scalar {
  c := a.Clone()
  return c.Log(a)
}

func Erf(a Scalar) Scalar {
  c := a.Clone()
  return c.Erf(a)
}

func Erfc(a Scalar) Scalar {
  c := a.Clone()
  return c.Erfc(a)
}

func Gamma(a Scalar) Scalar {
  c := a.Clone()
  return c.Gamma(a)
}

func Mlgamma(a Scalar, k int) Scalar {
  c := a.Clone()
  return c.Mlgamma(a, k)
}
