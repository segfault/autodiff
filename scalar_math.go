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
  return a.Neg()
}

func Abs(a Scalar) Scalar {
  return Sqrt(Pow(a, NewBareReal(2.0)))
}

func Add(a, b Scalar) Scalar {
  return a.Add(b)
}

func Sub(a, b Scalar) Scalar {
  return a.Sub(b)
}

func Mul(a, b Scalar) Scalar {
  return a.Mul(b)
}

func Div(a, b Scalar) Scalar {
  return a.Div(b)
}

func Pow(a Scalar, k Scalar) Scalar {
  return a.Pow(k)
}

func Sqrt(a Scalar) Scalar {
  return a.Sqrt()
}

func Sin(a Scalar) Scalar {
  return a.Sin()
}

func Sinh(a Scalar) Scalar {
  return a.Sinh()
}

func Cos(a Scalar) Scalar {
  return a.Cos()
}

func Cosh(a Scalar) Scalar {
  return a.Cosh()
}

func Tan(a Scalar) Scalar {
  return a.Tan()
}

func Tanh(a Scalar) Scalar {
  return a.Tanh()
}

func Exp(a Scalar) Scalar {
  return a.Exp()
}

func Log(a Scalar) Scalar {
  return a.Log()
}

func Gamma(a Scalar) Scalar {
  return a.Gamma()
}
