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

/* -------------------------------------------------------------------------- */

func Add(a Scalar, b Scalar) Scalar {
  return Scalar{a.Value() + b.Value(), a.Derivative() + b.Derivative()}
}

func Sub(a Scalar, b Scalar) Scalar {
  return Scalar{a.Value() - b.Value(), a.Derivative() - b.Derivative()}
}

func Mul(a Scalar, b Scalar) Scalar {
  return Scalar{a.Value()*b.Value(), a.Value()*b.Derivative() + a.Derivative()*b.Value()}
}

func Div(a Scalar, b Scalar) Scalar {
  return Scalar{a.Value()/b.Value(), (a.Derivative()*b.Value() - a.Value()*b.Derivative())/(b.Value()*b.Value())}
}

/* -------------------------------------------------------------------------- */

func Sin(a Scalar) Scalar {
  return Scalar{math.Sin(a.Value()), a.Derivative()*math.Cos(a.Value())}
}

func Sinh(a Scalar) Scalar {
  return Scalar{math.Sinh(a.Value()), a.Derivative()*math.Cosh(a.Value())}
}

func Cos(a Scalar) Scalar {
  return Scalar{math.Cos(a.Value()), -a.Derivative()*math.Sin(a.Value())}
}

func Cosh(a Scalar) Scalar {
  return Scalar{math.Cosh(a.Value()), a.Derivative()*math.Sinh(a.Value())}
}

func Tan(a Scalar) Scalar {
  return Scalar{math.Tan(a.Value()), a.Derivative()*(1.0+math.Pow(math.Tan(a.Value()), 2))}
}

func Tanh(a Scalar) Scalar {
  return Scalar{math.Tanh(a.Value()), a.Derivative()*(1.0-math.Pow(math.Tanh(a.Value()), 2))}
}

func Exp(a Scalar) Scalar {
  return Scalar{math.Exp(a.Value()), a.Derivative()*math.Exp(a.Value())}
}

func Log(a Scalar) Scalar {
  return Scalar{math.Log(a.Value()), a.Derivative()/a.Value()}
}

func Pow(a Scalar, k float64) Scalar {
  return Scalar{math.Pow(a.Value(), k), k*math.Pow(a.Value(), k-1)*a.Derivative()}
}
