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
  n := IMax(a.Order(), b.Order())
  c := NewScalar(a.Value() + b.Value(), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1) + b.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2) + b.Derivative(2)
  }
  return c
}

func Sub(a Scalar, b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := NewScalar(a.Value() - b.Value(), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1) - b.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2) - b.Derivative(2)
  }
  return c
}

func Mul(a Scalar, b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := NewScalar(a.Value()*b.Value(), n)
  if n >= 1 {
    c.derivative[0] = a.Value()*b.Derivative(1) + a.Derivative(1)*b.Value()
  }
  if n >= 2 {
    c.derivative[1] = a.Value()*b.Derivative(2) + a.Derivative(2)*b.Value() + 2*a.Derivative(1)*b.Derivative(1)
  }
  return c
}

func Div(a Scalar, b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := NewScalar(a.Value()/b.Value(), n)
  if n >= 1 {
    c.derivative[0] = (a.Derivative(1)*b.Value() - a.Value()*b.Derivative(1))/(b.Value()*b.Value())
  }
  if n >= 2 {
    c.derivative[1] = (2*a.Value()*math.Pow(b.Derivative(1), 2) + math.Pow(b.Value(), 2)*a.Derivative(2) - b.Value()*(2*a.Derivative(1)*b.Derivative(1) + a.Value()*b.Derivative(2)))/math.Pow(b.Value(), 3)
  }
  return c
}

/* -------------------------------------------------------------------------- */

func Sin(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Sin(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cos(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cos(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Sin(a.Value())
  }
  return c
}

func Sinh(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Sinh(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cosh(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Sinh(a.Value())
  }
  return c
}

func Cos(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Cos(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = -a.Derivative(1)*math.Sin(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = -a.Derivative(2)*math.Sin(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return c
}

func Cosh(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Cosh(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Sin(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Sin(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return c
}

func Tan(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Tan(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0+math.Pow(math.Tan(a.Value()), 2))
  }
  if n >= 2 {
    c.derivative[1] = (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return c
}

func Tanh(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Tanh(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0-math.Pow(math.Tanh(a.Value()), 2))
  }
  if n >= 2 {
    c.derivative[1] = (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return c
}

func Exp(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Exp(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Exp(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = (a.Derivative(2) + math.Pow(a.Derivative(1), 2))*math.Exp(a.Value())
  }
  return c
}

func Log(a Scalar) Scalar {
  n := a.Order()
  c := NewScalar(math.Log(a.Value()), n)
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)/a.Value()
  }
  if n >= 2 {
    c.derivative[1] = (a.Derivative(2)*a.Value() - a.Derivative(1)*a.Derivative(1))/(a.Value()*a.Value())
  }
  return c
}

func Pow(a Scalar, k float64) Scalar {
  n := a.Order()
  c := NewScalar(math.Pow(a.Value(), k), n)
  if n >= 1 {
    c.derivative[0] = k*math.Pow(a.Value(), k-1)*a.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = k*math.Pow(a.Value(), k-1)*a.Derivative(2) + k*(k-1)*math.Pow(a.Value(), k-2)*math.Pow(a.Derivative(1), 2)
  }
  return c
}
