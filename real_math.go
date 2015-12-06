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

func (a *Real) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

func (a *Real) Add(b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := Real{value: a.Value() + b.Value(), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1) + b.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2) + b.Derivative(2)
  }
  return &c
}

func (a *Real) Sub(b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := Real{value: a.Value() - b.Value(), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1) - b.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2) - b.Derivative(2)
  }
  return &c
}

func (a *Real) Mul(b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := Real{value: a.Value()*b.Value(), order: n}
  if n >= 1 {
    c.derivative[0] = a.Value()*b.Derivative(1) + a.Derivative(1)*b.Value()
  }
  if n >= 2 {
    c.derivative[1] = a.Value()*b.Derivative(2) + a.Derivative(2)*b.Value() + 2*a.Derivative(1)*b.Derivative(1)
  }
  return &c
}

func (a *Real) Div(b Scalar) Scalar {
  n := IMax(a.Order(), b.Order())
  c := Real{value: a.Value()/b.Value(), order: n}
  if n >= 1 {
    c.derivative[0] = (a.Derivative(1)*b.Value() - a.Value()*b.Derivative(1))/(b.Value()*b.Value())
  }
  if n >= 2 {
    c.derivative[1] = (2*a.Value()*math.Pow(b.Derivative(1), 2) + math.Pow(b.Value(), 2)*a.Derivative(2) - b.Value()*(2*a.Derivative(1)*b.Derivative(1) + a.Value()*b.Derivative(2)))/math.Pow(b.Value(), 3)
  }
  return &c
}

func (a *Real) Pow(k float64) Scalar {
  n := a.Order()
  c := Real{value: math.Pow(a.Value(), k), order: n}
  if n >= 1 {
    c.derivative[0] = k*math.Pow(a.Value(), k-1)*a.Derivative(1)
  }
  if n >= 2 {
    c.derivative[1] = k*math.Pow(a.Value(), k-1)*a.Derivative(2) + k*(k-1)*math.Pow(a.Value(), k-2)*math.Pow(a.Derivative(1), 2)
  }
  return &c
}

func (a *Real) Sqrt() Scalar {
  return a.Pow(1.0/2.0)
}

/* -------------------------------------------------------------------------- */

func (a *Real) Sin() Scalar {
  n := a.Order()
  c := Real{value: math.Sin(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cos(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cos(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Sin(a.Value())
  }
  return &c
}

func (a *Real) Sinh() Scalar {
  n := a.Order()
  c := Real{value: math.Sinh(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cosh(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Sinh(a.Value())
  }
  return &c
}

func (a *Real) Cos() Scalar {
  n := a.Order()
  c := Real{value: math.Cos(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = -a.Derivative(1)*math.Sin(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = -a.Derivative(2)*math.Sin(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return &c
}

func (a *Real) Cosh() Scalar {
  n := a.Order()
  c := Real{value: math.Cosh(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Sin(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Sin(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return &c
}

func (a *Real) Tan() Scalar {
  n := a.Order()
  c := Real{value: math.Tan(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0+math.Pow(math.Tan(a.Value()), 2))
  }
  if n >= 2 {
    c.derivative[1] = (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return &c
}

func (a *Real) Tanh() Scalar {
  n := a.Order()
  c := Real{value: math.Tanh(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0-math.Pow(math.Tanh(a.Value()), 2))
  }
  if n >= 2 {
    c.derivative[1] = (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return &c
}

func (a *Real) Exp() Scalar {
  n := a.Order()
  c := Real{value: math.Exp(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Exp(a.Value())
  }
  if n >= 2 {
    c.derivative[1] = (a.Derivative(2) + math.Pow(a.Derivative(1), 2))*math.Exp(a.Value())
  }
  return &c
}

func (a *Real) Log() Scalar {
  n := a.Order()
  c := Real{value: math.Log(a.Value()), order: n}
  if n >= 1 {
    c.derivative[0] = a.Derivative(1)/a.Value()
  }
  if n >= 2 {
    c.derivative[1] = (a.Derivative(2)*a.Value() - a.Derivative(1)*a.Derivative(1))/(a.Value()*a.Value())
  }
  return &c
}
