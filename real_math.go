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

func (a *Real) Neg() Scalar {
  c := NewReal(-a.Value())
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = -a.Derivative(1)
  }
  if c.order >= 2 {
    c.derivative[1] = -a.Derivative(2)
  }
  return c
}

func (a *Real) Add(b Scalar) Scalar {
  c := NewReal(a.Value() + b.Value())
  c.order = IMax(a.Order(), b.Order())
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1) + b.Derivative(1)
  }
  if c.order >= 2 {
    c.derivative[1] = a.Derivative(2) + b.Derivative(2)
  }
  return c
}

func (a *Real) Sub(b Scalar) Scalar {
  c := NewReal(a.Value() - b.Value())
  c.order = IMax(a.Order(), b.Order())
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1) - b.Derivative(1)
  }
  if c.order >= 2 {
    c.derivative[1] = a.Derivative(2) - b.Derivative(2)
  }
  return c
}

func (a *Real) Mul(b Scalar) Scalar {
  c := NewReal(a.Value()*b.Value())
  c.order = IMax(a.Order(), b.Order())
  if c.order >= 1 {
    c.derivative[0] = a.Value()*b.Derivative(1) + a.Derivative(1)*b.Value()
  }
  if c.order >= 2 {
    c.derivative[1] = a.Value()*b.Derivative(2) + a.Derivative(2)*b.Value() + 2*a.Derivative(1)*b.Derivative(1)
  }
  return c
}

func (a *Real) Div(b Scalar) Scalar {
  c := NewReal(a.Value()/b.Value())
  c.order = IMax(a.Order(), b.Order())
  if c.order >= 1 {
    c.derivative[0] = (a.Derivative(1)*b.Value() - a.Value()*b.Derivative(1))/(b.Value()*b.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = (2*a.Value()*math.Pow(b.Derivative(1), 2) + math.Pow(b.Value(), 2)*a.Derivative(2) - b.Value()*(2*a.Derivative(1)*b.Derivative(1) + a.Value()*b.Derivative(2)))/math.Pow(b.Value(), 3)
  }
  return c
}

func (a *Real) Pow(k float64) Scalar {
  c := NewReal(math.Pow(a.Value(), k))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = k*math.Pow(a.Value(), k-1)*a.Derivative(1)
  }
  if c.order >= 2 {
    c.derivative[1] = k*math.Pow(a.Value(), k-1)*a.Derivative(2) + k*(k-1)*math.Pow(a.Value(), k-2)*math.Pow(a.Derivative(1), 2)
  }
  return c
}

func (a *Real) Sqrt() Scalar {
  return a.Pow(1.0/2.0)
}

/* -------------------------------------------------------------------------- */

func (a *Real) Sin() Scalar {
  c := NewReal(math.Sin(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cos(a.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cos(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Sin(a.Value())
  }
  return c
}

func (a *Real) Sinh() Scalar {
  c := NewReal(math.Sinh(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Cosh(a.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Sinh(a.Value())
  }
  return c
}

func (a *Real) Cos() Scalar {
  c := NewReal(math.Cos(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = -a.Derivative(1)*math.Sin(a.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = -a.Derivative(2)*math.Sin(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return c
}

func (a *Real) Cosh() Scalar {
  c := NewReal(math.Cosh(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Sin(a.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = a.Derivative(2)*math.Sin(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Cos(a.Value())
  }
  return c
}

func (a *Real) Tan() Scalar {
  c := NewReal(math.Tan(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0+math.Pow(math.Tan(a.Value()), 2))
  }
  if c.order >= 2 {
    c.derivative[1] = (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return c
}

func (a *Real) Tanh() Scalar {
  c := NewReal(math.Tanh(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*(1.0-math.Pow(math.Tanh(a.Value()), 2))
  }
  if c.order >= 2 {
    c.derivative[1] = (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1), 2))
  }
  return c
}

func (a *Real) Exp() Scalar {
  c := NewReal(math.Exp(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)*math.Exp(a.Value())
  }
  if c.order >= 2 {
    c.derivative[1] = (a.Derivative(2) + math.Pow(a.Derivative(1), 2))*math.Exp(a.Value())
  }
  return c
}

func (a *Real) Log() Scalar {
  c := NewReal(math.Log(a.Value()))
  c.order = a.Order()
  if c.order >= 1 {
    c.derivative[0] = a.Derivative(1)/a.Value()
  }
  if c.order >= 2 {
    c.derivative[1] = (a.Derivative(2)*a.Value() - a.Derivative(1)*a.Derivative(1))/(a.Value()*a.Value())
  }
  return c
}
