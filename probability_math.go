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

func (a *Probability) Equals(b Scalar) bool {
  epsilon := 1e-12
  if math.IsNaN(a.LogValue()) && math.IsNaN(b.LogValue()) {
    return true
  }
  if math.IsInf(a.LogValue(), -1) && math.IsInf(b.LogValue(), -1) {
    return true
  }
  if !math.IsInf(a.LogValue(), -1) && !math.IsInf(b.LogValue(), -1) {
    return math.Abs(a.LogValue() - b.LogValue()) < epsilon
  }
  return false
}

func (a *Probability) Greater(b Scalar) bool {
  return a.LogValue() > b.LogValue()
}

func (a *Probability) Smaller(b Scalar) bool {
  return a.LogValue() < b.LogValue()
}

func (a *Probability) Neg() Scalar {
  c := NewReal(-a.Value())
  c.order = a.Order()
  if c.order >= 1 {
    c.SetDerivative(1, -a.Derivative(1))
  }
  if c.order >= 2 {
    c.SetDerivative(2, -a.Derivative(2))
  }
  return c
}

func (a *Probability) Add(b Scalar) Scalar {
  var c Scalar
  if Smaller(b, NewReal(0.0)) {
    // b < 0.0
    c = NewReal(0.0)
    c.(*Real).value = a.Value() - b.Value()
  } else {
    // b >= 0.0
    c = NewProbability(0.0)
    c.(*Probability).value = logAdd(a.LogValue(), b.LogValue())
  }
  c.Variable(iMax(a.Order(), b.Order()))
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1) + b.Derivative(1))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Derivative(2) + b.Derivative(2))
  }
  return c
}

func (a *Probability) Sub(b Scalar) Scalar {
  var c Scalar
  if Smaller(b, NewReal(0.0)) {
    // b < 0.0
    c = NewProbability(0.0)
    c.(*Probability).value = logAdd(a.LogValue(), b.Neg().LogValue())
  } else {
    // b >= 0.0
    c = NewReal(0.0)
    c.(*Real).value = a.Value() - b.Value()
  }
  c.Variable(iMax(a.Order(), b.Order()))
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1) - b.Derivative(1))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Derivative(2) - b.Derivative(2))
  }
  return c
}

func (a *Probability) Mul(b Scalar) Scalar {
  var c Scalar
  if Smaller(b, NewReal(0.0)) {
    c = NewReal(0.0)
    c.(*Real).value = -math.Exp(a.LogValue() + Neg(b).LogValue())
  } else {
    c = NewProbability(0.0)
    c.(*Probability).value = a.LogValue() + b.LogValue()
  }
  c.Variable(iMax(a.Order(), b.Order()))
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Value()*b.Derivative(1) + a.Derivative(1)*b.Value())
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Value()*b.Derivative(2) + a.Derivative(2)*b.Value() + 2*a.Derivative(1)*b.Derivative(1))
  }
  return c
}

func (a *Probability) Div(b Scalar) Scalar {
  var c Scalar
  if Smaller(b, NewReal(0.0)) {
    c = NewReal(0.0)
    c.(*Real).value = -math.Exp(a.LogValue() - Neg(b).LogValue())
  } else {
    c = NewProbability(0.0)
    c.(*Probability).value = a.LogValue() - b.LogValue()
  }
  c.Variable(iMax(a.Order(), b.Order()))
  if c.Order() >= 1 {
    c.SetDerivative(1, (a.Derivative(1)*b.Value() - a.Value()*b.Derivative(1))/(b.Value()*b.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, (2*a.Value()*math.Pow(b.Derivative(1), 2) + math.Pow(b.Value(), 2)*a.Derivative(2) - b.Value()*(2*a.Derivative(1)*b.Derivative(1) + a.Value()*b.Derivative(2)))/math.Pow(b.Value(), 3))
  }
  return c
}

func (a *Probability) Pow(k float64) Scalar {
  c := NewProbability(1.0)
  c.order = a.Order()
  c.value = k*a.LogValue()
  if c.Order() >= 1 {
    c.SetDerivative(1, k*math.Pow(a.Value(), k-1)*a.Derivative(1))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, k*math.Pow(a.Value(), k-1)*a.Derivative(2) + k*(k-1)*math.Pow(a.Value(), k-2)*math.Pow(a.Derivative(1), 2))
  }
  return c
}

func (a *Probability) Sqrt() Scalar {
  return a.Pow(1.0/2.0)
}

/* -------------------------------------------------------------------------- */

func (a *Probability) Sin() Scalar {
  c := NewProbability(math.Sin(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*math.Cos(a.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Derivative(2)*math.Cos(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Sin(a.Value()))
  }
  return c
}

func (a *Probability) Sinh() Scalar {
  c := NewProbability(math.Sinh(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*math.Cosh(a.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Derivative(2)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Sinh(a.Value()))
  }
  return c
}

func (a *Probability) Cos() Scalar {
  c := NewProbability(math.Cos(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, -a.Derivative(1)*math.Sin(a.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, -a.Derivative(2)*math.Sin(a.Value()) - math.Pow(a.Derivative(1), 2)*math.Cos(a.Value()))
  }
  return c
}

func (a *Probability) Cosh() Scalar {
  c := NewProbability(math.Cosh(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*math.Sin(a.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, a.Derivative(2)*math.Sin(a.Value()) + math.Pow(a.Derivative(1), 2)*math.Cos(a.Value()))
  }
  return c
}

func (a *Probability) Tan() Scalar {
  c := NewProbability(math.Tan(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*(1.0+math.Pow(math.Tan(a.Value()), 2)))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1), 2)))
  }
  return c
}

func (a *Probability) Tanh() Scalar {
  c := NewProbability(math.Tanh(a.Value()))
  c.order = a.Order()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*(1.0-math.Pow(math.Tanh(a.Value()), 2)))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1), 2)))
  }
  return c
}

func (a *Probability) Exp() Scalar {
  c := NewProbability(1.0)
  c.order = a.Order()
  c.value = a.Value()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)*math.Exp(a.Value()))
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, (a.Derivative(2) + math.Pow(a.Derivative(1), 2))*math.Exp(a.Value()))
  }
  return c
}

func (a *Probability) Log() Scalar {
  c := NewReal(1.0)
  c.order = a.Order()
  c.value = a.LogValue()
  if c.Order() >= 1 {
    c.SetDerivative(1, a.Derivative(1)/a.Value())
  }
  if c.Order() >= 2 {
    c.SetDerivative(2, (a.Derivative(2)*a.Value() - a.Derivative(1)*a.Derivative(1))/(a.Value()*a.Value()))
  }
  return c
}
