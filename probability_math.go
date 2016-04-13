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
  c := NewReal(-a.Value(), a.N())
  c.order = a.Order()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i))
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i))
    }
  }
  return c
}

func (a *Probability) Add(b Scalar) Scalar {
  var c Scalar
  n := iMax(a.N(), b.N())
  if Smaller(b, NewReal(0.0)) {
    // b < 0.0
    c = NewReal(0.0, n)
    c.(*Real).order = iMax(a.Order(), b.Order())
    c.(*Real).value = a.Value() + b.Value()
  } else {
    // b >= 0.0
    c = NewProbability(0.0, n)
    c.(*Probability).order = iMax(a.Order(), b.Order())
    c.(*Probability).value = logAdd(a.LogValue(), b.LogValue())
  }
  if c.Order() >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) + b.Derivative(1, i))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) + b.Derivative(2, i))
    }
  }
  return c
}

func (a *Probability) Sub(b Scalar) Scalar {
  var c Scalar
  n := iMax(a.N(), b.N())
  if Smaller(b, NewReal(0.0)) {
    // b < 0.0
    c = NewProbability(0.0, n)
    c.(*Probability).order = iMax(a.Order(), b.Order())
    c.(*Probability).value = logAdd(a.LogValue(), b.Neg().LogValue())
  } else {
    // b >= 0.0
    c = NewReal(0.0, n)
    c.(*Real).order = iMax(a.Order(), b.Order())
    c.(*Real).value = a.Value() - b.Value()
  }
  if c.Order() >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Derivative(1, i) - b.Derivative(1, i))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Derivative(2, i) - b.Derivative(2, i))
    }
  }
  return c
}

func (a *Probability) Mul(b Scalar) Scalar {
  var c Scalar
  n := iMax(a.N(), b.N())
  if Smaller(b, NewReal(0.0)) {
    c = NewReal(0.0, n)
    c.(*Real).order = iMax(a.Order(), b.Order())
    c.(*Real).value = -math.Exp(a.LogValue() + Neg(b).LogValue())
  } else {
    c = NewProbability(0.0, n)
    c.(*Probability).order = iMax(a.Order(), b.Order())
    c.(*Probability).value = a.LogValue() + b.LogValue()
  }
  if c.Order() >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, a.Value()*b.Derivative(1, i) + a.Derivative(1, i)*b.Value())
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, a.Value()*b.Derivative(2, i) + a.Derivative(2, i)*b.Value() + 2*a.Derivative(1, i)*b.Derivative(1, i))
    }
  }
  return c
}

func (a *Probability) Div(b Scalar) Scalar {
  var c Scalar
  n := iMax(a.N(), b.N())
  if Smaller(b, NewReal(0.0)) {
    c = NewReal(0.0, n)
    c.(*Real).order = iMax(a.Order(), b.Order())
    c.(*Real).value = -math.Exp(a.LogValue() - Neg(b).LogValue())
  } else {
    c = NewProbability(0.0, n)
    c.(*Probability).order = iMax(a.Order(), b.Order())
    c.(*Probability).value = a.LogValue() - b.LogValue()
  }
  if c.Order() >= 1 {
    for i := 0; i < n; i++ {
      c.SetDerivative(1, i, (a.Derivative(1, i)*b.Value() - a.Value()*b.Derivative(1, i))/(b.Value()*b.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < n; i++ {
      c.SetDerivative(2, i, (2*a.Value()*math.Pow(b.Derivative(1, i), 2) + math.Pow(b.Value(), 2)*a.Derivative(2, i) - b.Value()*(2*a.Derivative(1, i)*b.Derivative(1, i) + a.Value()*b.Derivative(2, i)))/math.Pow(b.Value(), 3))
    }
  }
  return c
}

func (a *Probability) Pow(k Scalar) Scalar {
  c := NewProbability(1.0, a.N())
  c.order = a.Order()
  c.value = k.Value()*a.LogValue()
  if c.order >= 1 {
    for i := 0; i < a.N(); i++ {
      if k.Order() >= 1 && k.Derivative(1, i) != 0.0 {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*(
          k.Value()*a.Derivative(1, i) + a.Value()*math.Log(a.Value())*k.Derivative(1, i)))
      } else {
        c.SetDerivative(1, i, math.Pow(a.Value(), k.Value()-1)*k.Value()*a.Derivative(1, i))
      }
    }
  }
  if c.order >= 2 {
    for i := 0; i < a.N(); i++ {
      if k.Order() >= 1 && k.Derivative(1, i) != 0.0 {
        c.SetDerivative(2, i,
          math.Pow(a.Value(), k.Value())*(
            (k.Value()-1.0)*k.Value()*math.Pow(a.Derivative(1, i), 2)/math.Pow(a.Value(), 2) +
              (2.0*(1.0 + k.Value()*math.Log(a.Value()))*a.Derivative(1, i)*k.Derivative(1, i) + k.Value()*a.Derivative(2, i))/a.Value() +
              math.Log(a.Value())*(math.Log(a.Value())*math.Pow(k.Derivative(1, i), 2.0) + k.Derivative(2, i))))
      } else {
        c.SetDerivative(2, i, k.Value()*math.Pow(a.Value(), k.Value()-1)*a.Derivative(2, i) + k.Value()*(k.Value()-1)*math.Pow(a.Value(), k.Value()-2)*math.Pow(a.Derivative(1, i), 2))
      }
    }
  }
  return c
}

func (a *Probability) Sqrt() Scalar {
  return a.Pow(NewBareReal(1.0/2.0))
}

/* -------------------------------------------------------------------------- */

func (a *Probability) Sin() Scalar {
  c := NewProbability(math.Sin(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cos(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cos(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Sin(a.Value()))
    }
  }
  return c
}

func (a *Probability) Sinh() Scalar {
  c := NewProbability(math.Sinh(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Cosh(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Cosh(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Sinh(a.Value()))
    }
  }
  return c
}

func (a *Probability) Cos() Scalar {
  c := NewProbability(math.Cos(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, -a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, -a.Derivative(2, i)*math.Sin(a.Value()) - math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (a *Probability) Cosh() Scalar {
  c := NewProbability(math.Cosh(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Sin(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, a.Derivative(2, i)*math.Sin(a.Value()) + math.Pow(a.Derivative(1, i), 2)*math.Cos(a.Value()))
    }
  }
  return c
}

func (a *Probability) Tan() Scalar {
  c := NewProbability(math.Tan(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0+math.Pow(math.Tan(a.Value()), 2)))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0+math.Pow(math.Tan(a.Value()), 2))*(a.Derivative(2, i) + 2*math.Tan(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (a *Probability) Tanh() Scalar {
  c := NewProbability(math.Tanh(a.Value()), a.N())
  c.order = a.Order()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*(1.0-math.Pow(math.Tanh(a.Value()), 2)))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (1.0-math.Pow(math.Tanh(a.Value()), 2))*(a.Derivative(2, i) - 2*math.Tanh(a.Value())*math.Pow(a.Derivative(1, i), 2)))
    }
  }
  return c
}

func (a *Probability) Exp() Scalar {
  c := NewProbability(1.0, a.N())
  c.order = a.Order()
  c.value = a.Value()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)*math.Exp(a.Value()))
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i) + math.Pow(a.Derivative(1, i), 2))*math.Exp(a.Value()))
    }
  }
  return c
}

func (a *Probability) Log() Scalar {
  c := NewReal(1.0, a.N())
  c.order = a.Order()
  c.value = a.LogValue()
  if c.Order() >= 1 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(1, i, a.Derivative(1, i)/a.Value())
    }
  }
  if c.Order() >= 2 {
    for i := 0; i < a.N(); i++ {
      c.SetDerivative(2, i, (a.Derivative(2, i)*a.Value() - a.Derivative(1, i)*a.Derivative(1, i))/(a.Value()*a.Value()))
    }
  }
  return c
}
