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

func (a *BareReal) Equals(b Scalar) bool {
  epsilon := 1e-12
  return math.Abs(a.Value() - b.Value()) < epsilon
}

func (a *BareReal) Greater(b Scalar) bool {
  return a.Value() > b.Value()
}

func (a *BareReal) Smaller(b Scalar) bool {
  return a.Value() < b.Value()
}

func (a *BareReal) Neg() Scalar {
  return NewBareReal(-a.Value())
}

func (a *BareReal) Add(b Scalar) Scalar {
  return NewBareReal(a.Value() + b.Value())
}

func (a *BareReal) Sub(b Scalar) Scalar {
  return NewBareReal(a.Value() - b.Value())
}

func (a *BareReal) Mul(b Scalar) Scalar {
  return NewBareReal(a.Value()*b.Value())
}

func (a *BareReal) Div(b Scalar) Scalar {
  return NewBareReal(a.Value()/b.Value())
}

func (a *BareReal) Pow(k float64) Scalar {
  return NewBareReal(math.Pow(a.Value(), k))
}

func (a *BareReal) Sqrt() Scalar {
  return a.Pow(1.0/2.0)
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Sin() Scalar {
  return NewBareReal(math.Sin(a.Value()))
}

func (a *BareReal) Sinh() Scalar {
  return NewBareReal(math.Sinh(a.Value()))
}

func (a *BareReal) Cos() Scalar {
  return NewBareReal(math.Cos(a.Value()))
}

func (a *BareReal) Cosh() Scalar {
  return NewBareReal(math.Cosh(a.Value()))
}

func (a *BareReal) Tan() Scalar {
  return NewBareReal(math.Tan(a.Value()))
}

func (a *BareReal) Tanh() Scalar {
  return NewBareReal(math.Tanh(a.Value()))
}

func (a *BareReal) Exp() Scalar {
  return NewBareReal(math.Exp(a.Value()))
}

func (a *BareReal) Log() Scalar {
  return NewBareReal(math.Log(a.Value()))
}
