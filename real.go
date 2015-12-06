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

import "fmt"
import "reflect"

/* -------------------------------------------------------------------------- */

type Real struct {
  value         float64
  derivative [2]float64
  order         int
}

/* register scalar type
 * -------------------------------------------------------------------------- */

var RealType ScalarType = NewReal(0.0).Type()

func init() {
  f := func(value float64) Scalar { return NewReal(value) }
  RegisterScalar(RealType, f)
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewReal(v float64) *Real {
  s := Real{
    value     : v,
    derivative: [2]float64{0, 0},
    order     : 0 }
  return &s
}

/* -------------------------------------------------------------------------- */

func (a *Real) Clone() Scalar {
  r := NewReal(a.value)
  r.order         = a.order
  r.derivative[0] = a.derivative[0]
  r.derivative[1] = a.derivative[1]
  return r
}

func (a *Real) Copy(b Scalar) {
  a.order = b.Order()
  a.value = b.Value()
  a.derivative[0] = b.Derivative(1)
  a.derivative[1] = b.Derivative(2)
}

/* field access
 * -------------------------------------------------------------------------- */

func (a *Real) Set(v float64) {
  a.value = v
  a.derivative[0] = 0.0
  a.derivative[1] = 0.0
}

func (a *Real) Order() int {
  return a.order
}

func (a *Real) Value() float64 {
  return a.value
}

func (a *Real) Derivative(i int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  return a.derivative[i-1]
}

func (a *Real) Variable(order int) {
  a.order = order
  a.derivative[0] = 1
  a.derivative[1] = 0
}

func (a *Real) Constant() {
  a.order = 0
  a.derivative[0] = 0
  a.derivative[1] = 0
}

func (a *Real) Type() ScalarType {
  return reflect.TypeOf(a)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a *Real) String() string {
  switch a.order {
  case 0:
    return fmt.Sprintf("<%e>", a.Value())
  case 1:
    return fmt.Sprintf("<%e,%e>", a.Value(), a.Derivative(1))
  case 2:
    return fmt.Sprintf("<%e,%e,%e>", a.Value(), a.Derivative(1), a.Derivative(2))
  default:
    panic("Invalid order!")
  }
}
