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

/* -------------------------------------------------------------------------- */

type Scalar struct {
  value         float64
  derivative [2]float64
  order         int
}

func NewScalar(v float64, order int) Scalar {
  s := Scalar{
    value     : v,
    derivative: [2]float64{0, 0},
    order     : order }
  return s
}

func NewConstant(v float64) Scalar {
  s := Scalar{
    value     : v,
    derivative: [2]float64{0, 0},
    order     : 0 }
  return s
}

func NewVariable(v float64, order int) Scalar {
  s := Scalar{
    value     : v,
    derivative: [2]float64{0, 0},
    order     : order }
  s.derivative[0] = 1
  s.derivative[1] = 0
  return s
}

/* -------------------------------------------------------------------------- */

func (a Scalar) Order() int {
  return a.order
}

func (a Scalar) Value() float64 {
  return a.value
}

func (a Scalar) Derivative(i int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  return a.derivative[i-1]
}

func (a *Scalar) Variable(order int) *Scalar {
  a.order = order
  a.derivative[0] = 1
  a.derivative[1] = 0
  return a
}

func (a *Scalar) Constant() *Scalar {
  a.order = 0
  a.derivative[0] = 0
  a.derivative[1] = 0
  return a
}

func (a *Scalar) String() string {
  switch a.order {
  case 0:
    return fmt.Sprintf("<%f>", a.Value())
  case 1:
    return fmt.Sprintf("<%f,%f>", a.Value(), a.Derivative(1))
  case 2:
    return fmt.Sprintf("<%f,%f,%f>", a.Value(), a.Derivative(1), a.Derivative(2))
  default:
    panic("Invalid order!")
  }
}
