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

type BasicState struct {
  value         float64
  derivative [2]float64
  order         int
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewBasicState(v float64) *BasicState {
  s := BasicState{
    value     : v,
    derivative: [2]float64{0, 0},
    order     : 0 }
  return &s
}

/* -------------------------------------------------------------------------- */

func (a *BasicState) Copy(b Scalar) {
  a.order = b.Order()
  a.value = b.Value()
  a.derivative[0] = b.Derivative(1)
  a.derivative[1] = b.Derivative(2)
}

/* field access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Set(v float64) {
  a.value = v
  if a.order == 0 {
    a.derivative[0] = 0
    a.derivative[1] = 0
  } else {
    a.derivative[0] = 1
    a.derivative[1] = 0
  }
}

func (a *BasicState) Order() int {
  return a.order
}

func (a *BasicState) Value() float64 {
  return a.value
}

func (a *BasicState) Derivative(i int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  return a.derivative[i-1]
}

func (a *BasicState) Variable(order int) {
  a.order = order
  a.derivative[0] = 1
  a.derivative[1] = 0
}

func (a *BasicState) Constant() {
  a.order = 0
  a.derivative[0] = 0
  a.derivative[1] = 0
}
