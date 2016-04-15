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

type BasicState struct {
  value            float64
  order            int
  derivative  [][2]float64
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewBasicState(value float64, args ...int) *BasicState {
  // number of variables for the gradient
  n := 0
  // get optional gradient length
  if len(args) >= 1 {
    n = args[0]
  }
  a := BasicState{}
  a.value = value
  a.order = 0
  a.Alloc(n)
  return &a
}

/* -------------------------------------------------------------------------- */

func (a *BasicState) Copy(b Scalar) {
  a.value = b.Value()
  a.order = b.Order()
  a.Alloc(b.N())
  for i := 0; i < b.N(); i++ {
    a.derivative[i][0] = b.Derivative(1, i)
    a.derivative[i][1] = b.Derivative(2, i)
  }
}

func (a *BasicState) Alloc(n int) {
  if len(a.derivative) != n {
    a.derivative = make([][2]float64, n)
  }
}

func (c *BasicState) AllocFor(args ...Scalar) {
  switch len(args) {
  case 1:
    c.Alloc(args[0].N())
    c.order = args[0].Order()
  case 2:
    c.Alloc(iMax(args[0].N(), args[1].N()))
    c.order = iMax(args[0].Order(), args[1].Order())
  }
}

/* read access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Order() int {
  return a.order
}

func (a *BasicState) Value() float64 {
  return a.value
}

func (a *BasicState) LogValue() float64 {
  return math.Log(a.value)
}

func (a *BasicState) Derivative(i, j int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  if len(a.derivative) > 0 {
    return a.derivative[j][i-1]
  } else {
    return 0.0
  }
}

func (a *BasicState) N() int {
  return len(a.derivative)
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Set(b Scalar) {
  a.Copy(b)
}

func (a *BasicState) SetValue(v float64) {
  a.value = v
  for i := 0; i < len(a.derivative); i++ {
    if a.order == 0 {
      a.derivative[i][0] = 0
    } else {
      a.derivative[i][0] = 1
    }
    a.derivative[i][1] = 0
  }
}

func (a *BasicState) SetDerivative(i, j int, v float64) {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  a.derivative[j][i-1] = v
}

func (a *BasicState) SetVariable(i, n, order int) {
  a.derivative = make([][2]float64, n)
  a.order = order
  if order > 0 {
    a.derivative[i][0] = 1
  }
}
