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

// This is the basic state used by real and probability scalars.
type BasicState struct {
  value            float64
  order            int
  derivative  [][2]float64
  n                int
}

/* constructors
 * -------------------------------------------------------------------------- */

// Create a new basic state. As an optional argument the number of variables
// for which derivatives are computed may be passed.
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

// Copy the basic state from b. Allocate memory if needed.
func (a *BasicState) Copy(b Scalar) {
  a.value = b.Value()
  a.order = b.Order()
  a.Alloc(b.N())
  for i := 0; i < b.N(); i++ {
    a.derivative[i][0] = b.Derivative(1, i)
    a.derivative[i][1] = b.Derivative(2, i)
  }
}

// Allocate memory for derivatives of n variables.
func (a *BasicState) Alloc(n int) {
  if a.n != n {
    a.derivative = make([][2]float64, n)
    a.n          = n
  }
}

// Allocate memory for the results of mathematical operations on
// the given variables.
func (c *BasicState) AllocForOne(a Scalar) {
  c.Alloc(a.N())
  c.order = a.Order()
}
func (c *BasicState) AllocForTwo(a, b Scalar) {
  c.Alloc(iMax(a.N(), b.N()))
  c.order = iMax(a.Order(), b.Order())
}

/* read access
 * -------------------------------------------------------------------------- */

// Indicates the maximal order of derivatives that are computed for this
// variable. `0' means no derivatives, `1' only the first derivative, and
// `2' the first and second derivative.
func (a *BasicState) Order() int {
  return a.order
}

// Returns the value of the variable.
func (a *BasicState) Value() float64 {
  return a.value
}

// Returns the value of the variable on log scale.
func (a *BasicState) LogValue() float64 {
  return math.Log(a.value)
}

// Returns the ith derivative of the jth variable.
func (a *BasicState) Derivative(i, j int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  if a.n > 0 {
    return a.derivative[j][i-1]
  } else {
    return 0.0
  }
}

// Number of variables for which derivates are stored.
func (a *BasicState) N() int {
  return a.n
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Reset() {
  a.value = 0.0
  for i := 0; i < a.n; i++ {
    a.derivative[i][0] = 0.0
    a.derivative[i][1] = 0.0
  }
}

// Set the state to b. This includes the value and all derivatives.
func (a *BasicState) Set(b Scalar) {
  a.Copy(b)
}

// Set only the value of the variable.
func (a *BasicState) SetValue(v float64) {
  a.value = v
  // for i := 0; i < len(a.derivative); i++ {
  //   if a.derivative[i][0] != 0.0 {
  //     a.derivative[i][0] = 1
  //   }
  //   a.derivative[i][1] = 0
  // }
}

// Set the ith derivative of the jth variable to v.
func (a *BasicState) SetDerivative(i, j int, v float64) {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  a.derivative[j][i-1] = v
}

// Allocate memory for n variables and set the derivative
// of the ith variable to 1 (initial value).
func (a *BasicState) SetVariable(i, n, order int) {
  a.derivative = make([][2]float64, n)
  a.n          = n
  a.order      = order
  if order > 0 {
    a.derivative[i][0] = 1
  }
}
