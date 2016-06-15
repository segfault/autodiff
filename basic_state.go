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
  Value            float64
  Order            int
  Derivative  [][2]float64
  N                int
}

/* constructors
 * -------------------------------------------------------------------------- */

// Create a new basic state. As an optional argument the number of variables
// for which derivatives are computed may be passed.
func NewBasicState(value float64) *BasicState {
  a := BasicState{}
  a.Value = value
  a.Order = 0
  a.N     = 0
  return &a
}

/* -------------------------------------------------------------------------- */

// Copy the basic state from b. Allocate memory if needed.
func (a *BasicState) Copy(b Scalar) {
  a.Value = b.GetValue()
  a.Order = b.GetOrder()
  a.Alloc(b.GetN())
  for i := 0; i < b.GetN(); i++ {
    a.Derivative[i][0] = b.GetDerivative(1, i)
    a.Derivative[i][1] = b.GetDerivative(2, i)
  }
}

// Allocate memory for derivatives of n variables.
func (a *BasicState) Alloc(n int) {
  if a.N != n {
    a.Derivative = make([][2]float64, n)
    a.N          = n
  }
}

// Allocate memory for the results of mathematical operations on
// the given variables.
func (c *BasicState) AllocForOne(a Scalar) {
  c.Alloc(a.GetN())
  c.Order = a.GetOrder()
}
func (c *BasicState) AllocForTwo(a, b Scalar) {
  c.Alloc(iMax(a.GetN(), b.GetN()))
  c.Order = iMax(a.GetOrder(), b.GetOrder())
}

/* read access
 * -------------------------------------------------------------------------- */

// Indicates the maximal order of derivatives that are computed for this
// variable. `0' means no derivatives, `1' only the first derivative, and
// `2' the first and second derivative.
func (a *BasicState) GetOrder() int {
  return a.Order
}

// Returns the value of the variable.
func (a *BasicState) GetValue() float64 {
  return a.Value
}

// Returns the value of the variable on log scale.
func (a *BasicState) GetLogValue() float64 {
  return math.Log(a.Value)
}

// Returns the ith derivative of the jth variable.
func (a *BasicState) GetDerivative(i, j int) float64 {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  if j < a.N {
    return a.Derivative[j][i-1]
  } else {
    return 0.0
  }
}

// Number of variables for which derivates are stored.
func (a *BasicState) GetN() int {
  return a.N
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BasicState) Reset() {
  a.Value = 0.0
  for i := 0; i < a.N; i++ {
    a.Derivative[i][0] = 0.0
    a.Derivative[i][1] = 0.0
  }
}

func (a *BasicState) ResetDerivatives() {
  for i := 0; i < a.N; i++ {
    a.Derivative[i][0] = 0.0
    a.Derivative[i][1] = 0.0
  }
}

// Set the state to b. This includes the value and all derivatives.
func (a *BasicState) Set(b Scalar) {
  a.Copy(b)
}

// Set only the value of the variable.
func (a *BasicState) SetValue(v float64) {
  a.Value = v
  // for i := 0; i < len(a.Derivative); i++ {
  //   if a.Derivative[i][0] != 0.0 {
  //     a.Derivative[i][0] = 1
  //   }
  //   a.Derivative[i][1] = 0
  // }
}

// Set the ith derivative of the jth variable to v.
func (a *BasicState) SetDerivative(i, j int, v float64) {
  if i != 1 && i != 2 {
    panic("Invalid order!")
  }
  a.Derivative[j][i-1] = v
}

// Allocate memory for n variables and set the derivative
// of the ith variable to 1 (initial value).
func (a *BasicState) SetVariable(i, n, order int) {
  a.Derivative = make([][2]float64, n)
  a.N          = n
  a.Order      = order
  if order > 0 {
    a.Derivative[i][0] = 1
  }
}
