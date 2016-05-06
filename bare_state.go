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

type BareState struct {
  value float64
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewBareState(v float64) *BareState {
  return &BareState{v}
}

/* -------------------------------------------------------------------------- */

func (a *BareState) Copy(b Scalar) {
  a.value = b.Value()
}

func (a *BareState) Alloc(n int) {
}

func (c *BareState) AllocForOne(a Scalar) {
}

func (c *BareState) AllocForTwo(a, b Scalar) {
}

/* read access
 * -------------------------------------------------------------------------- */

func (a *BareState) Order() int {
  return 0
}

func (a *BareState) Value() float64 {
  return a.value
}

func (a *BareState) LogValue() float64 {
  return math.Log(a.Value())
}

func (a *BareState) Derivative(i, j int) float64 {
  return 0.0
}

func (a *BareState) N() int {
  return 0
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BareState) Reset() {
  a.value = 0.0
}

func (a *BareState) Set(b Scalar) {
  a.value = b.Value()
}

func (a *BareState) SetValue(v float64) {
  a.value = v
}

func (a *BareState) SetDerivative(i, j int, v float64) {
}

func (a *BareState) SetVariable(i, n, order int) {
}
