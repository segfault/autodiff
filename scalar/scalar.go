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

package scalar

/* -------------------------------------------------------------------------- */

import "fmt"

/* -------------------------------------------------------------------------- */

type Scalar struct {
  value      float64
  derivative float64
}

func NewScalar(v float64) *Scalar {
  s := new(Scalar)
  s.value      = v
  s.derivative = 0
  return s
}

func (a *Scalar) Value() float64 {
  return a.value
}

func (a *Scalar) Derivative() float64 {
  return a.derivative
}

func (a *Scalar) Differentiate() {
  a.derivative = 1
}

func (a *Scalar) Reset() {
  a.derivative = 0
}

func (a *Scalar) Assign(v float64) {
  a.value      = v
  a.derivative = 0
}

func (a Scalar) String() string {
  return fmt.Sprintf("<%f,%f>", a.Value(), a.Derivative())
}
