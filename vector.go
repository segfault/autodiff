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

import "bytes"

/* -------------------------------------------------------------------------- */

type Vector []Scalar

func NewVector(w []float64) Vector {
  v := make(Vector, len(w))

  for i, _ := range w {
    v[i] = NewScalar(w[i])
  }
  return v
}

func MakeVector(n int) Vector {
  return make(Vector, n)
}

/* -------------------------------------------------------------------------- */

func (v Vector) Clone() Vector {
  result := make(Vector, len(v))
  copy(result, v)
  return result
}

func (v Vector) Value(i int) float64 {
  return v[i].Value()
}

func (v Vector) Derivative(i int) float64 {
  return v[i].Derivative()
}

func (v Vector) Variable(i int) Vector {
  v[i].Variable()
  return v
}

func (v Vector) Constant(i int) Vector {
  v[i].Constant()
  return v
}

func (v Vector) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(v[i].String())
  }
  buffer.WriteString("]")

  return buffer.String()
}