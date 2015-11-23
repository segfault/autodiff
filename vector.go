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

func NewVector(values []float64) Vector {
  v := make(Vector, len(values))

  for i, _ := range values {
    v[i] = NewConstant(values[i])
  }
  return v
}

func MakeVector(length int) Vector {
  v := make(Vector, length)

  for i := 0; i < length; i++ {
    v[i] = NewConstant(0.0)
  }
  return v
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
  return v[i].Derivative(i)
}

func (v Vector) Variable(i, order int) Vector {
  v[i].Variable(order)
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

func (v *Vector) CopyFrom(w Vector) {
  if len(*v) != len(w) {
    panic("CopyFrom(): Vector dimensions do not match!")
  }
  for i := 0; i < len(w); i++ {
    (*v)[i] = w[i]
  }
}

/* -------------------------------------------------------------------------- */

func VAdd(a, b Vector) Vector {
  if len(a) != len(b) {
    panic("Vector dimensions do not match!")
  }
  r := MakeVector(len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Add(a[i], b[i])
  }
  return r
}

func VSub(a, b Vector) Vector {
  if len(a) != len(b) {
    panic("Vector dimensions do not match!")
  }
  r := MakeVector(len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Sub(a[i], b[i])
  }
  return r
}

func VNorm(a Vector) Scalar {
  r := NewConstant(0.0)
  for i := 0; i < len(a); i++ {
    r = Add(r, Pow(a[i], 2))
  }
  return r
}
