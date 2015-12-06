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
import "reflect"

/* vector type declaration
 * -------------------------------------------------------------------------- */

type Vector []Scalar

/* constructors
 * -------------------------------------------------------------------------- */

func NewVector(t ScalarType, values []float64) Vector {
  v := make(Vector, len(values))

  for i, _ := range values {
    v[i] = NewScalar(t, values[i])
  }
  return v
}

func NullVector(t ScalarType, length int) Vector {
  v := make(Vector, length)

  for i := 0; i < length; i++ {
    v[i] = NewScalar(t, 0.0)
  }
  return v
}

func (v Vector) Clone() Vector {
  result := make(Vector, len(v))

  for i, _ := range v {
    result[i] = v[i].Clone()
  }
  return result
}

func (v Vector) Copy(w Vector) {
  if len(v) != len(w) {
    panic("CopyFrom(): Vector dimensions do not match!")
  }
  for i := 0; i < len(w); i++ {
    v[i].Copy(w[i])
  }
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */

func (v Vector) At(args ...int) Scalar {
  return v[args[0]].Clone()
}

func (v Vector) Set(value Scalar, args ...int) {
  v[args[0]].Copy(value)
}

func (v Vector) ElementType() ScalarType {
  if len(v) > 0 {
    return reflect.TypeOf(v[0])
  }
  return nil
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (v Vector) Matrix(n, m int) Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  return Matrix{v, n, m, false}

}

func (v Vector) Slice() []float64 {
  s := make([]float64, len(v))
  for i, _ := range v {
    s[i] = v[i].Value()
  }
  return s
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
