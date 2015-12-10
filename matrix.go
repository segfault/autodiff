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

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type Matrix struct {
  values Vector
  rows   int
  cols   int
  t      bool
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewMatrix(t ScalarType, rows, cols int, values []float64) Matrix {
  tmp := NullVector(t, rows*cols)
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      tmp[i] = NewScalar(t, values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      tmp[i] = NewScalar(t, values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  return Matrix{tmp, rows, cols, false}
}

func NullMatrix(t ScalarType, rows, cols int) Matrix {
  return Matrix{NullVector(t, rows*cols), rows, cols, false}
}

/* copy and cloning
 * -------------------------------------------------------------------------- */

func (matrix Matrix) Clone() Matrix {
  return Matrix{
    values: matrix.values.Clone(),
    rows  : matrix.rows,
    cols  : matrix.cols,
    t     : matrix.t}
}

func (m1 Matrix) Copy(m2 Matrix) {
  if m1.rows != m2.rows || m1.cols != m2.cols {
    panic("Copy(): Matrix dimension does not match!")
  }
  m1.values.Copy(m2.values)
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func IdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.Set(NewScalar(t, 1), i, i)
  }
  return matrix
}

/* field access
 * -------------------------------------------------------------------------- */

func (matrix Matrix) index(i, j int) int {
  var k int
  if matrix.t {
    k = j*matrix.rows + i
  } else {
    k = i*matrix.cols + j
  }
  return k
}

func (matrix Matrix) Dims() (int, int) {
  return matrix.rows, matrix.cols
}

func (matrix Matrix) Values() Vector {
  return matrix.values
}

func (matrix *Matrix) SetValues(v Vector) {
  matrix.values = v
}

func (matrix *Matrix) Row(i int) Vector {
  n := matrix.index(i, 0)
  m := matrix.index(i, matrix.cols)
  return matrix.values[n:m]
}

func (matrix *Matrix) Col(j int) Vector {
  v := NilVector(matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    v[i] = matrix.values[matrix.index(i, j)]
  }
  return v
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */

func (matrix Matrix) At(args ...int) Scalar {
  i := args[0]
  j := args[1]
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  // to avoid confusion we clone the value before
  // returning a reference to it
  //
  // take for instane:
  // c := m.At(0, 0)
  // m.Set(1, 0, 0)
  // which would alter the value of c!
  return matrix.values[k].Clone()
}

func (matrix Matrix) Set(s Scalar, args ...int) {
  i := args[0]
  j := args[1]
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("Set(): Index out of bounds!")
  }
  matrix.values[k].Copy(s)
}

func (matrix Matrix) ElementType() ScalarType {
  if matrix.rows > 0 && matrix.cols > 0 {
    return reflect.TypeOf(matrix.values[0])
  }
  return nil
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (m Matrix) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i := 0; i < m.rows; i++ {
    if i != 0 {
      buffer.WriteString(",\n ")
    }
    buffer.WriteString("[")
    for j := 0; j < m.cols; j++ {
      if j != 0 {
        buffer.WriteString(", ")
      }
      buffer.WriteString(m.At(i,j).String())
    }
    buffer.WriteString("]")
  }
  buffer.WriteString("]")

  return buffer.String()
}
