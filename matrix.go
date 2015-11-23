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

type Matrix struct {
  values Vector
  rows   int
  cols   int
  t      bool
}

func NewMatrix(rows, cols int, values []float64) Matrix {
  tmp := MakeVector(rows*cols)
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      tmp[i] = NewConstant(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      tmp[i] = NewConstant(values[i])
    }
  } else {
    panic("Matrix dimension does not fit input values!")
  }
  return Matrix{tmp, rows, cols, false}
}

func MakeMatrix(rows, cols int) Matrix {
  return Matrix{MakeVector(rows*cols), rows, cols, false}
}

func IdentityMatrix(dim int) Matrix {
  matrix := MakeMatrix(dim, dim)
  for i := 0; i < dim; i++ {
    matrix.Set(i, i, 1)
  }
  return matrix
}

/* -------------------------------------------------------------------------- */

func (matrix Matrix) Clone() Matrix {
  return Matrix{
    values: matrix.values.Clone(),
    rows  : matrix.rows,
    cols  : matrix.cols,
    t     : matrix.t}
}

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

func (matrix Matrix) At(i, j int) float64 {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k].Value()
}

func (matrix Matrix) Values() Vector {
  return matrix.values
}

func (matrix *Matrix) SetValues(v Vector) {
  matrix.values = v
}

func (matrix Matrix) ScalarAt(i, j int) Scalar {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k]
}

func (matrix Matrix) Set(i, j int, v float64) {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  matrix.values[k].value = v
}

func (matrix Matrix) ScalarSet(i, j int, s Scalar) {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  matrix.values[k] = s
}

func (matrix Matrix) T() Matrix {
  return Matrix{
    values:  matrix.values,
    rows  :  matrix.cols,
    cols  :  matrix.rows,
    t     : !matrix.t}
}

/* -------------------------------------------------------------------------- */

func MAdd(a, b Matrix) Matrix {
  if a.rows != b.rows || a.cols != b.cols {
    panic("Matrix dimensions do not match!")
  }
  rows := a.rows
  cols := a.cols
  r := MakeMatrix(rows, cols)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      r.ScalarSet(i, j, Add(a.ScalarAt(i, j), b.ScalarAt(i, j)))
    }
  }
  return r
}

func MSub(a, b Matrix) Matrix {
  if a.rows != b.rows || a.cols != b.cols {
    panic("Matrix dimensions do not match!")
  }
  rows := a.rows
  cols := a.cols
  r := MakeMatrix(rows, cols)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      r.ScalarSet(i, j, Sub(a.ScalarAt(i, j), b.ScalarAt(i, j)))
    }
  }
  return r
}

func MMul(a, b Matrix) Matrix {
  if a.cols != b.rows {
    panic("Matrix dimensions do not match!")
  }
  r := MakeMatrix(a.rows, b.cols)
  for i := 0; i < r.rows; i++ {
    for j := 0; j < r.cols; j++ {
      for n := 0; n < a.cols; n++ {
        r.ScalarSet(i, j, Add(r.ScalarAt(i, j), Mul(a.ScalarAt(i, n), b.ScalarAt(n, j))))
      }
    }
  }
  return r
}

func MxV(a Matrix, b Vector) Vector {
  if a.cols != len(b) {
    panic("Matrix/Vector dimensions do not match!")
  }
  r := MakeVector(a.rows)
  for i := 0; i < len(r); i++ {
    for n := 0; n < a.cols; n++ {
      r[i] = Add(r[i], Mul(a.ScalarAt(i, n), b[n]))
    }
  }
  return r
}

func VxM(a Vector, b Matrix) Vector {
  if len(a) != b.rows {
    panic("Matrix/Vector dimensions do not match!")
  }
  r := MakeVector(b.cols)
  for i := 0; i < len(r); i++ {
    for n := 0; n < b.rows; n++ {
      r[i] = Add(r[i], Mul(a[n], b.ScalarAt(n, i)))
    }
  }
  return r
}

func MTrace(matrix Matrix) Scalar {
  t := NewConstant(0.0)
  if matrix.rows != matrix.cols {
    panic("Not a square matrix!")
  }
  for i := 0; i < matrix.rows; i++ {
    t = Add(t, matrix.ScalarAt(i,i))
  }
  return t
}

func MNorm(matrix Matrix) Scalar {
  s := NewConstant(0.0)
  for _, v := range matrix.values {
    s = Add(s, Pow(v, 2.0))
  }
  return s
}

func MInverse(matrix Matrix) Matrix {
  if matrix.rows != matrix.cols {
    panic("Not a square matrix!")
  }
  I := IdentityMatrix(matrix.rows)
  r := matrix.Clone()
  // objective function
  f := func(x Vector) Scalar {
    r.SetValues(x)
    s := MNorm(MSub(MMul(matrix, r), I))
    return s
  }
  x, _ := Rprop(f, r.Values(), 0.01, 1e-12, 0.1)
  r.SetValues(x)
  return r
}

/* -------------------------------------------------------------------------- */

func Jacobian(f func(Vector) Vector, x Vector) Matrix {
  n := 0
  m := len(x)
  r := Matrix{}
  for j := 0; j < m; j++ {
    x[j].Constant()
  }
  for j := 0; j < m; j++ {
    // differentiate with respect to the ith variable
    x[j].Variable(1)
    y := f(x)
    if j == 0 {
      n = len(y)
      r = MakeMatrix(n, m)
    }
    if n != len(y) {
      panic("Jacobian(): dimensions do not match!")
    }
    // copy derivatives
    for i := 0; i < n; i++ {
      r.Set(i, j, y[i].Derivative(1))
    }
    x[j].Constant()
  }
  return r
}
