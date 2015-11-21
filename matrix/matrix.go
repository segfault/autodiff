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

package matrix

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff/scalar"

/* -------------------------------------------------------------------------- */

type Matrix struct {
  values Vector
  n      int
  m      int
  t      bool
}

func NewMatrix(n, m int, values []float64) Matrix {
  tmp := MakeVector(n*m)
  if len(values) == 1 {
    for i := 0; i < n*m; i++ {
      tmp[i] = NewScalar(values[0])
    }
  } else if len(values) == n*m {
    for i := 0; i < n*m; i++ {
      tmp[i] = NewScalar(values[i])
    }
  } else {
    panic("Matrix dimension does not fit input values!")
  }
  return Matrix{tmp, n, m, false}
}

func MakeMatrix(n, m int) Matrix {
  return Matrix{MakeVector(n*m), n, m, false}
}

/* -------------------------------------------------------------------------- */

func (matrix Matrix) index(i, j int) int {
  var k int
  if matrix.t {
    k = j*matrix.n + i
  } else {
    k = i*matrix.m + j
  }
  return k
}

func (matrix Matrix) Dims() (int, int) {
  return matrix.n, matrix.m
}

func (matrix Matrix) At(i, j int) float64 {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k].Value()
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
  matrix.values[k] = NewScalar(v)
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
    n     :  matrix.m,
    m     :  matrix.n,
    t     : !matrix.t}
}

/* -------------------------------------------------------------------------- */

func MMul(a, b Matrix) Matrix {
  if a.m != b.n {
    panic("Matrix dimensions do not match!")
  }
  r := NewMatrix(a.n, b.m, []float64{0.0})
  for i := 0; i < r.n; i++ {
    for j := 0; j < r.m; j++ {
      for n := 0; n < a.m; n++ {
        r.ScalarSet(i, j, Add(r.ScalarAt(i, j), Mul(a.ScalarAt(i, n), b.ScalarAt(n, j))))
      }
    }
  }
  return r
}

func Trace(matrix Matrix) Scalar {
  t := NewScalar(0.0)
  if matrix.n != matrix.m {
    panic("Not a square matrix!")
  }
  for i := 0; i < matrix.n; i++ {
    t = Add(t, matrix.ScalarAt(i,i))
  }
  return t
}
