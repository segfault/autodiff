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

import .     "github.com/pbenner/autodiff/scalar"
//import gonum "github.com/gonum/matrix"

/* -------------------------------------------------------------------------- */

type Matrix struct {
  values Vector
  n      int
  m      int
  t      bool
}

func NewMatrix(values []float64, n, m int) Matrix {
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

func (matrix Matrix) Dims() (int, int) {
  return matrix.n, matrix.m
}

func (matrix Matrix) At(i, j int) float64 {
  var k int
  if matrix.t {
    k = j*matrix.n + i
  } else {
    k = i*matrix.m + j
  }
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k].Value()
}

func (matrix Matrix) T() Matrix {
  return Matrix{
    values:  matrix.values,
    n     :  matrix.m,
    m     :  matrix.n,
    t     : !matrix.t}
}
