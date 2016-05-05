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

package cholesky

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func cholesky(A Matrix) Matrix {
  n, _ := A.Dims()
  t    := A.ElementType()
  L    := NullMatrix(t, n, n)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s := NewScalar(t, 0.0)
      for k := 0; k < j; k++ {
        s = Add(s, Mul(L.ReferenceAt(i,k), L.ReferenceAt(j,k)))
      }
      b := Sub(A.ReferenceAt(i, j), s)
      if i == j {
        if b.Value() < 0.0 {
          panic("matrix is not positive definite")
        }
        L.Set(Sqrt(b), i, j)
      } else {
        L.Set(Div(b, L.ReferenceAt(j, j)), i, j)
      }
    }
  }
  return L
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) Matrix {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("Cholesky(): Not a square matrix!")
  }
  if rows == 0 {
    panic("Cholesky(): Empty matrix!")
  }
  return cholesky(matrix)
}
