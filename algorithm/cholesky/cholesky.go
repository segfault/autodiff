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

type InSitu struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

func cholesky(A Matrix) Matrix {
  n, _  := A.Dims()
  eType := A.ElementType()
  t     := NewScalar(eType, 0.0)
  s     := NewScalar(eType, 0.0)
  L     := NullMatrix(eType, n, n)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(L.ReferenceAt2(i,k), L.ReferenceAt2(j,k))
        s.Add(s, t)
      }
      t.Sub(A.ReferenceAt2(i, j), s)
      if i == j {
        if t.Value() < 0.0 {
          panic("matrix is not positive definite")
        }
        L.ReferenceAt2(i, j).Sqrt(t)
      } else {
        L.ReferenceAt2(i, j).Div(t, L.ReferenceAt2(j, j))
      }
    }
  }
  return L
}

func cholesky_insitu(A Matrix) Matrix {
  n, _  := A.Dims()
  eType := A.ElementType()
  t     := NewScalar(eType, 0.0)
  s     := NewScalar(eType, 0.0)
  Aii   := NewScalar(eType, 0.0)

  for i := 0; i < n; i++ {
    Aii.Copy(A.ReferenceAt2(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(A.ReferenceAt2(i,k), A.ReferenceAt2(j,k))
        s.Add(s, t)
      }
      if i == j {
        t.Sub(Aii, s)
        if t.Value() < 0.0 {
          panic("matrix is not positive definite")
        }
        A.ReferenceAt2(j, i).Sqrt(t)
      } else {
        t.Sub(A.ReferenceAt2(i, j), s)
        A.ReferenceAt2(i, j).Div(t, A.ReferenceAt2(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.ReferenceAt2(j, i)
      A.ReferenceAt2(j, i).Copy(r)
      r.Reset()
    }
  }
  return A
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
  inSitu := false

  for _, arg := range args {
    switch a := arg.(type) {
    case InSitu:
      inSitu = a.Value
    default:
      panic("Cholesky(): Invalid optional argument!")
    }
  }
  if inSitu {
    return cholesky_insitu(matrix)
  } else {
    return cholesky(matrix)
  }
}
