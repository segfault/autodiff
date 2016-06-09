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
import   "errors"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

func cholesky(A Matrix) (Matrix, error) {
  n, _  := A.Dims()
  eType := A.ElementType()
  t     := NewScalar(eType, 0.0)
  s     := NewScalar(eType, 0.0)
  L     := NullDenseMatrix(eType, n, n)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(L.ReferenceAt(i,k), L.ReferenceAt(j,k))
        s.Add(s, t)
      }
      t.Sub(A.ReferenceAt(i, j), s)
      if i == j {
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        L.ReferenceAt(i, j).Sqrt(t)
      } else {
        L.ReferenceAt(i, j).Div(t, L.ReferenceAt(j, j))
      }
    }
  }
  return L, nil
}

func choleskyInSitu(A Matrix) (Matrix, error) {
  n, _  := A.Dims()
  eType := A.ElementType()
  t     := NewScalar(eType, 0.0)
  s     := NewScalar(eType, 0.0)
  Aii   := NewScalar(eType, 0.0)

  for i := 0; i < n; i++ {
    Aii.Copy(A.ReferenceAt(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.Mul(A.ReferenceAt(i,k), A.ReferenceAt(j,k))
        s.Add(s, t)
      }
      if i == j {
        t.Sub(Aii, s)
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        A.ReferenceAt(j, i).Sqrt(t)
      } else {
        t.Sub(A.ReferenceAt(i, j), s)
        A.ReferenceAt(i, j).Div(t, A.ReferenceAt(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.ReferenceAt(j, i)
      A.ReferenceAt(j, i).Copy(r)
      r.Reset()
    }
  }
  return A, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, error) {
  n, m := a.Dims()
  if n != m {
    panic("Cholesky(): Not a square matrix!")
  }
  if n == 0 {
    panic("Cholesky(): Empty matrix!")
  }
  inSitu := false

  for _, arg := range args {
    switch t := arg.(type) {
    case InSitu:
      inSitu = t.Value
    default:
      panic("Cholesky(): Invalid optional argument!")
    }
  }
  if ad, ok := a.(*DenseMatrix); ok {
    t := a.ElementType()
    if t == RealType && inSitu == true {
      return choleskyInSitu_RealDense(ad)
    } else if t == RealType && inSitu == false {
      return cholesky_RealDense(ad)
    } else if t == BareRealType && inSitu == true {
      return choleskyInSitu_BareRealDense(ad)
    } else if t == BareRealType && inSitu == false {
      return cholesky_BareRealDense(ad)
    }
  }
  if inSitu {
    return choleskyInSitu(a)
  } else {
    return cholesky(a)
  }
}
