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

func cholesky_RealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewReal(0.0)
  s     := NewReal(0.0)
  L     := NullMatrix(RealType, n, n).(*DenseMatrix)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.RealMul(L.RealReferenceAt2(i,k), L.RealReferenceAt2(j,k))
        s.RealAdd(s, t)
      }
      t.RealSub(A.RealReferenceAt2(i, j), s)
      if i == j {
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        L.RealReferenceAt2(i, j).RealSqrt(t)
      } else {
        L.RealReferenceAt2(i, j).RealDiv(t, L.RealReferenceAt2(j, j))
      }
    }
  }
  return L, nil
}

func choleskyInSitu_RealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewReal(0.0)
  s     := NewReal(0.0)
  Aii   := NewReal(0.0)

  for i := 0; i < n; i++ {
    Aii.Copy(A.RealReferenceAt2(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.RealMul(A.RealReferenceAt2(i,k), A.RealReferenceAt2(j,k))
        s.RealAdd(s, t)
      }
      if i == j {
        t.RealSub(Aii, s)
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        A.RealReferenceAt2(j, i).RealSqrt(t)
      } else {
        t.RealSub(A.RealReferenceAt2(i, j), s)
        A.RealReferenceAt2(i, j).RealDiv(t, A.RealReferenceAt2(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.RealReferenceAt2(j, i)
      A.RealReferenceAt2(j, i).Copy(r)
      r.Reset()
    }
  }
  return A, nil
}

/* -------------------------------------------------------------------------- */

func cholesky_BareRealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewBareReal(0.0)
  s     := NewBareReal(0.0)
  L     := NullMatrix(BareRealType, n, n).(*DenseMatrix)
 
  for i := 0; i < n; i++ {
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.BareRealMul(L.BareRealReferenceAt2(i,k), L.BareRealReferenceAt2(j,k))
        s.BareRealAdd(s, t)
      }
      t.BareRealSub(A.BareRealReferenceAt2(i, j), s)
      if i == j {
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        L.BareRealReferenceAt2(i, j).BareRealSqrt(t)
      } else {
        L.BareRealReferenceAt2(i, j).BareRealDiv(t, L.BareRealReferenceAt2(j, j))
      }
    }
  }
  return L, nil
}

func choleskyInSitu_BareRealDense(A *DenseMatrix) (*DenseMatrix, error) {
  n, _  := A.Dims()
  t     := NewBareReal(0.0)
  s     := NewBareReal(0.0)
  Aii   := NewBareReal(0.0)

  for i := 0; i < n; i++ {
    Aii.Copy(A.BareRealReferenceAt2(i,i))
    for j := 0; j < (i+1); j++ {
      s.Reset()
      for k := 0; k < j; k++ {
        t.BareRealMul(A.BareRealReferenceAt2(i,k), A.BareRealReferenceAt2(j,k))
        s.BareRealAdd(s, t)
      }
      if i == j {
        t.BareRealSub(Aii, s)
        if t.Value() < 0.0 {
          return nil, errors.New("matrix is not positive definite")
        }
        A.BareRealReferenceAt2(j, i).BareRealSqrt(t)
      } else {
        t.BareRealSub(A.BareRealReferenceAt2(i, j), s)
        A.BareRealReferenceAt2(i, j).BareRealDiv(t, A.BareRealReferenceAt2(j, j))
      }
    }
  }
  // move elements from upper triangular matrix
  for i := 0; i < n; i++ {
    for j := 0; j < i; j++ {
      r := A.BareRealReferenceAt2(j, i)
      A.BareRealReferenceAt2(j, i).Copy(r)
      r.Reset()
    }
  }
  return A, nil
}
