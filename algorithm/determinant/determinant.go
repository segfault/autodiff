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

package determinant

/* -------------------------------------------------------------------------- */

//import   "fmt"
//import   "math"
import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/cholesky"

/* -------------------------------------------------------------------------- */

type PositiveDefinite struct {
  Value bool
}

type LogScale struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

func determinantNaive(a Matrix) Scalar {
  n, _ := a.Dims()
  t1   := ZeroScalar(a.ElementType())
  t2   := ZeroScalar(a.ElementType())
  det  := ZeroScalar(a.ElementType())

  if (n < 1) {
    /* nothing to do */
  } else if n == 1 {
    det.Copy(a.ReferenceAt2(0, 0))
  } else if n == 2 {
    t1.Mul(a.ReferenceAt2(0, 0), a.ReferenceAt2(1, 1))
    t2.Mul(a.ReferenceAt2(1, 0), a.ReferenceAt2(0, 1))
    det.Sub(t1, t2)
  } else {
    m := NullMatrix(a.ElementType(), n-1, n-1)
    for j1 := 0; j1 < n; j1++ {
      for i := 1; i < n; i++ {
        j2 := 0
        for j := 0; j < n; j++ {
          if j == j1 {
            continue
          }
          m.ReferenceAt2(i-1, j2).Copy(a.ReferenceAt2(i, j))
          j2++;
        }
      }
      if j1 % 2 == 0 {
        t1.Mul(a.ReferenceAt2(0, j1), determinantNaive(m))
        det.Add(det, t1)
      } else {
        t1.Mul(a.ReferenceAt2(0, j1), determinantNaive(m))
        det.Sub(det, t1)
      }
    }
  }
  return det
}

func determinantPD(a Matrix, logScale bool) Scalar {
  n, m := a.Dims()
  r := ZeroScalar(a.ElementType())
  t := ZeroScalar(a.ElementType())
  if n != m {
    panic("Matrix is not a square matrix!")
  }
  L := cholesky.Run(a)
  if logScale {
    r.SetValue(0.0)
    for i := 0; i < n; i++ {
      t.Log(L.ReferenceAt2(i, i))
      r.Add(r, t)
    }
    r.Add(r, r)
  } else {
    r.SetValue(1.0)
    for i := 0; i < n; i++ {
      r.Mul(r, L.ReferenceAt2(i, i))
    }
    r.Mul(r, r)
  }
  return r
}

func determinant(a Matrix, positiveDefinite, logScale bool) Scalar {
  if positiveDefinite {
    return determinantPD(a, logScale)
  } else {
    return determinantNaive(a)
  }
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) Scalar {
  positiveDefinite := false
  logScale := false

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case PositiveDefinite:
      positiveDefinite = a.Value
    case LogScale:
      logScale = a.Value
    default:
      panic("Determinant(): Invalid optional argument!")
    }
  }
  if logScale && !positiveDefinite {
    panic("Parameter LogScale is valid only for positive definite matrices!")
  }
  return determinant(a, positiveDefinite, logScale)
}
