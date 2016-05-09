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

package gaussJordan

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func gaussJordan_bare_real_dense(a, x *DenseMatrix, b Vector, submatrix []bool, epsilon float64) {
  t := NewBareReal(0.0)
  c := NewBareReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // permutation of the rows
  p := make([]int, n)
  for i := 0; i < n; i++ {
    p[i] = i
  }
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    panic("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    panic("GaussJordan(): b has invalid dimension!")
  }
  // loop over columns
  for i := 0; i < n; i++ {
    if !submatrix[i] {
      continue
    }
    // check if matrix is singular
    if math.Abs(a.BareRealReferenceAt2(p[i], i).Value()) < epsilon {
      panic("GaussJordan(): matrix is singular!")
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.BareRealReferenceAt2(p[j], i).Value()) > math.Abs(a.BareRealReferenceAt2(p[maxrow], i).Value()) {
        maxrow = j
      }
    }
    // swap rows
    p[i], p[maxrow] = p[maxrow], p[i]
    // eliminate column i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      // c = a[j, i] / a[i, i]
      c.BareRealDiv(a.BareRealReferenceAt2(p[j], i), a.BareRealReferenceAt2(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.BareRealMul(a.BareRealReferenceAt2(p[i], k), c)
        a.BareRealReferenceAt2(p[j], k).BareRealSub(a.BareRealReferenceAt2(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.BareRealMul(x.BareRealReferenceAt2(p[i], k), c)
        x.BareRealReferenceAt2(p[j], k).BareRealSub(x.BareRealReferenceAt2(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.BareRealMul(b[p[i]].(*BareReal), c)
      b[p[j]].(*BareReal).BareRealSub(b[p[j]].(*BareReal), t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealReferenceAt2(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealReferenceAt2(p[j], i), b[p[i]].(*BareReal))
      t.BareRealDiv(t, c)
      b[p[j]].(*BareReal).BareRealSub(b[p[j]].(*BareReal), t)
      if math.IsNaN(b[p[j]].(*BareReal).Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt2(p[j], i), x.BareRealReferenceAt2(p[i], k))
        t.BareRealDiv(t, c)
        x.BareRealReferenceAt2(p[j], k).BareRealSub(x.BareRealReferenceAt2(p[j], k), t)
        if math.IsNaN(x.BareRealReferenceAt2(p[j], k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt2(p[j], i), a.BareRealReferenceAt2(p[i], k))
        t.BareRealDiv(t, c)
        a.BareRealReferenceAt2(p[j], k).BareRealSub(a.BareRealReferenceAt2(p[j], k), t)
        if math.IsNaN(a.BareRealReferenceAt2(p[j], k).Value()) {
          goto singular
        }
      }
    }
    a.BareRealReferenceAt2(p[i], i).BareRealDiv(a.BareRealReferenceAt2(p[i], i), c)
    if math.IsNaN(a.BareRealReferenceAt2(p[i], i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealReferenceAt2(p[i], k).BareRealDiv(x.BareRealReferenceAt2(p[i], k), c)
    }
    // normalize ith element in b
    b[p[i]].(*BareReal).BareRealDiv(b[p[i]].(*BareReal), c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return
singular:
  panic("system is computationally singular")
}

func gaussJordanTriangular_bare_real_dense(a, x *DenseMatrix, b Vector, submatrix []bool) {
  fmt.Println("running dense")
  t := NewBareReal(0.0)
  c := NewBareReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    panic("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    panic("GaussJordan(): b has invalid dimension!")
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealReferenceAt2(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealReferenceAt2(j, i), b[i].(*BareReal))
      t.BareRealDiv(t, c)
      b[j].(*BareReal).BareRealSub(b[j].(*BareReal), t)
      if math.IsNaN(b[j].(*BareReal).Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt2(j, i), x.BareRealReferenceAt2(i, k))
        t.BareRealDiv(t, c)
        x.BareRealReferenceAt2(j, k).BareRealSub(x.BareRealReferenceAt2(j, k), t)
        if math.IsNaN(x.BareRealReferenceAt2(j, k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt2(j, i), a.BareRealReferenceAt2(i, k))
        t.BareRealDiv(t, c)
        a.BareRealReferenceAt2(j, k).BareRealSub(a.BareRealReferenceAt2(j, k),t)
        if math.IsNaN(a.BareRealReferenceAt2(j, k).Value()) {
          goto singular
        }
      }
    }
    a.BareRealReferenceAt2(i, i).BareRealDiv(a.BareRealReferenceAt2(i, i), c)
    if math.IsNaN(a.BareRealReferenceAt2(i, i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealReferenceAt2(i, k).BareRealDiv(x.BareRealReferenceAt2(i, k), c)
    }
    // normalize ith element in b
    b[i].(*BareReal).BareRealDiv(b[i].(*BareReal), c)
  }
  return
singular:
  panic("system is computationally singular")
}
