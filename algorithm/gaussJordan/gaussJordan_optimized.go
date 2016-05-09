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

//import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func gaussJordan_RealDense(a, x *DenseMatrix, b Vector, submatrix []bool, epsilon float64) {
  t := NewReal(0.0)
  c := NewReal(0.0)
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
    if math.Abs(a.RealReferenceAt2(p[i], i).Value()) < epsilon {
      panic("GaussJordan(): matrix is singular!")
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.RealReferenceAt2(p[j], i).Value()) > math.Abs(a.RealReferenceAt2(p[maxrow], i).Value()) {
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
      c.RealDiv(a.RealReferenceAt2(p[j], i), a.RealReferenceAt2(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.RealMul(a.RealReferenceAt2(p[i], k), c)
        a.RealReferenceAt2(p[j], k).RealSub(a.RealReferenceAt2(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.RealMul(x.RealReferenceAt2(p[i], k), c)
        x.RealReferenceAt2(p[j], k).RealSub(x.RealReferenceAt2(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.RealMul(b[p[i]].(*Real), c)
      b[p[j]].(*Real).RealSub(b[p[j]].(*Real), t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.RealReferenceAt2(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealReferenceAt2(p[j], i), b[p[i]].(*Real))
      t.RealDiv(t, c)
      b[p[j]].(*Real).RealSub(b[p[j]].(*Real), t)
      if math.IsNaN(b[p[j]].(*Real).Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealReferenceAt2(p[j], i), x.RealReferenceAt2(p[i], k))
        t.RealDiv(t, c)
        x.RealReferenceAt2(p[j], k).RealSub(x.RealReferenceAt2(p[j], k), t)
        if math.IsNaN(x.RealReferenceAt2(p[j], k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealReferenceAt2(p[j], i), a.RealReferenceAt2(p[i], k))
        t.RealDiv(t, c)
        a.RealReferenceAt2(p[j], k).RealSub(a.RealReferenceAt2(p[j], k), t)
        if math.IsNaN(a.RealReferenceAt2(p[j], k).Value()) {
          goto singular
        }
      }
    }
    a.RealReferenceAt2(p[i], i).RealDiv(a.RealReferenceAt2(p[i], i), c)
    if math.IsNaN(a.RealReferenceAt2(p[i], i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealReferenceAt2(p[i], k).RealDiv(x.RealReferenceAt2(p[i], k), c)
    }
    // normalize ith element in b
    b[p[i]].(*Real).RealDiv(b[p[i]].(*Real), c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return
singular:
  panic("system is computationally singular")
}

func gaussJordanTriangular_RealDense(a, x *DenseMatrix, b Vector, submatrix []bool) {
  t := NewReal(0.0)
  c := NewReal(0.0)
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
    c.Copy(a.RealReferenceAt2(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealReferenceAt2(j, i), b.RealReferenceAt1(i))
      t.RealDiv(t, c)
      b.RealReferenceAt1(j).RealSub(b.RealReferenceAt1(j), t)
      if math.IsNaN(b.RealReferenceAt1(j).Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealReferenceAt2(j, i), x.RealReferenceAt2(i, k))
        t.RealDiv(t, c)
        x.RealReferenceAt2(j, k).RealSub(x.RealReferenceAt2(j, k), t)
        if math.IsNaN(x.RealReferenceAt2(j, k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealReferenceAt2(j, i), a.RealReferenceAt2(i, k))
        t.RealDiv(t, c)
        a.RealReferenceAt2(j, k).RealSub(a.RealReferenceAt2(j, k),t)
        if math.IsNaN(a.RealReferenceAt2(j, k).Value()) {
          goto singular
        }
      }
    }
    a.RealReferenceAt2(i, i).RealDiv(a.RealReferenceAt2(i, i), c)
    if math.IsNaN(a.RealReferenceAt2(i, i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealReferenceAt2(i, k).RealDiv(x.RealReferenceAt2(i, k), c)
    }
    // normalize ith element in b
    b.RealReferenceAt1(i).RealDiv(b.RealReferenceAt1(i), c)
  }
  return
singular:
  panic("system is computationally singular")
}

/* -------------------------------------------------------------------------- */

func gaussJordan_BareRealDense(a, x *DenseMatrix, b Vector, submatrix []bool, epsilon float64) {
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
      t.BareRealMul(b.BareRealReferenceAt1(p[i]), c)
      b.BareRealReferenceAt1(p[j]).BareRealSub(b.BareRealReferenceAt1(p[j]), t)
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
      t.BareRealMul(a.BareRealReferenceAt2(p[j], i), b.BareRealReferenceAt1(p[i]))
      t.BareRealDiv(t, c)
      b.BareRealReferenceAt1(p[j]).BareRealSub(b.BareRealReferenceAt1(p[j]), t)
      if math.IsNaN(b.BareRealReferenceAt1(p[j]).Value()) {
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
    b.BareRealReferenceAt1(p[i]).BareRealDiv(b.BareRealReferenceAt1(p[i]), c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return
singular:
  panic("system is computationally singular")
}

func gaussJordanTriangular_BareRealDense(a, x *DenseMatrix, b Vector, submatrix []bool) {
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
      t.BareRealMul(a.BareRealReferenceAt2(j, i), b.BareRealReferenceAt1(i))
      t.BareRealDiv(t, c)
      b.BareRealReferenceAt1(j).BareRealSub(b.BareRealReferenceAt1(j), t)
      if math.IsNaN(b.BareRealReferenceAt1(j).Value()) {
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
    b.BareRealReferenceAt1(i).BareRealDiv(b.BareRealReferenceAt1(i), c)
  }
  return
singular:
  panic("system is computationally singular")
}
