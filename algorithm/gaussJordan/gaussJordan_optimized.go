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
import   "errors"
import   "math"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func gaussJordan_RealDense(a, x *DenseMatrix, b Vector, submatrix []bool) error {
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
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.RealReferenceAt(p[j], i).GetValue()) > math.Abs(a.RealReferenceAt(p[maxrow], i).GetValue()) {
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
      c.RealDiv(a.RealReferenceAt(p[j], i), a.RealReferenceAt(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.RealMul(a.RealReferenceAt(p[i], k), c)
        a.RealReferenceAt(p[j], k).RealSub(a.RealReferenceAt(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.RealMul(x.RealReferenceAt(p[i], k), c)
        x.RealReferenceAt(p[j], k).RealSub(x.RealReferenceAt(p[j], k), t)
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
    c.Copy(a.RealReferenceAt(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealReferenceAt(p[j], i), b[p[i]].(*Real))
      t.RealDiv(t, c)
      b[p[j]].(*Real).RealSub(b[p[j]].(*Real), t)
      if math.IsNaN(b[p[j]].(*Real).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealReferenceAt(p[j], i), x.RealReferenceAt(p[i], k))
        t.RealDiv(t, c)
        x.RealReferenceAt(p[j], k).RealSub(x.RealReferenceAt(p[j], k), t)
        if math.IsNaN(x.RealReferenceAt(p[j], k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealReferenceAt(p[j], i), a.RealReferenceAt(p[i], k))
        t.RealDiv(t, c)
        a.RealReferenceAt(p[j], k).RealSub(a.RealReferenceAt(p[j], k), t)
        if math.IsNaN(a.RealReferenceAt(p[j], k).GetValue()) {
          goto singular
        }
      }
    }
    a.RealReferenceAt(p[i], i).RealDiv(a.RealReferenceAt(p[i], i), c)
    if math.IsNaN(a.RealReferenceAt(p[i], i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealReferenceAt(p[i], k).RealDiv(x.RealReferenceAt(p[i], k), c)
    }
    // normalize ith element in b
    b[p[i]].(*Real).RealDiv(b[p[i]].(*Real), c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return nil
singular:
  return errors.New("system is computationally singular")
}

func gaussJordanTriangular_RealDense(a, x *DenseMatrix, b Vector, submatrix []bool) error {
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
    c.Copy(a.RealReferenceAt(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.RealMul(a.RealReferenceAt(j, i), b.RealReferenceAt(i))
      t.RealDiv(t, c)
      b.RealReferenceAt(j).RealSub(b.RealReferenceAt(j), t)
      if math.IsNaN(b.RealReferenceAt(j).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.RealMul(a.RealReferenceAt(j, i), x.RealReferenceAt(i, k))
        t.RealDiv(t, c)
        x.RealReferenceAt(j, k).RealSub(x.RealReferenceAt(j, k), t)
        if math.IsNaN(x.RealReferenceAt(j, k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.RealMul(a.RealReferenceAt(j, i), a.RealReferenceAt(i, k))
        t.RealDiv(t, c)
        a.RealReferenceAt(j, k).RealSub(a.RealReferenceAt(j, k),t)
        if math.IsNaN(a.RealReferenceAt(j, k).GetValue()) {
          goto singular
        }
      }
    }
    a.RealReferenceAt(i, i).RealDiv(a.RealReferenceAt(i, i), c)
    if math.IsNaN(a.RealReferenceAt(i, i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.RealReferenceAt(i, k).RealDiv(x.RealReferenceAt(i, k), c)
    }
    // normalize ith element in b
    b.RealReferenceAt(i).RealDiv(b.RealReferenceAt(i), c)
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}

/* -------------------------------------------------------------------------- */

func gaussJordan_BareRealDense(a, x *DenseMatrix, b Vector, submatrix []bool) error {
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
    return errors.New("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    return errors.New("GaussJordan(): b has invalid dimension!")
  }
  // loop over columns
  for i := 0; i < n; i++ {
    if !submatrix[i] {
      continue
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.BareRealReferenceAt(p[j], i).GetValue()) > math.Abs(a.BareRealReferenceAt(p[maxrow], i).GetValue()) {
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
      c.BareRealDiv(a.BareRealReferenceAt(p[j], i), a.BareRealReferenceAt(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.BareRealMul(a.BareRealReferenceAt(p[i], k), c)
        a.BareRealReferenceAt(p[j], k).BareRealSub(a.BareRealReferenceAt(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.BareRealMul(x.BareRealReferenceAt(p[i], k), c)
        x.BareRealReferenceAt(p[j], k).BareRealSub(x.BareRealReferenceAt(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.BareRealMul(b.BareRealReferenceAt(p[i]), c)
      b.BareRealReferenceAt(p[j]).BareRealSub(b.BareRealReferenceAt(p[j]), t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealReferenceAt(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealReferenceAt(p[j], i), b.BareRealReferenceAt(p[i]))
      t.BareRealDiv(t, c)
      b.BareRealReferenceAt(p[j]).BareRealSub(b.BareRealReferenceAt(p[j]), t)
      if math.IsNaN(b.BareRealReferenceAt(p[j]).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt(p[j], i), x.BareRealReferenceAt(p[i], k))
        t.BareRealDiv(t, c)
        x.BareRealReferenceAt(p[j], k).BareRealSub(x.BareRealReferenceAt(p[j], k), t)
        if math.IsNaN(x.BareRealReferenceAt(p[j], k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt(p[j], i), a.BareRealReferenceAt(p[i], k))
        t.BareRealDiv(t, c)
        a.BareRealReferenceAt(p[j], k).BareRealSub(a.BareRealReferenceAt(p[j], k), t)
        if math.IsNaN(a.BareRealReferenceAt(p[j], k).GetValue()) {
          goto singular
        }
      }
    }
    a.BareRealReferenceAt(p[i], i).BareRealDiv(a.BareRealReferenceAt(p[i], i), c)
    if math.IsNaN(a.BareRealReferenceAt(p[i], i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealReferenceAt(p[i], k).BareRealDiv(x.BareRealReferenceAt(p[i], k), c)
    }
    // normalize ith element in b
    b.BareRealReferenceAt(p[i]).BareRealDiv(b.BareRealReferenceAt(p[i]), c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return nil
singular:
  return errors.New("system is computationally singular")
}

func gaussJordanTriangular_BareRealDense(a, x *DenseMatrix, b Vector, submatrix []bool) error {
  t := NewBareReal(0.0)
  c := NewBareReal(0.0)
  // number of rows
  n, _ := a.Dims()
  // x and b should have the same number of rows
  if m, _ := x.Dims(); m != n {
    return errors.New("GaussJordan(): x has invalid dimension!")
  }
  if len(b) != n {
    return errors.New("GaussJordan(): b has invalid dimension!")
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.BareRealReferenceAt(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.BareRealMul(a.BareRealReferenceAt(j, i), b.BareRealReferenceAt(i))
      t.BareRealDiv(t, c)
      b.BareRealReferenceAt(j).BareRealSub(b.BareRealReferenceAt(j), t)
      if math.IsNaN(b.BareRealReferenceAt(j).GetValue()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt(j, i), x.BareRealReferenceAt(i, k))
        t.BareRealDiv(t, c)
        x.BareRealReferenceAt(j, k).BareRealSub(x.BareRealReferenceAt(j, k), t)
        if math.IsNaN(x.BareRealReferenceAt(j, k).GetValue()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.BareRealMul(a.BareRealReferenceAt(j, i), a.BareRealReferenceAt(i, k))
        t.BareRealDiv(t, c)
        a.BareRealReferenceAt(j, k).BareRealSub(a.BareRealReferenceAt(j, k),t)
        if math.IsNaN(a.BareRealReferenceAt(j, k).GetValue()) {
          goto singular
        }
      }
    }
    a.BareRealReferenceAt(i, i).BareRealDiv(a.BareRealReferenceAt(i, i), c)
    if math.IsNaN(a.BareRealReferenceAt(i, i).GetValue()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.BareRealReferenceAt(i, k).BareRealDiv(x.BareRealReferenceAt(i, k), c)
    }
    // normalize ith element in b
    b.BareRealReferenceAt(i).BareRealDiv(b.BareRealReferenceAt(i), c)
  }
  return nil
singular:
  return errors.New("system is computationally singular")
}
