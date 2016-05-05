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

type Epsilon struct {
  Value float64
}

type Submatrix struct {
  Value []bool
}

type Triangular struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

func gaussJordan(a, x Matrix, b Vector, submatrix []bool, epsilon float64) {
  t := NewScalar(a.ElementType(), 0.0)
  c := NewScalar(a.ElementType(), 0.0)
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
    if math.Abs(a.ReferenceAt(p[i], i).Value()) < epsilon {
      panic("GaussJordan(): matrix is singular!")
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.ReferenceAt(p[j], i).Value()) > math.Abs(a.ReferenceAt(p[maxrow], i).Value()) {
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
      c.Div(a.ReferenceAt(p[j], i), a.ReferenceAt(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        t.Mul(a.ReferenceAt(p[i], k), c)
        a.ReferenceAt(p[j], k).Sub(a.ReferenceAt(p[j], k), t)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // x[j, k] -= x[i, k]*c
        t.Mul(x.ReferenceAt(p[i], k), c)
        x.ReferenceAt(p[j], k).Sub(x.ReferenceAt(p[j], k), t)
      }
      // same for b: b[j] -= b[j]*c
      t.Mul(b[p[i]], c)
      b[p[j]].Sub(b[p[j]], t)
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c.Copy(a.ReferenceAt(p[i], i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.Mul(a.ReferenceAt(p[j], i), b[p[i]])
      t.Div(t, c)
      b[p[j]].Sub(b[p[j]], t)
      if math.IsNaN(b[p[j]].Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.Mul(a.ReferenceAt(p[j], i), x.ReferenceAt(p[i], k))
        t.Div(t, c)
        x.ReferenceAt(p[j], k).Sub(x.ReferenceAt(p[j], k), t)
        if math.IsNaN(x.ReferenceAt(p[j], k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.Mul(a.ReferenceAt(p[j], i), a.ReferenceAt(p[i], k))
        t.Div(t, c)
        a.ReferenceAt(p[j], k).Sub(a.ReferenceAt(p[j], k), t)
        if math.IsNaN(a.ReferenceAt(p[j], k).Value()) {
          goto singular
        }
      }
    }
    a.ReferenceAt(p[i], i).Div(a.ReferenceAt(p[i], i), c)
    if math.IsNaN(a.ReferenceAt(p[i], i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.ReferenceAt(p[i], k).Div(x.ReferenceAt(p[i], k), c)
    }
    // normalize ith element in b
    b[p[i]].Div(b[p[i]], c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
  return
singular:
  panic("system is computationally singular")
}

func gaussJordanTriangular(a, x Matrix, b Vector, submatrix []bool) {
  t := NewScalar(a.ElementType(), 0.0)
  c := NewScalar(a.ElementType(), 0.0)
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
    c.Copy(a.ReferenceAt(i, i))
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      t.Mul(a.ReferenceAt(j, i), b[i])
      t.Div(t, c)
      b[j].Sub(b[j], t)
      if math.IsNaN(b[j].Value()) {
        goto singular
      }
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        t.Mul(a.ReferenceAt(j, i), x.ReferenceAt(i, k))
        t.Div(t, c)
        x.ReferenceAt(j, k).Sub(x.ReferenceAt(j, k), t)
        if math.IsNaN(x.ReferenceAt(j, k).Value()) {
          goto singular
        }
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        t.Mul(a.ReferenceAt(j, i), a.ReferenceAt(i, k))
        t.Div(t, c)
        a.ReferenceAt(j, k).Sub(a.ReferenceAt(j, k),t)
        if math.IsNaN(a.ReferenceAt(j, k).Value()) {
          goto singular
        }
      }
    }
    a.ReferenceAt(i, i).Div(a.ReferenceAt(i, i), c)
    if math.IsNaN(a.ReferenceAt(i, i).Value()) {
      goto singular
    }
    // normalize ith row in x
    for k := i; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.ReferenceAt(i, k).Div(x.ReferenceAt(i, k), c)
    }
    // normalize ith element in b
    b[i].Div(b[i], c)
  }
  return
singular:
  panic("system is computationally singular")
}

/* -------------------------------------------------------------------------- */

func Run(a, x Matrix, b Vector, args ...interface{}) {

  epsilon    := Epsilon  {1e-120}.Value
  submatrix  := Submatrix{   nil}.Value
  triangular := false

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case Submatrix:
      submatrix = a.Value
    case Epsilon:
      epsilon = a.Value
    case Triangular:
      triangular = a.Value
    default:
      panic("GaussJordan(): Invalid optional argument!")
    }
  }
  // initialize with default values
  if submatrix == nil {
    n, _ := a.Dims()
    submatrix = make([]bool, n)
    for i, _ := range submatrix {
      submatrix[i] = true
    }
  }
  if triangular {
    gaussJordanTriangular(a, x, b, submatrix)
  } else {
    gaussJordan(a, x, b, submatrix, epsilon)
  }
}
