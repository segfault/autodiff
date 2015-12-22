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

import   "math"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Submatrix struct {
  Value []bool
}

/* -------------------------------------------------------------------------- */

func gaussJordan(a, x Matrix, b Vector, submatrix []bool, epsilon float64) {
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
    if math.Abs(a.At(p[i], i).Value()) < epsilon {
      panic("GaussJordan(): matrix is singular!")
    }
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if !submatrix[j] {
        continue
      }
      if math.Abs(a.At(p[j], i).Value()) > math.Abs(a.At(p[maxrow], i).Value()) {
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
      c := Div(a.At(p[j], i), a.At(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        a.Set(Sub(a.At(p[j], k), Mul(a.At(p[i], k), c)),
          p[j], k)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
        if !submatrix[k] {
          continue
        }
        // a[j, k] -= a[i, k]*c
        x.Set(Sub(x.At(p[j], k), Mul(x.At(p[i], k), c)),
          p[j], k)
      }
      // same for b: b[j] -= b[j]*c
      b[p[j]] = Sub(b[p[j]], Mul(b[p[i]], c))
    }
  }
  // backsubstitute
  for i := n-1; i >= 0; i-- {
    if !submatrix[i] {
      continue
    }
    c := a.At(p[i], i)
    for j := 0; j < i; j++ {
      if !submatrix[j] {
        continue
      }
      // b[j] -= a[j,i]*b[i]/c
      b[p[j]] = Sub(b[p[j]], Div(Mul(a.At(p[j], i), b[p[i]]), c))
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // x[j,k] -= a[j,i]*x[i,k]/c
        x.Set(Sub(x.At(p[j], k), Div(Mul(a.At(p[j], i), x.At(p[i], k)), c)),
          p[j], k)
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        if !submatrix[k] {
          continue
        }
        // a[j,k] -= a[j,i]*a[i,k]/c
        a.Set(Sub(a.At(p[j], k), Div(Mul(a.At(p[j], i), a.At(p[i], k)), c)),
          p[j], k)
      }
    }
    a.Set(Div(a.At(p[i], i), c),
      p[i], i)
    // normalize ith row in x
    for k := 0; k < n; k++ {
      if !submatrix[k] {
        continue
      }
      x.Set(Div(x.At(p[i], k), c),
        p[i], k)
    }
    // normalize ith element in b
    b[p[i]] = Div(b[p[i]], c)
  }
  a.PermuteRows(p)
  x.PermuteRows(p)
  b.Permute(p)
}

/* -------------------------------------------------------------------------- */

func Run(a, x Matrix, b Vector, args ...interface{}) {

  epsilon   := Epsilon  {1e-120}.Value
  submatrix := Submatrix{   nil}.Value

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case []bool:
      submatrix = a
    case float64:
      epsilon = a
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
  gaussJordan(a, x, b, submatrix, epsilon)
}
