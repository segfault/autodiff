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

package autodiff

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

func MEqual(a, b Matrix) bool {
  if a.rows != b.rows || a.cols != b.cols {
    panic("MEqual(): matrix dimensions do not match!")
  }
  for i, _ := range (a.values) {
    if !Equal(a.values[i], b.values[i]) {
      return false
    }
  }
  return true
}

func MAdd(a, b Matrix) Matrix {
  if a.rows != b.rows || a.cols != b.cols {
    panic("MAdd(): Matrix dimensions do not match!")
  }
  rows := a.rows
  cols := a.cols
  r := NullMatrix(a.ElementType(), rows, cols)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      r.Set(Add(a.At(i, j), b.At(i, j)),
        i, j)
    }
  }
  return r
}

func MSub(a, b Matrix) Matrix {
  if a.rows != b.rows || a.cols != b.cols {
    panic("MSub(): Matrix dimensions do not match!")
  }
  rows := a.rows
  cols := a.cols
  r := NullMatrix(a.ElementType(), rows, cols)
  for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
      r.Set(Sub(a.At(i, j), b.At(i, j)),
        i, j)
    }
  }
  return r
}

func MMul(a, b Matrix) Matrix {
  if a.cols != b.rows {
    panic("MMul(): Matrix dimensions do not match!")
  }
  r := NullMatrix(a.ElementType(), a.rows, b.cols)
  for i := 0; i < r.rows; i++ {
    for j := 0; j < r.cols; j++ {
      for n := 0; n < a.cols; n++ {
        r.Set(Add(r.At(i, j), Mul(a.At(i, n), b.At(n, j))),
          i, j)
      }
    }
  }
  return r
}

func MxV(a Matrix, b Vector) Vector {
  if a.cols != len(b) {
    panic("MxV(): Matrix/Vector dimensions do not match!")
  }
  r := NullVector(a.ElementType(), a.rows)
  for i := 0; i < len(r); i++ {
    for n := 0; n < a.cols; n++ {
      r[i] = Add(r[i], Mul(a.At(i, n), b[n]))
    }
  }
  return r
}

func VxM(a Vector, b Matrix) Vector {
  if len(a) != b.rows {
    panic("VxM(): Matrix/Vector dimensions do not match!")
  }
  r := NullVector(a.ElementType(), b.cols)
  for i := 0; i < len(r); i++ {
    for n := 0; n < b.rows; n++ {
      r[i] = Add(r[i], Mul(a[n], b.At(n, i)))
    }
  }
  return r
}

/* -------------------------------------------------------------------------- */

func MTrace(matrix Matrix) Scalar {
  if matrix.rows != matrix.cols {
    panic("MTrace(): Not a square matrix!")
  }
  if matrix.rows == 0 {
    return nil
  }
  t := matrix.At(0, 0)
  for i := 1; i < matrix.rows; i++ {
    t = Add(t, matrix.At(i,i))
  }
  return t
}

func MNorm(matrix Matrix) Scalar {
  if matrix.rows == 0 && matrix.cols == 0 {
    return nil
  }
  s := Pow(matrix.values[0], 2.0)
  for i := 1; i < len(matrix.values); i++ {
    s = Add(s, Pow(matrix.values[i], 2.0))
  }
  return s
}

// compute the inverse of a matrix with a
// gradient descent method
func mInverse(matrix Matrix) Matrix {
  if matrix.rows != matrix.cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := IdentityMatrix(matrix.ElementType(), matrix.rows)
  r := matrix.Clone()
  // objective function
  f := func(x Vector) Scalar {
    r.SetValues(x)
    s := MNorm(MSub(MMul(matrix, r), I))
    return s
  }
  x, _ := Rprop(f, r.Values(), 1e-12, 0.01, 0.1)
  r.SetValues(x)
  return r
}

func MInverse(matrix Matrix) Matrix {
  if matrix.rows != matrix.cols {
    panic("MInverse(): Not a square matrix!")
  }
  if matrix.rows == 0 {
    panic("MInverse(): Empty matrix!")
  }
  t := matrix.ElementType()
  a := matrix.Clone()
  x := IdentityMatrix(t, matrix.rows)
  b := NullVector(t, matrix.rows) 
  // initialize b with ones
  for i, _ := range b {
    b[i] = NewScalar(t, 1.0)
  }
  // call Gauss-Jordan algorithm
  GaussJordan(a, x, b)
  return x
}

func Jacobian(f func(Vector) Vector, x Vector) Matrix {
  n := 0
  m := len(x)
  r := Matrix{}
  for j := 0; j < m; j++ {
    x[j].Constant()
  }
  for j := 0; j < m; j++ {
    // differentiate with respect to the ith variable
    x[j].Variable(1)
    y := f(x)
    if j == 0 {
      n = len(y)
      r = NullMatrix(x.ElementType(), n, m)
    }
    if n != len(y) {
      panic("Jacobian(): dimensions do not match!")
    }
    // copy derivatives
    for i := 0; i < n; i++ {
      r.Set(NewReal(y[i].Derivative(1)),
        i, j)
    }
    x[j].Constant()
  }
  return r
}

func GaussJordan(a, x Matrix, b Vector) {
  epsilon := 1e-12
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
    // find row with maximum value at column i
    maxrow := i
    for j := i+1; j < n; j++ {
      if math.Abs(a.At(p[j], i).Value()) > math.Abs(a.At(p[maxrow], i).Value()) {
        maxrow = j
      }
    }
    // swap rows
    p[i], p[maxrow] = p[maxrow], p[i]
    // check if matrix is singular
    if math.Abs(a.At(p[i], i).Value()) < epsilon {
      panic("GaussJordan(): matrix is singular!")
    }
    // eliminate column i
    for j := i+1; j < n; j++ {
      // c = a[j, i] / a[i, i]
      c := Div(a.At(p[j], i), a.At(p[i], i))
      // loop over columns in a
      for k := i; k < n; k++ {
        // a[j, k] -= a[i, k]*c
        a.Set(Sub(a.At(p[j], k), Mul(a.At(p[i], k), c)),
          p[j], k)
      }
      // loop over columns in x
      for k := 0; k < n; k++ {
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
    c := a.At(p[i], i)
    for j := 0; j < i; j++ {
      // b[j] -= a[j,i]*b[i]/c
      b[p[j]] = Sub(b[p[j]], Div(Mul(a.At(p[j], i), b[p[i]]), c))
      // loop over colums in x
      for k := n-1; k >= 0; k-- {
        // x[j,k] -= a[j,i]*x[i,k]/c
        x.Set(Sub(x.At(p[j], k), Div(Mul(a.At(p[j], i), x.At(p[i], k)), c)),
          p[j], k)
      }
      // loop over colums in a
      for k := n-1; k >= 0; k-- {
        // a[j,k] -= a[j,i]*a[i,k]/c
        a.Set(Sub(a.At(p[j], k), Div(Mul(a.At(p[j], i), a.At(p[i], k)), c)),
          p[j], k)
      }
    }
    a.Set(Div(a.At(p[i], i), c),
      p[i], i)
    // normalize ith row in x
    for k := 0; k < n; k++ {
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
