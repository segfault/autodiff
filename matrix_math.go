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

//import "fmt"

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

func MxS(a Matrix, s Scalar) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  for i := 0; i < a.rows; i++ {
    for j := 0; j < a.cols; j++ {
      r.Set(Mul(a.At(i,j), s), i, j)
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

func Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
  x := x_.Clone()
  t := x_.ElementType()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  n := len(y)
  m := len(x)
  r := NullMatrix(x.ElementType(), n, m)
  if n != len(y) {
    panic("Jacobian(): dimensions do not match!")
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.Set(NewScalar(t, y[i].Derivative(1, j)),
        i, j)
    }
  }
  return r
}
