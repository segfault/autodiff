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

// True if matrix a equals b.
func Mequal(a, b Matrix) bool {
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

/* -------------------------------------------------------------------------- */

// Element-wise addition of two matrices. The result is stored in r.
func (r *Matrix) MaddM(a, b Matrix) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m || b.rows != n || b.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Add(a.ReferenceAt(i, j), b.ReferenceAt(i, j))
    }
  }
  return *r
}

// Element-wise addition of two matrices.
func MaddM(a, b Matrix) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MaddM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Add scalar b to all elements of a. The result is stored in r.
func (r *Matrix) MaddS(a Matrix, b Scalar) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Add(a.ReferenceAt(i, j), b)
    }
  }
  return *r
}

// Add scalar b to all elements of a.
func MaddS(a Matrix, b Scalar) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MaddS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of two matrices. The result is stored in r.
func (r *Matrix) MsubM(a, b Matrix) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m || b.rows != n || b.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Sub(a.ReferenceAt(i, j), b.ReferenceAt(i, j))
    }
  }
  return *r
}

// Element-wise substraction of two matrices.
func MsubM(a, b Matrix) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MsubM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Substract b from all elements of a. The result is stored in r.
func (r *Matrix) MsubS(a Matrix, b Scalar) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Sub(a.ReferenceAt(i, j), b)
    }
  }
  return *r
}

// Substract b from all elements of a.
func MsubS(a Matrix, b Scalar) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MsubS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise multiplication of two matrices. The result is stored in r.
func (r *Matrix) MmulM(a, b Matrix) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m || b.rows != n || b.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Mul(a.ReferenceAt(i, j), b.ReferenceAt(i, j))
    }
  }
  return *r
}

// Element-wise multiplication of two matrices.
func MmulM(a, b Matrix) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MmulM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Multiply all elements of a with b. The result is stored in r.
func (r *Matrix) MmulS(a Matrix, b Scalar) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Mul(a.ReferenceAt(i, j), b)
    }
  }
  return *r
}

// Multiply all elements of a with b.
func MmulS(a Matrix, b Scalar) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MmulS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of two matrices. The result is stored in r.
func (r *Matrix) MdivM(a, b Matrix) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m || b.rows != n || b.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Div(a.ReferenceAt(i, j), b.ReferenceAt(i, j))
    }
  }
  return *r
}

// Element-wise division of two matrices.
func MdivM(a, b Matrix) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MdivM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Divide all elements of a by b. The result is stored in r.
func (r *Matrix) MdivS(a Matrix, b Scalar) Matrix {
  n, m := r.Dims()
  if a.rows != n || a.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Div(a.ReferenceAt(i, j), b)
    }
  }
  return *r
}

// Divide all elements of a by b.
func MdivS(a Matrix, b Scalar) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, a.cols)
  r.MdivS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Matrix product of a and b. The result is stored in r.
func (r *Matrix) MdotM(a, b Matrix) Matrix {
  n, m := r.Dims()
  if a.rows != n || b.cols != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Reset()
      for n := 0; n < a.cols; n++ {
        r.ReferenceAt(i, j).Add(r.ReferenceAt(i, j), Mul(a.ReferenceAt(i, n), b.ReferenceAt(n, j)))
      }
    }
  }
  return *r
}

// Matrix product of a and b.
func MdotM(a, b Matrix) Matrix {
  r := NullMatrix(a.ElementType(), a.rows, b.cols)
  r.MdotM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Matrix vector product of a and b. The result is stored in r.
func (r Vector) MdotV(a Matrix, b Vector) Vector {
  n, m := a.Dims()
  if len(r) != n || len(b) != m {
    panic("matrix/vector dimensions do not match!")
  }
  t := ZeroScalar(a.ElementType())
  for i := 0; i < n; i++ {
    r[i].Reset()
    for j := 0; j < m; j++ {
      t.Mul(a.ReferenceAt(i, j), b[j])
      r[i].Add(r[i], t)
    }
  }
  return r
}

// Matrix vector product of a and b.
func MdotV(a Matrix, b Vector) Vector {
  r := NullVector(a.ElementType(), a.rows)
  r.MdotV(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Vector matrix product of a and b. The result is stored in r.
func (r Vector) VdotM(a Vector, b Matrix) Vector {
  n, m := b.Dims()
  if len(r) != m || len(a) != n {
    panic("matrix/vector dimensions do not match!")
  }
  t := ZeroScalar(a.ElementType())
  for i := 0; i < m; i++ {
    r[i].Reset()
    for j := 0; j < n; j++ {
      t.Mul(a[j], b.ReferenceAt(j, i))
      r[i].Add(r[i], t)
    }
  }
  return r
}

// Vector matrix product of a and b.
func VdotM(a Vector, b Matrix) Vector {
  r := NullVector(a.ElementType(), b.cols)
  r.VdotM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Outer product of two vectors. The result is stored in r.
func (r *Matrix) Outer(a, b Vector) Matrix {
  n, m := r.Dims()
  if len(a) != n || len(b) != m {
    panic("matrix/vector dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).Mul(a[i], b[j])
    }
  }
  return *r
}

// Outer product of two vectors.
func Outer(a, b Vector) Matrix {
  r := NullMatrix(a.ElementType(), len(a), len(b))
  r.Outer(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Returns the trace of a.
func Mtrace(a Matrix) Scalar {
  n, m := a.Dims()
  if n != m {
    panic("not a square matrix")
  }
  if n == 0 {
    return nil
  }
  t := a.At(0, 0)
  for i := 1; i < n; i++ {
    t.Add(t, a.ReferenceAt(i,i))
  }
  return t
}

/* -------------------------------------------------------------------------- */

// Frobenius norm.
func Mnorm(a Matrix) Scalar {
  n, m := a.Dims()
  if n == 0 || m == 0 {
    return nil
  }
  c := NewBareReal(2.0)
  t := NewScalar(a.ElementType(), 0.0)
  s := Pow(a.values[0], NewBareReal(2.0))
  for i := 1; i < len(a.values); i++ {
    t.Pow(a.values[i], c)
    s.Add(s, t)
  }
  return s
}

/* -------------------------------------------------------------------------- */

// Compute the Jacobian of f at x_. The result is stored in r.
func (r *Matrix) Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
  n, m := r.Dims()
  x := x_.Clone()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  if len(x) != m || len(y) != n {
    panic("matrix/vector dimensions do not match")
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt(i, j).SetValue(y[i].Derivative(1, j))
    }
  }
  return *r
}

// Compute the Jacobian of f at x_.
func Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
  x := x_.Clone()
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
      r.ReferenceAt(i, j).SetValue(y[i].Derivative(1, j))
    }
  }
  return r
}
