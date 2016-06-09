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
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("MEqual(): matrix dimensions do not match!")
  }
  v1 := a.GetValues()
  v2 := b.GetValues()
  for i := 0; i < len(v1); i++ {
    if !Equal(v1[i], v2[i]) {
      return false
    }
  }
  return true
}

/* -------------------------------------------------------------------------- */

// Element-wise addition of two matrices. The result is stored in r.
func (r *DenseMatrix) MaddM(a, b Matrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Add(a.ReferenceAt2(i, j), b.ReferenceAt2(i, j))
    }
  }
  return r
}

// Element-wise addition of two matrices.
func MaddM(a, b Matrix) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MaddM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Add scalar b to all elements of a. The result is stored in r.
func (r *DenseMatrix) MaddS(a Matrix, b Scalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Add(a.ReferenceAt2(i, j), b)
    }
  }
  return r
}

// Add scalar b to all elements of a.
func MaddS(a Matrix, b Scalar) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MaddS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise substraction of two matrices. The result is stored in r.
func (r *DenseMatrix) MsubM(a, b Matrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Sub(a.ReferenceAt2(i, j), b.ReferenceAt2(i, j))
    }
  }
  return r
}

// Element-wise substraction of two matrices.
func MsubM(a, b Matrix) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MsubM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Substract b from all elements of a. The result is stored in r.
func (r *DenseMatrix) MsubS(a Matrix, b Scalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Sub(a.ReferenceAt2(i, j), b)
    }
  }
  return r
}

// Substract b from all elements of a.
func MsubS(a Matrix, b Scalar) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MsubS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise multiplication of two matrices. The result is stored in r.
func (r *DenseMatrix) MmulM(a, b Matrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Mul(a.ReferenceAt2(i, j), b.ReferenceAt2(i, j))
    }
  }
  return r
}

// Element-wise multiplication of two matrices.
func MmulM(a, b Matrix) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MmulM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Multiply all elements of a with b. The result is stored in r.
func (r *DenseMatrix) MmulS(a Matrix, b Scalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Mul(a.ReferenceAt2(i, j), b)
    }
  }
  return r
}

// Multiply all elements of a with b.
func MmulS(a Matrix, b Scalar) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MmulS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Element-wise division of two matrices. The result is stored in r.
func (r *DenseMatrix) MdivM(a, b Matrix) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n || m1 != m || n2 != n || m2 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Div(a.ReferenceAt2(i, j), b.ReferenceAt2(i, j))
    }
  }
  return r
}

// Element-wise division of two matrices.
func MdivM(a, b Matrix) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MdivM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Divide all elements of a by b. The result is stored in r.
func (r *DenseMatrix) MdivS(a Matrix, b Scalar) Matrix {
  n,  m  := r.Dims()
  n1, m1 := a.Dims()
  if n1 != n || m1 != m {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Div(a.ReferenceAt2(i, j), b)
    }
  }
  return r
}

// Divide all elements of a by b.
func MdivS(a Matrix, b Scalar) Matrix {
  n, m := a.Dims()
  r := NullDenseMatrix(a.ElementType(), n, m)
  r.MdivS(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Matrix product of a and b. The result is stored in r.
func (r *DenseMatrix) MdotM(a, b Matrix) Matrix {
  n, m := r.Dims()
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  t := ZeroScalar(a.ElementType())
  if n1 != n || m2 != m || n1 != m2 || m1 != n2 {
    panic("matrix dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Reset()
      for k := 0; k < m1; k++ {
        t.Mul(a.ReferenceAt2(i, k), b.ReferenceAt2(k, j))
        r.ReferenceAt2(i, j).Add(r.ReferenceAt2(i, j), t)
      }
    }
  }
  return r
}

// Matrix product of a and b.
func MdotM(a, b Matrix) Matrix {
  n1, _  := a.Dims()
  _,  m2 := b.Dims()
  r := NullDenseMatrix(a.ElementType(), n1, m2)
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
      t.Mul(a.ReferenceAt2(i, j), b[j])
      r[i].Add(r[i], t)
    }
  }
  return r
}

// Matrix vector product of a and b.
func MdotV(a Matrix, b Vector) Vector {
  n, _ := a.Dims()
  r := NullVector(a.ElementType(), n)
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
      t.Mul(a[j], b.ReferenceAt2(j, i))
      r[i].Add(r[i], t)
    }
  }
  return r
}

// Vector matrix product of a and b.
func VdotM(a Vector, b Matrix) Vector {
  _, m := b.Dims()
  r := NullVector(a.ElementType(), m)
  r.VdotM(a, b)
  return r
}

/* -------------------------------------------------------------------------- */

// Outer product of two vectors. The result is stored in r.
func (r *DenseMatrix) Outer(a, b Vector) Matrix {
  n, m := r.Dims()
  if len(a) != n || len(b) != m {
    panic("matrix/vector dimensions do not match!")
  }
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).Mul(a[i], b[j])
    }
  }
  return r
}

// Outer product of two vectors.
func Outer(a, b Vector) Matrix {
  r := NullDenseMatrix(a.ElementType(), len(a), len(b))
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
    t.Add(t, a.ReferenceAt2(i,i))
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
  v := a.GetValues()
  s := Pow(v[0], NewBareReal(2.0))
  for i := 1; i < len(v); i++ {
    t.Pow(v[i], c)
    s.Add(s, t)
  }
  return s
}

/* -------------------------------------------------------------------------- */

// Compute the Jacobian of f at x_. The result is stored in r.
func (r *DenseMatrix) Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
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
      r.ReferenceAt2(i, j).SetValue(y[i].Derivative(1, j))
    }
  }
  return r
}

// Compute the Jacobian of f at x_.
func Jacobian(f func(Vector) Vector, x_ Vector) Matrix {
  x := x_.Clone()
  x.Variables(1)
  // compute Jacobian
  y := f(x)
  n := len(y)
  m := len(x)
  r := NullDenseMatrix(x.ElementType(), n, m)
  if n != len(y) {
    panic("Jacobian(): dimensions do not match!")
  }
  // copy derivatives
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ReferenceAt2(i, j).SetValue(y[i].Derivative(1, j))
    }
  }
  return r
}
