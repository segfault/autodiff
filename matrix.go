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

type Matrix struct {
  values Vector
  n      int
  m      int
  t      bool
}

func NewMatrix(n, m int, values []float64) Matrix {
  tmp := MakeVector(n*m)
  if len(values) == 1 {
    for i := 0; i < n*m; i++ {
      tmp[i] = NewScalar(values[0])
    }
  } else if len(values) == n*m {
    for i := 0; i < n*m; i++ {
      tmp[i] = NewScalar(values[i])
    }
  } else {
    panic("Matrix dimension does not fit input values!")
  }
  return Matrix{tmp, n, m, false}
}

func MakeMatrix(n, m int) Matrix {
  return Matrix{MakeVector(n*m), n, m, false}
}

func IdentityMatrix(n int) Matrix {
  matrix := MakeMatrix(n, n)
  for i := 0; i < n; i++ {
    matrix.Set(i, i, 1)
  }
  return matrix
}

/* -------------------------------------------------------------------------- */

func (matrix Matrix) Clone() Matrix {
  return Matrix{
    values: matrix.values.Clone(),
    n     : matrix.n,
    m     : matrix.m,
    t     : matrix.t}
}

func (matrix Matrix) index(i, j int) int {
  var k int
  if matrix.t {
    k = j*matrix.n + i
  } else {
    k = i*matrix.m + j
  }
  return k
}

func (matrix Matrix) Dims() (int, int) {
  return matrix.n, matrix.m
}

func (matrix Matrix) At(i, j int) float64 {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k].Value()
}

func (matrix Matrix) ScalarAt(i, j int) Scalar {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  return matrix.values[k]
}

func (matrix Matrix) Set(i, j int, v float64) {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  matrix.values[k] = NewScalar(v)
}

func (matrix Matrix) ScalarSet(i, j int, s Scalar) {
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  matrix.values[k] = s
}

func (matrix Matrix) T() Matrix {
  return Matrix{
    values:  matrix.values,
    n     :  matrix.m,
    m     :  matrix.n,
    t     : !matrix.t}
}

/* -------------------------------------------------------------------------- */

func MAdd(a, b Matrix) Matrix {
  if a.n != b.n || a.m != b.m {
    panic("Matrix dimensions do not match!")
  }
  n := a.n
  m := a.m
  r := MakeMatrix(n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ScalarSet(i, j, Add(a.ScalarAt(i, j), b.ScalarAt(i, j)))
    }
  }
  return r
}

func MSub(a, b Matrix) Matrix {
  if a.n != b.n || a.m != b.m {
    panic("Matrix dimensions do not match!")
  }
  n := a.n
  m := a.m
  r := MakeMatrix(n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.ScalarSet(i, j, Sub(a.ScalarAt(i, j), b.ScalarAt(i, j)))
    }
  }
  return r
}

func MMul(a, b Matrix) Matrix {
  if a.m != b.n {
    panic("Matrix dimensions do not match!")
  }
  r := MakeMatrix(a.n, b.m)
  for i := 0; i < r.n; i++ {
    for j := 0; j < r.m; j++ {
      for n := 0; n < a.m; n++ {
        r.ScalarSet(i, j, Add(r.ScalarAt(i, j), Mul(a.ScalarAt(i, n), b.ScalarAt(n, j))))
      }
    }
  }
  return r
}

func MTrace(matrix Matrix) Scalar {
  t := NewScalar(0.0)
  if matrix.n != matrix.m {
    panic("Not a square matrix!")
  }
  for i := 0; i < matrix.n; i++ {
    t = Add(t, matrix.ScalarAt(i,i))
  }
  return t
}

func MNorm(matrix Matrix) Scalar {
  s := NewScalar(0.0)
  for _, v := range matrix.values {
    s = Add(s, Pow(v, 2.0))
  }
  return s
}

func MInverse(matrix Matrix) Matrix {
  if matrix.n != matrix.m {
    panic("Not a square matrix!")
  }
  I := IdentityMatrix(matrix.n)
  r := matrix.Clone()
  // objective function
  f := func(variables Vector) Scalar {
    s := MNorm(MSub(MMul(matrix, r), I))
    return s
  }
  Rprop(f, r.values, 0.01, 1e-12, 0.1)
  return r
}
