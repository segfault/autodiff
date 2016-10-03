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

package hessnebergReduction

/* -------------------------------------------------------------------------- */

import   "fmt"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  H Matrix
  X Vector
  U Vector
  S Scalar
}

/* -------------------------------------------------------------------------- */

func fu(x, u Vector, s Scalar) Vector {
  // s = -||x||
  s.Vnorm(x)
  s.Neg(s)
  // u = x - s e_1
  u[0].Sub(x[0], s)
  for i := 1; i < len(x); i++ {
    u[i].Set(x[i])
  }
  // s = ||u||
  s.Vnorm(u)
  // u = u/s
  u.VdivS(u, s)
  return u
}

func hessenbergReduction(a Matrix, x, u Vector, s Scalar) (Matrix, error) {
  n, _ := a.Dims()

  for k := 0; k < n-2; k++ {
    // copy column below main diagonal from A to x,
    // x = (A[k+1,k], A[k+2,k], ..., A[n-1,k])
    for i := k+1; i < n; i++ {
      x[i].Set(a.ReferenceAt(i, k))
    }
    fu(x[k+1:n], u[k+1:n], s)
    // A <- P_k A = A - 2 u (u^t A)
    // i) compute u^t A and store it in x
    for j := k; j < n; j++ {
      x[j].Reset()
      for i := k+1; i < n; i++ {
        s.Mul(u[i], a.ReferenceAt(i, j))
        x[j].Add(x[j], s)
      }
    }
    // ii) compute A - 2 u (u^t A) = A - 2 u x^t
    for i := k+1; i < n; i++ {
      for j := k; j < n; j++ {
        s.Mul(u[i], x[j])
        s.Add(s, s)
        a.ReferenceAt(i, j).Sub(a.ReferenceAt(i, j), s)
      }
    }
    // A <- A P_k = A - 2 (A u) u^t
    // i) compute A u and store it in x
    for i := 0; i < n; i++ {
      x[i].Reset()
      for j := k+1; j < n; j++ {
        s.Mul(a.ReferenceAt(i, j), u[j])
        x[i].Add(x[i], s)
      }
    }
    // ii) compute A - 2 (A u) u^t = A - 2 x u^t
    for i := 0; i < n; i++ {
      for j := k+1; j < n; j++ {
        s.Mul(x[i], u[j])
        s.Add(s, s)
        a.ReferenceAt(i, j).Sub(a.ReferenceAt(i, j), s)
      }
    }
  }
  return a, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  var h Matrix
  var x Vector
  var u Vector
  var s Scalar

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case InSitu:
      h = a.H
    }
  }
  if h == nil {
    h = NullDenseMatrix(t, n, m)
  } else {
    if u, v := h.Dims(); u != n || v != m {
      return nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if x == nil {
    x = NullVector(t, n)
  } else {
    if u := len(x); u != n {
      return nil, fmt.Errorf("x has invalid dimension (%d instead of %d)", u, n)
    }
  }
  if u == nil {
    u = NullVector(t, n)
  } else {
    if u := len(u); u != n {
      return nil, fmt.Errorf("u has invalid dimension (%d instead of %d)", u, n)
    }
  }
  if s == nil {
    s = NullScalar(t)
  }
  return hessenbergReduction(a, x, u, s)
}