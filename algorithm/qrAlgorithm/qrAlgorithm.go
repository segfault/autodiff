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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/gramSchmidt"
import   "github.com/pbenner/autodiff/algorithm/hessenbergReduction"

/* -------------------------------------------------------------------------- */

type Shift struct {
  Value bool
}

type Epsilon struct {
  Value float64
}

type InSitu struct {
  H Matrix
  U Matrix
  C Vector
  S Vector
  T1, T2, T3 Scalar
}

/* -------------------------------------------------------------------------- */

func givens(a, b, c, s Scalar) {
  // fake temporary variables
  t1 := s
  t2 := c

  t1.Reset()
  // t1 = a^2 + b^2
  t2.Mul(a, a)
  t1.Add(t1, t2)

  t2.Mul(b, b)
  t1.Add(t1, t2)
  // t1 = sqrt(a^2 + b^2)
  t1.Sqrt(t1)

  c.Div(a, t1)
  s.Div(b, t1)
}

func hessenbergQrAlgorithmStep(h, u Matrix, c, s Vector, t1, t2, t3 Scalar, n int, shift bool) {

  if shift {
    t3.Set(h.ReferenceAt(n-1, n-1))
    for i := 0; i < n; i++ {
      g := h.ReferenceAt(i, i)
      g.Sub(g, t3)
    }
  }
  for i := 0; i < n-1; i++ {
    givens(h.ReferenceAt(i, i), h.ReferenceAt(i+1, i), c[i], s[i])
    // multiply with Givens matrix (G H)
    for j := i; j < n; j++ {
      h1 := h.ReferenceAt(i+0, j)
      h2 := h.ReferenceAt(i+1, j)
      // backup h1
      t1.Set(h1)       // t1 = h1
      // update h1
      h1.Mul(c[i], h1) // h1 = c h1
      t2.Mul(s[i], h2) // t2 = s h2
      h1.Add(h1, t2)   // h1 = c h1 + s h2
      // update h2
      t1.Mul(s[i], t1) // t1 =  s h1
      t1.Neg(t1)       // t1 = -s h1
      t2.Mul(c[i], h2) // t2 =  c h2
      h2.Add(t1, t2)   // h2 = -s h1 + c h2
    }
  }
  for i := 0; i < n-1; i++ {
    // multiply with Givens matrix (H G)
    for j := 0; j <= i+1; j++ {
      h1 := h.ReferenceAt(j, i+0)
      h2 := h.ReferenceAt(j, i+1)
      // backup h1
      t1.Set(h1)       // t1 = h1
      // update h1
      h1.Mul(c[i], h1) // h1 = c h1
      t2.Mul(s[i], h2) // t2 = s h2
      h1.Add(h1, t2)   // h1 = c h1 + s h2
      // update h2
      t1.Mul(s[i], t1) // t1 =  s h1
      t1.Neg(t1)       // t1 = -s h1
      t2.Mul(c[i], h2) // t2 =  c h2
      h2.Add(t1, t2)   // h2 = -s h1 + c h2
    }
  }
  if shift {
    for i := 0; i < n; i++ {
      g := h.ReferenceAt(i, i)
      g.Add(g, t3)
    }
  }
  if u != nil {
    for i := 0; i < n-1; i++ {
      for j := 0; j < n; j++ {
        u1 := u.ReferenceAt(j, i+0)
        u2 := u.ReferenceAt(j, i+1)
        // backup u1
        t1.Set(u1)       // t1 = u1
        // update u1
        u1.Mul(c[i], u1) // u1 = c u1
        t2.Mul(s[i], u2) // t2 = s u2
        u1.Add(u1, t2)   // u1 = c u1 + s u2
        // update u2
        t1.Mul(s[i], t1) // t1 =  s u1
        t1.Neg(t1)       // t1 = -s u1
        t2.Mul(c[i], u2) // t2 =  c u2
        u2.Add(t1, t2)   // u2 = -s u1 + c u2
      }
    }
  }
}

func hessenbergQrAlgorithm(h, u Matrix, c, s Vector, t1, t2, t3 Scalar, epsilon float64, shift bool) (Matrix, Matrix, error) {
  n, _ := h.Dims()

  _, _, err := hessenbergReduction.Run(h, hessenbergReduction.InSitu{
    H: h, V: u, X: c, U: s, S: t1})
  if err != nil {
    return nil, nil, err
  }

  for n > 1 {

    if v := h.ReferenceAt(n-1, n-2); math.Abs(v.GetValue()) < epsilon {
      n--
    } else {
      hessenbergQrAlgorithmStep(h, u, c, s, t1, t2, t3, n, false)
    }
  }

  return h, u, nil
}

/* -------------------------------------------------------------------------- */

func qrAlgorithm(a, b, q, r Matrix, t ScalarType) (Matrix, Matrix, error) {

  n, m := a.Dims()
  
  a = a.Clone()

  for {
    // reset values of q and r
    for i := 0; i < n; i++ {
      for j := 0; j < m; j++ {
        q.ReferenceAt(i, j).Reset()
        r.ReferenceAt(i, j).Reset()
      }
    }
    q, r, _ = gramSchmidt.Run(a, gramSchmidt.InSitu{q, r})
    b.MdotM(r, q)
    if Mnorm(a.MsubM(a, b)).GetValue() < 1e-12 {
      break
    }
    a, b = b, a
  }

  return q, r, nil
}

/* -------------------------------------------------------------------------- */

func Run(a Matrix, args ...interface{}) (Matrix, Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  var h Matrix
  var u Matrix
  var c Vector
  var s Vector
  var t1 Scalar
  var t2 Scalar
  var t3 Scalar

  epsilon := 1e-12
  shift   := true

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case InSitu:
      h = tmp.H
      u = tmp.U
      c = tmp.C
      s = tmp.S
      t1 = tmp.T1
      t2 = tmp.T2
      t3 = tmp.T3
    case Epsilon:
      epsilon = tmp.Value
    case Shift:
      shift = tmp.Value
    }
  }
  if h == nil {
    h = a.Clone()
  } else {
    if n1, m1 := u.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
    // initialize h if necessary
    if h != a {
      h.Copy(a)
    }
  }
  if u == nil {
    u = IdentityMatrix(t, n)
  } else {
    if n1, m1 := u.Dims(); n1 != n || m1 != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", n1, m1, n, m)
    }
  }
  if c == nil {
    c = NullVector(t, n)
  } else {
    if n1 := len(c); n1 != n {
      return nil, nil, fmt.Errorf("c has invalid dimension (%d instead of %d)", n1, n)
    }
  }
  if s == nil {
    s = NullVector(t, n)
  } else {
    if n1 := len(s); n1 != n {
      return nil, nil, fmt.Errorf("c has invalid dimension (%d instead of %d)", n1, n)
    }
  }
  if t1 == nil {
    t1 = NullScalar(t)
  }
  if t2 == nil {
    t2 = NullScalar(t)
  }
  if t3 == nil {
    t3 = NullScalar(t)
  }
  return hessenbergQrAlgorithm(h, u, c, s, t1, t2, t3, epsilon, shift)
}
