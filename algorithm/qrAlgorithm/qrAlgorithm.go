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

type Epsilon struct {
  Value float64
}

type InSitu struct {
  // Hessenberg QR algorithm
  C Vector
  S Vector
  T1, T2 Scalar
  // vanilla QR algorithm
  B Matrix
  Q Matrix
  R Matrix
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

func hessenbergQrAlgorithmStep(h Matrix, c, s Vector, t1, t2 Scalar, n int) {

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

}

func hessenbergQrAlgorithm(a Matrix, c, s Vector, t1, t2 Scalar, epsilon float64) (Matrix, error) {
  n, _ := a.Dims()

  h, err := hessenbergReduction.Run(a)
  if err != nil {
    return nil, err
  }

  for n > 1 {

    if v := h.ReferenceAt(n-1, n-2); math.Abs(v.GetValue()) < epsilon {
      n--
    } else {
      hessenbergQrAlgorithmStep(h, c, s, t1, t2, n)
    }

  }
  return h, nil
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

func Run(a Matrix, args ...interface{}) (Matrix, error) {

  n, m := a.Dims()
  t := a.ElementType()

  var b Matrix
  var q Matrix
  var r Matrix
  var c Vector
  var s Vector
  var t1 Scalar
  var t2 Scalar

  epsilon := 1e-12

  // loop over optional arguments
  for _, arg := range args {
    switch tmp := arg.(type) {
    case InSitu:
      b = tmp.B
      q = tmp.Q
      r = tmp.R
      c = tmp.C
      s = tmp.S
      t1 = tmp.T1
      t2 = tmp.T2
    case Epsilon:
      epsilon = tmp.Value
    }
  }
  if b == nil {
    b = NullDenseMatrix(t, n, m)
  } else {
    if u, v := b.Dims(); u != n || v != m {
      return nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if q == nil {
    q = NullDenseMatrix(t, n, m)
  } else {
    if u, v := q.Dims(); u != n || v != m {
      return nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if r == nil {
    r = NullDenseMatrix(t, n, m)
  } else {
    if u, v := r.Dims(); u != n || v != m {
      return nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if c == nil {
    c = NullVector(t, n)
  } else {
    if u := len(c); u != n {
      return nil, fmt.Errorf("c has invalid dimension (%d instead of %d)", u, n)
    }
  }
  if s == nil {
    s = NullVector(t, n)
  } else {
    if u := len(s); u != n {
      return nil, fmt.Errorf("c has invalid dimension (%d instead of %d)", u, n)
    }
  }
  if t1 == nil {
    t1 = NullScalar(t)
  }
  if t2 == nil {
    t2 = NullScalar(t)
  }
  return hessenbergQrAlgorithm(a, c, s, t1, t2, epsilon)
}
