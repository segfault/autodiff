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

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/gramSchmidt"

/* -------------------------------------------------------------------------- */

type InSitu struct {
  B Matrix
  Q Matrix
  R Matrix
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

  var b Matrix
  var q Matrix
  var r Matrix

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case InSitu:
      b = a.B
      q = a.Q
      r = a.R
    }
  }
  if b == nil {
    b = NullDenseMatrix(t, n, m)
  } else {
    if u, v := b.Dims(); u != n || v != m {
      return nil, nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if q == nil {
    q = NullDenseMatrix(t, n, m)
  } else {
    if u, v := q.Dims(); u != n || v != m {
      return nil, nil, fmt.Errorf("q has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  if r == nil {
    r = NullDenseMatrix(t, n, m)
  } else {
    if u, v := r.Dims(); u != n || v != m {
      return nil, nil, fmt.Errorf("r has invalid dimension (%dx%d instead of %dx%d)", u, v, n, m)
    }
  }
  return qrAlgorithm(a, b, q, r, t)
}
