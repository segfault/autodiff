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

package matrixInverse

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/gaussJordan"
import   "github.com/pbenner/autodiff/algorithm/rprop"

/* -------------------------------------------------------------------------- */

// compute the inverse of a matrix with a
// gradient descent method
func mInverse(matrix Matrix) Matrix {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := IdentityMatrix(matrix.ElementType(), rows)
  r := matrix.Clone()
  // objective function
  f := func(x Vector) Scalar {
    r.SetValues(x)
    s := MNorm(MSub(MMul(matrix, r), I))
    return s
  }
  x := rprop.Run(f, r.Values(), 0.01, 0.1)
  r.SetValues(x)
  return r
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) Matrix {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  if rows == 0 {
    panic("MInverse(): Empty matrix!")
  }
  t := matrix.ElementType()
  a := matrix.Clone()
  x := IdentityMatrix(t, rows)
  b := NullVector(t, rows) 
  // initialize b with ones
  for i, _ := range b {
    b[i] = NewScalar(t, 1.0)
  }
  // call Gauss-Jordan algorithm
  gaussJordan.Run(a, x, b, args...)
  return x
}
