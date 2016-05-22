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

//import   "fmt"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/cholesky"
import   "github.com/pbenner/autodiff/algorithm/gaussJordan"
import   "github.com/pbenner/autodiff/algorithm/rprop"

/* -------------------------------------------------------------------------- */

type PositiveDefinite struct {
  Value bool
}

type InSitu struct {
  Value bool
}

/* -------------------------------------------------------------------------- */

// compute the inverse of a matrix with a
// gradient descent method
func mInverseGradient(matrix Matrix) (Matrix, error) {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := IdentityMatrix(matrix.ElementType(), rows)
  r := matrix.Clone()
  // objective function
  f := func(x Vector) (Scalar, error) {
    r.SetValues(x)
    s := Mnorm(MsubM(MdotM(matrix, r), I))
    return s, nil
  }
  x, _ := rprop.Run(f, r.Values(), 0.01, 0.1)
  r.SetValues(x)
  return r, nil
}

func mInverse(matrix Matrix, args ...interface{}) (Matrix, error) {
  rows, _ := matrix.Dims()
  t := matrix.ElementType()
  a := matrix.Clone()
  x := IdentityMatrix(t, rows)
  b := NullVector(t, rows)
  // initialize b with ones
  for i, _ := range b {
    b[i].SetValue(1.0)
  }
  // call Gauss-Jordan algorithm
  err := gaussJordan.Run(a, x, b, args...)
  return x, err
}

func mInversePD(matrix Matrix, s InSitu, args ...interface{}) (Matrix, error) {
  rows, _ := matrix.Dims()
  t := matrix.ElementType()
  a, err := cholesky.Run(matrix, cholesky.InSitu{s.Value})
  a = a.T()
  if err != nil {
    return nil, err
  }
  x := IdentityMatrix(t, rows)
  b := NullVector(t, rows)
  // initialize b with ones
  for i, _ := range b {
    b[i].SetValue(1.0)
  }
  args = append(args, gaussJordan.Triangular{true})
  // call Gauss-Jordan algorithm
  gaussJordan.Run(a, x, b, args...)
  // recycle a to store the result
  return a.MdotM(x, x.T()), nil
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) (Matrix, error) {
  rows, cols := matrix.Dims()
  gArgs := []interface{}{}
  if rows != cols {
    panic("not a square matrix")
  }
  if rows == 0 {
    panic("empty matrix")
  }
  positiveDefinite := false
  inSitu           := false

  // loop over optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case PositiveDefinite:
      positiveDefinite = a.Value
    case InSitu:
      inSitu = a.Value
    default:
      // all other arguments are passed to the
      // Gauss-Jordan algorithm
      gArgs = append(gArgs, arg)
    }
  }
  if positiveDefinite {
    return mInversePD(matrix, InSitu{inSitu}, gArgs...)
  } else {
    return mInverse(matrix, gArgs...)
  }
}
