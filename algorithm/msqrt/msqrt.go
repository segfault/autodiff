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

package msqrt

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

// Denma-Beavers algorithm (not guaranteed to converge!)
// Other methods rely on the Schur decomposition, see:
// Higham, N.~J. (2008). Functions of Matrices: Theory and Computation;
// Society for Industrial and Applied Mathematics, Philadelphia, PA, USA.

func mSqrt(matrix Matrix) Matrix {
  n, _ := matrix.Dims()
  c  := NewScalar(matrix.ElementType(), 0.5)
  Y0 := matrix
  Z0 := IdentityMatrix(matrix.ElementType(), n)
  Y1 := MmulS(MaddM(Y0, matrixInverse.Run(Z0)), c)
  Z1 := MmulS(MaddM(Z0, matrixInverse.Run(Y0)), c)
  for Mnorm(MsubM(Y0, Y1)).Value() > 1e-8 {
    Y0 = Y1
    Z0 = Z1
    Y1 = MmulS(MaddM(Y0, matrixInverse.Run(Z0)), c)
    Z1 = MmulS(MaddM(Z0, matrixInverse.Run(Y0)), c)
  }
  return Y1
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) Matrix {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("MSqrt(): Not a square matrix!")
  }
  if rows == 0 {
    panic("MSqrt(): Empty matrix!")
  }
  return mSqrt(matrix)
}
