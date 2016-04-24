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

package msqrtInv

/* -------------------------------------------------------------------------- */

//import   "fmt"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

// Sherif, Nagwa. "On the computation of a matrix inverse square root."
// Computing 46.4 (1991): 295-305.

func mSqrtInv(matrix Matrix) Matrix {
  n, _ := matrix.Dims()
  c  := NewScalar(matrix.ElementType(), 2.0)
  A  := matrix
  I  := IdentityMatrix(matrix.ElementType(), n)
  X0 := IdentityMatrix(matrix.ElementType(), n)
  X1 := MmulS(MmulM(X0, matrixInverse.Run(MaddM(I, MmulM(A, MmulM(X0, X0))))), c)
  for Mnorm(MsubM(X0, X1)).Value() > 1e-8 {
    X0 = X1
    X1 = MmulS(MmulM(X0, matrixInverse.Run(MaddM(I, MmulM(A, MmulM(X0, X0))))), c)
  }
  return X1
}

/* -------------------------------------------------------------------------- */

func Run(matrix Matrix, args ...interface{}) Matrix {
  rows, cols := matrix.Dims()
  if rows != cols {
    panic("MSqrtInv(): Not a square matrix!")
  }
  if rows == 0 {
    panic("MSqrtInv(): Empty matrix!")
  }
  return mSqrtInv(matrix)
}
