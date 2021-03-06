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

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type Matrix interface {
  ScalarContainer
  // matrix access with improved performance
  At              (i, j int)           Scalar
  ReferenceAt     (i, j int)           Scalar
  Set             (v Scalar, i, j int)
  SetReference    (v Scalar, i, j int)
  // set to identity matrix
  SetIdentity     ()
  Reset           ()
  ResetDerivatives()
  // basic methods
  Clone           ()                   Matrix
  CloneShallow    ()                   Matrix
  Copy            (Matrix)
  Dims            ()                   (int, int)
  GetValues       ()                   Vector
  SetValues       (v Vector)
  Row             (i int)              Vector
  Col             (j int)              Vector
  Diag            ()                   Vector
  T               ()                   Matrix
  Table           ()                   string
  WriteMatrix     (string)             error
  Submatrix       (rfrom, rto, cfrom, cto int) Matrix
  PermuteRows     ([]int)
  Reshape         (int, int)           error
  // math operations
  MaddM(a, b Matrix) Matrix
  MaddS(a Matrix, b Scalar) Matrix
  MsubM(a, b Matrix) Matrix
  MsubS(a Matrix, b Scalar) Matrix
  MmulM(a, b Matrix) Matrix
  MmulS(a Matrix, b Scalar) Matrix
  MdivM(a, b Matrix) Matrix
  MdivS(a Matrix, b Scalar) Matrix
  MdotM(a, b Matrix) Matrix
  Outer(a, b Vector) Matrix
  Jacobian(f func(Vector) Vector, x_ Vector) Matrix
}
