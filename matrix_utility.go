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

func (matrix *DenseMatrix) T() Matrix {
  return &DenseMatrix{
    Values    :  matrix.Values,
    Rows      :  matrix.Cols,
    Cols      :  matrix.Rows,
    Transposed: !matrix.Transposed}
}

func (matrix *DenseMatrix) PermuteRows(_p []int) {
  if len(_p) != matrix.Rows {
    panic("PermuteRows(): permutation vector has invalid length!")
  }
  // make a copy of _p
  p := make([]int, len(_p))
  copy(p, _p)
  // permute matrix
  for i := 0; i < matrix.Rows; i++ {
    if i != p[i] {
      for j := 0; j < matrix.Cols; j++ {
        v1 := matrix.At(  i , j)
        v2 := matrix.At(p[i], j)
        matrix.Set(v2,   i , j)
        matrix.Set(v1, p[i], j)
      }
      // save permutation
      for k := 0; k < matrix.Rows; k++ {
        if p[k] == i {
          p[k], p[i] = p[i], i
          break
        }
      }
    }
  }
}
