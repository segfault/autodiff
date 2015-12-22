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

import   "testing"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestMatrixInverse(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  m2 := Run(m1)
  m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestSubmatrixInverse(t *testing.T) {

  // exclude the third row/column
  submatrix := []bool{true, true, false}

  m1 := NewMatrix(RealType, 3, 3, []float64{1,2,50,3,4,60,70,80,90})
  m2 := Run(m1, submatrix)
  m3 := NewMatrix(RealType, 3, 3, []float64{-2, 1, 0, 1.5, -0.5, 0, 0, 0, 1})

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}
