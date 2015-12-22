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

package gaussJordan

/* -------------------------------------------------------------------------- */

import   "testing"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestGaussJordan(t *testing.T) {
  n := 3
  a := NewMatrix(RealType, n, n, []float64{1, 1, 1, 2, 1, 1, 1, 2, 1})
  x := IdentityMatrix(RealType, n)
  b := NewVector(RealType, []float64{1,1,1})
  r := NewMatrix(RealType, n, n, []float64{-1, 1, 0, -1, 0, 1, 3, -1, -1})

  Run(a, x, b)

  if !MEqual(x, r)  {
    t.Error("Gauss-Jordan method failed!")
  }
}
