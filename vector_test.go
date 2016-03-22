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

import "testing"

/* -------------------------------------------------------------------------- */

func TestVector(t *testing.T) {

  v := NewVector(RealType, []float64{1,2,3,4,5,6})

  if v[1].Value() != 2.0 {
    t.Error("Vector initialization failed!")
  }
}

func TestVectorToMatrix(t *testing.T) {

  v := NewVector(RealType, []float64{1,2,3,4,5,6})
  m := v.Matrix(2, 3)

  if m.At(1,0).Value() != 4 {
    t.Error("Vector to matrix conversion failed!")
  }
}

func TestVxV(t *testing.T) {

  a := NewVector(RealType, []float64{1, 2,3,4})
  b := NewVector(RealType, []float64{2,-1,1,7})
  r := VxV(a, b)

  if r.Value() != 31 {
    t.Error("VxV() failed!")
  }
}
