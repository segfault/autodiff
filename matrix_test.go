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

  v := NewVector([]float64{1,2,3,4,5,6})

  if v[1].Value() != 2.0 {
    t.Error("Vector initialization failed!")
  }
}

func TestMatrix(t *testing.T) {

  m1 := NewMatrix(2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()

  if m1.At(1,2) != m2.At(2,1) {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixTrace(t *testing.T) {

  m1 := NewMatrix(2, 2, []float64{1,2,3,4})

  if MTrace(m1).Value() != 5 {
    t.Error("Wrong matrix trace!")
  }
}

func TestMatrixMul(t *testing.T) {

  m1 := NewMatrix(2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()
  m3 := MMul(m1, m2)

  if m3.At(0,0) != 14 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixInverse(t *testing.T) {

  m1 := NewMatrix(2, 2, []float64{1,2,3,4})
  m2 := Inverse(m1)
  m3 := NewMatrix(2, 2, []float64{-2, 1, 1.5, -0.5})

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}
