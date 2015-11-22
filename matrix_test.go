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

func TestMatrixMxV(t *testing.T) {

  m1 := NewMatrix(2, 2, []float64{1,2,3,4})
  v1 := NewVector([]float64{1, 2})
  v2 := MxV(m1, v1)
  v3 := NewVector([]float64{5, 11})

  if VNorm(VSub(v2, v3)).Value() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixVxM(t *testing.T) {

  m1 := NewMatrix(2, 2, []float64{1,2,3,4})
  v1 := NewVector([]float64{1, 2})
  v2 := VxM(v1, m1)
  v3 := NewVector([]float64{7, 10})

  if VNorm(VSub(v2, v3)).Value() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixInverse(t *testing.T) {

  m1 := NewMatrix(2, 2, []float64{1,2,3,4})
  m2 := MInverse(m1)
  m3 := NewMatrix(2, 2, []float64{-2, 1, 1.5, -0.5})

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestMatrixJacobian(t *testing.T) {

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := MakeVector(3)
    // x1^2 + y^2 - 6
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewScalar(6))
    // x^3 - y^2
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))
    y[2] = NewScalar(2)

    return y
  }

  v1 := NewVector([]float64{1,1})
  m1 := Jacobian(f, v1)
  m2 := NewMatrix(3, 2, []float64{2, 2, 3, -2, 0, 0})

  if MNorm(MSub(m1, m2)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestMatrixNewton(t *testing.T) {

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := MakeVector(2)
    // y1 = x1^2 + x2^2 - 6
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewScalar(6))
    // y2 = x1^3 - x2^2
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))

    return y
  }

  v1    := NewVector([]float64{1,1})
  v2, _ := Newton(f, v1, 1e-8)
  v3    := NewVector([]float64{1.537656, 1.906728})

  if VNorm(VSub(v2, v3)).Value() > 1e-8  {
    t.Error("Newton method failed!")
  }
}
