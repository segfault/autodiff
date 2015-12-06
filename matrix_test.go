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

func TestMatrix(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()

  if m1.At(1,2).Value() != m2.At(2,1).Value() {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixReference(t *testing.T) {

  m := NewMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  c := NewReal(163)

  m.Set(c, 0, 0)
  c.Set(400)

  if m.At(0,0).Value() != 163 {
    t.Error("Matrix transpose failed!")
  }

  m.Set(Sub(m.At(1,2), Mul(m.At(1,1), m.At(1,2))),
    1, 2)

  if m.At(1,2).Value() != -24 {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixTrace(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})

  if MTrace(m1).Value() != 5 {
    t.Error("Wrong matrix trace!")
  }
}

func TestMatrixMul(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()
  m3 := MMul(m1, m2)

  if m3.At(0,0).Value() != 14 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixMxV(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  v1 := NewVector(RealType, []float64{1, 2})
  v2 := MxV(m1, v1)
  v3 := NewVector(RealType, []float64{5, 11})

  if VNorm(VSub(v2, v3)).Value() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixVxM(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  v1 := NewVector(RealType, []float64{1, 2})
  v2 := VxM(v1, m1)
  v3 := NewVector(RealType, []float64{7, 10})

  if VNorm(VSub(v2, v3)).Value() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixInverse(t *testing.T) {

  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  m2 := MInverse(m1)
  m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestMatrixJacobian(t *testing.T) {

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := NullVector(RealType, 3)
    // x1^2 + y^2 - 6
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewReal(6))
    // x^3 - y^2
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))
    y[2] = NewReal(2)

    return y
  }

  v1 := NewVector(RealType, []float64{1,1})
  m1 := Jacobian(f, v1)
  m2 := NewMatrix(RealType, 3, 2, []float64{2, 2, 3, -2, 0, 0})

  if MNorm(MSub(m1, m2)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestMatrixNewton(t *testing.T) {

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := NullVector(RealType, 2)
    // y1 = x1^2 + x2^2 - 6
    y[0] = Sub(Add(Pow(x[0], 2), Pow(x[1], 2)), NewReal(6))
    // y2 = x1^3 - x2^2
    y[1] = Sub(Pow(x[0], 3), Pow(x[1], 2))

    return y
  }
  v1    := NewVector(RealType, []float64{1,1})
  v2, _ := Newton(f, v1, 1e-8)
  v3    := NewVector(RealType, []float64{1.537656, 1.906728})

  if VNorm(VSub(v2, v3)).Value() > 1e-6  {
    t.Error("Newton method failed!")
  }
}

func TestGaussJordan(t *testing.T) {
  n := 3
  a := NewMatrix(RealType, n, n, []float64{1, 1, 1, 2, 1, 1, 1, 2, 1})
  x := IdentityMatrix(RealType, n)
  b := NewVector(RealType, []float64{1,1,1})
  r := NewMatrix(RealType, n, n, []float64{-1, 1, 0, -1, 0, 1, 3, -1, -1})

  GaussJordan(a, x, b)

  if !MEqual(x, r)  {
    t.Error("Gauss-Jordan method failed!")
  }
}
