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

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestMatrix(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()

  if m1.At(1,2).GetValue() != m2.At(2,1).GetValue() {
    t.Error("Matrix transpose failed!")
  }
}

func TestMatrixRowCol(t *testing.T) {

  m  := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  v1 := m.Row(1)
  v2 := m.Col(2)

  v1.Set(NewReal(100), 0)
  v2.Set(NewReal(200), 0)

  if m.At(1,0).GetValue() != 100 {
    t.Error("Matrix Row() test failed!")
  }
  if m.At(0,2).GetValue() != 200 {
    t.Error("Matrix Col() test failed!")
  }
  if len(v1) != 3 {
    t.Error("Matrix Row() test failed!")
  }
  if len(v2) != 2 {
    t.Error("Matrix Col() test failed!")
  }
}

func TestMatrixDiag(t *testing.T) {

  m := NewDenseMatrix(RealType, 3, 3, []float64{1,2,3,4,5,6,7,8,9})
  v := m.Diag()

  if v[0].GetValue() != 1 ||
     v[1].GetValue() != 5 ||
     v[2].GetValue() != 9 {
    t.Error("Matrix diag failed!")
  }
}

func TestMatrixReference(t *testing.T) {

  m := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  c := NewReal(163)

  m.Set(c, 0, 0)
  c.SetValue(400)

  if m.At(0,0).GetValue() != 163 {
    t.Error("Matrix transpose failed!")
  }

  m.Set(Sub(m.At(1,2), Mul(m.At(1,1), m.At(1,2))),
    1, 2)

  if m.At(1,2).GetValue() != -24 {
    t.Error("Matrix transpose failed!")
  }
}

func TestSubmatrix(t *testing.T) {

  a := NewDenseMatrix(RealType, 3, 3, []float64{1,2,3,4,5,6,7,8,9})
  m := a.Submatrix(1,2,0,1)
  r := NewDenseMatrix(RealType, 2, 2, []float64{4,5,7,8})

  if Mnorm(MsubM(r, m)).GetValue() > 1e-8  {
    t.Error("Submatrix failed!")
  }
}

func TestMatrixTrace(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 2, []float64{1,2,3,4})

  if Mtrace(m1).GetValue() != 5 {
    t.Error("Wrong matrix trace!")
  }
}

func TestMatrixDot(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  m2 := m1.T()
  m3 := MdotM(m1, m2)

  if m3.At(0,0).GetValue() != 14 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixMul(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})
  m2 := NewDenseMatrix(RealType, 2, 3, []float64{6,5,4,3,2,1})
  m3 := MmulM(m1, m2)

  if m3.At(0,0).GetValue() != 6 {
    t.Error("Matrix multiplication failed!")
  }
}

func TestMatrixMdotV(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 2, []float64{1,2,3,4})
  v1 := NewVector(RealType, []float64{1, 2})
  v2 := MdotV(m1, v1)
  v3 := NewVector(RealType, []float64{5, 11})

  if Vnorm(VsubV(v2, v3)).GetValue() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixVdotM(t *testing.T) {

  m1 := NewDenseMatrix(RealType, 2, 2, []float64{1,2,3,4})
  v1 := NewVector(RealType, []float64{1, 2})
  v2 := VdotM(v1, m1)
  v3 := NewVector(RealType, []float64{7, 10})

  if Vnorm(VsubV(v2, v3)).GetValue() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
}

func TestMatrixMapReduce(t *testing.T) {

  r1 := NewDenseMatrix(RealType, 2, 2, []float64{2.718282e+00, 7.389056e+00, 2.008554e+01, 5.459815e+01})
  r2 := 84.79103
  m := NewDenseMatrix(RealType, 2, 2, []float64{1, 2,3,4})
  m.Map(Exp)
  a := m.Reduce(Add)

  if Mnorm(MsubM(m, r1)).GetValue() > 1e-8  {
    t.Error("Matrix/Vector multiplication failed!")
  }
  if math.Abs(a.GetValue() - r2) > 1e-2 {
    t.Error("Vector map/reduce failed!")
  }
}

func TestOuter(t *testing.T) {
  a := NewVector(RealType, []float64{1,3,2})
  b := NewVector(RealType, []float64{2,1,0,3})
  r := Outer(a,b)
  m := NewDenseMatrix(RealType, 3, 4, []float64{
    2,1,0,3,
    6,3,0,9,
    4,2,0,6 })

  if Mnorm(MsubM(r, m)).GetValue() > 1e-8  {
    t.Error("Outer product multiplication failed!")
  }

}

func TestMatrixJacobian(t *testing.T) {

  f := func(x Vector) Vector {
    if len(x) != 2 {
      panic("Invalid input vector!")
    }
    y := NullVector(RealType, 3)
    // x1^2 + y^2 - 6
    y[0] = Sub(Add(Pow(x[0], NewBareReal(2)), Pow(x[1], NewBareReal(2))), NewBareReal(6))
    // x^3 - y^2
    y[1] = Sub(Pow(x[0], NewBareReal(3)), Pow(x[1], NewBareReal(2)))
    y[2] = NewReal(2)

    return y
  }

  v1 := NewVector(RealType, []float64{1,1})
  m1 := Jacobian(f, v1)
  m2 := NewDenseMatrix(RealType, 3, 2, []float64{2, 2, 3, -2, 0, 0})

  if Mnorm(MsubM(m1, m2)).GetValue() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

func TestReadMatrix(t *testing.T) {

  m, err := ReadMatrix(RealType, "matrix_test.table")
  if err != nil {
    panic(err)
  }
  r := NewDenseMatrix(RealType, 2, 3, []float64{1,2,3,4,5,6})

  if Mnorm(MsubM(m, r)).GetValue() != 0.0 {
    t.Error("Read matrix failed!")
  }
}
