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

package bfgs

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "os"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestBfgsMatyas(t *testing.T) {

  fp, err := os.Create("bfgs_test1.table")
  if err != nil {
    panic(err)
  }
  defer fp.Close()

  f := func(x Vector) (Scalar, error) {
    // f(x1, x2) = 0.26(x1^2 + x2^2) - 0.48 x1 x2
    // minimum: f(x1,x2) = f(0, 0) = 0
    y := Sub(Mul(NewReal(0.26), Add(Mul(x[0], x[0]), Mul(x[1], x[1]))),
      Mul(NewReal(0.48), Mul(x[0], x[1])))
    return y, nil
  }
  hook := func(gradient, x Vector, y Scalar) bool {
    fmt.Fprintf(fp, "%s\n", x.Table())
    fmt.Println("gradient:", gradient)
    fmt.Println("x       :", x)
    fmt.Println("y       :", y)
    return false
  }

  x0 := NewVector(RealType, []float64{-2.5,2})
  B0 := NewDenseMatrix(RealType, 2, 2, []float64{1.0, 0.0, 0.0, 1.0})
  xr := NewVector(RealType, []float64{0, 0})
  xn, err := Run(f, x0, B0,
    Hook{hook},
    Epsilon{1e-8})
  if err != nil {
    panic(err)
  }
  if Vnorm(VsubV(xn, xr)).GetValue() > 1e-6 {
    t.Error("BFGS Matyas test failed!")
  }
}

func TestBfgsRosenbrock(t *testing.T) {

  fp, err := os.Create("bfgs_test2.table")
  if err != nil {
    panic(err)
  }
  defer fp.Close()

  f := func(x Vector) (Scalar, error) {
    // f(x1, x2) = (a - x1)^2 + b(x2 - x1^2)^2
    // a = 1
    // b = 100
    // minimum: (x1,x2) = (a, a^2)
    a := NewReal(  1.0)
    b := NewReal(100.0)
    s := Pow(Sub(a, x[0]), NewReal(2.0))
    t := Mul(b, Pow(Sub(x[1], Mul(x[0], x[0])), NewReal(2.0)))
    return Add(s, t), nil
  }
  hook := func(gradient, x Vector, y Scalar) bool {
    fmt.Fprintf(fp, "%s\n", x.Table())
    fmt.Println("gradient:", gradient)
    fmt.Println("x       :", x)
    fmt.Println("y       :", y)
    return false
  }

  x0 := NewVector(RealType, []float64{-10,10})
  B0 := NewDenseMatrix(RealType, 2, 2, []float64{1.0, 0.0, 0.0, 1.0})
  xr := NewVector(RealType, []float64{  1, 1})
  xn, err := Run(f, x0, B0,
    Hook{hook},
    Epsilon{1e-10})
  if err != nil {
    panic(err)
  }
  if Vnorm(VsubV(xn, xr)).GetValue() > 1e-8 {
    t.Error("BFGS Rosenbrock test failed!")
  }
}
