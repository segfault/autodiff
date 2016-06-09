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

package rprop

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "os"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  m2 := m1.Clone()
  m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

  rows, cols := m1.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := IdentityMatrix(m1.ElementType(), rows)
  // objective function
  f := func(x Vector) (Scalar, error) {
    m2.SetValues(x)
    s := Mnorm(MsubM(MdotM(m1, m2), I))
    return s, nil
  }
  x, _ := Run(f, m2.GetValues(), 0.01, []float64{2, 0.1})
  m2.SetValues(x)

  if Mnorm(MsubM(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}

/* -------------------------------------------------------------------------- */

func TestRPropRosenbrock(t *testing.T) {

  fp, err := os.Create("rprop_test.table")
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
  hook := func(gradient []float64, step []float64, x Vector, value Scalar) bool {
    fmt.Fprintf(fp, "%s\n", x.Table())
    return false
  }

  x0 := NewVector(RealType, []float64{-10,10})
  xr := NewVector(RealType, []float64{  1, 1})
  xn, _ := Run(f, x0, 0.01, []float64{1.2, 0.8},
    Hook{hook},
    Epsilon{1e-10})

  if Vnorm(VsubV(xn, xr)).Value() > 1e-8 {
    t.Error("Rosenbrock test failed!")
  }
}
