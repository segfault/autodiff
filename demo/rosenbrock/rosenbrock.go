/* Copyright (C) 2017 Philipp Benner
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

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "os"

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/bfgs"


/* -------------------------------------------------------------------------- */

func main() {

  fp, err := os.Create("rosenbrock.table")
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

  x0 := NewVector(RealType, []float64{-0.5, 2})
  bfgs.Run(f, x0,
    bfgs.Hook{hook},
    bfgs.Epsilon{1e-10})

}
