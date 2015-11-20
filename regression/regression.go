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

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math/rand"
import . "github.com/pbenner/autodiff/scalar"
import . "github.com/pbenner/autodiff/regression/line"


/* -------------------------------------------------------------------------- */

func sumOfSquares(x, y []*Scalar, l *Line) *Scalar {

  s := NewScalar(0)
  n := NewScalar(float64(len(x)))

  for i,_ := range x {
    s = Add(s, Pow(Sub(l.Eval(x[i]), y[i]), 2))
  }
  return Div(s, n)
}

func gradientDescent(x, y []*Scalar, l *Line) *Line {

  // gradient step size
  const epsilon = 0.00001
  const step    = 0.1

  // get a list of the variables
  variables := []*Scalar{l.Slope(), l.Intercept()}

  // create the objective function
  f := func(v []*Scalar) *Scalar {
    return sumOfSquares(x, y, l)
  }
//  GradientDescent(f, variables, step, epsilon)
  Rprop(f, variables, step, epsilon, 0.2)

  return l
}

func regression() {

  const n = 1000
  x := make([]*Scalar, n)
  y := make([]*Scalar, n)

  // random number generator
  r := rand.New(rand.NewSource(42))

  for i := 0; i < n; i++ {
    x[i] = NewScalar(r.NormFloat64() + 0)
    y[i] = NewScalar(r.NormFloat64() + 2*x[i].Value()+1)
  }

  l := NewLine(NewScalar(-1.23), NewScalar(1));
  l  = gradientDescent(x, y, l)

  fmt.Println("slope: ", l.Slope().Value(), "b: ", l.Intercept().Value())
}

func main() {

  regression()

}
