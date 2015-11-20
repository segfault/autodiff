
package main

/* -------------------------------------------------------------------------- */

import   "fmt"
//import   "math"
import . "github.com/pbenner/autodiff/scalar"
import . "github.com/pbenner/autodiff/regression/line"


/* -------------------------------------------------------------------------- */

func sumOfSquares(x, y []*Scalar, l *Line) *Scalar {

  s := NewScalar(0)

  for i,_ := range x {
    s = Add(s, Pow(Sub(l.Eval(x[i]), y[i]), 2))
  }
  return s
}

func gradientDescent(x, y []*Scalar, l *Line) *Line {

  // gradient step size
  const epsilon = 0.000001
  const step    = 0.01

  // get a list of the variables
  variables := []*Scalar{l.Slope(), l.Intercept()}

  // create the objective function
  f := func(v []*Scalar) *Scalar {
    return sumOfSquares(x, y, l)
  }
  GradientDescent(f, variables, step, epsilon)

  return l
}

func regression() {

  x := []*Scalar{
    NewScalar(1),
    NewScalar(2),
    NewScalar(3),
    NewScalar(4),
    NewScalar(5),
    NewScalar(6)}
  y := []*Scalar{
    NewScalar(1),
    NewScalar(2),
    NewScalar(3),
    NewScalar(4),
    NewScalar(5),
    NewScalar(6)}

  l := NewLine(NewScalar(-1.23), NewScalar(1));

  l = gradientDescent(x, y, l)

  fmt.Println("slope: ", l.Slope().Value(), "b: ", l.Intercept().Value())

}

func main() {

  regression()

}
