
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
  GradientDescent(f, variables, step, epsilon)

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

  l = gradientDescent(x, y, l)

  fmt.Println("slope: ", l.Slope().Value(), "b: ", l.Intercept().Value())

}

func main() {

  regression()

}
