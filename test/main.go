
package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

const epsilon float64 = 0.01

type line struct {
  A ADReal
  B ADReal
}

func (l *line) eval(x ADReal) ADReal {

    return Add(Mul(l.A, x), l.B)

}

func sumOfSquares(x, y []float64, l line) ADReal {

  s := ADReal{0, 0}

  for i,_ := range x {

    xx := ADReal{x[i], 0}
    yy := ADReal{y[i], 0}
    ll := l.eval(xx)

    s = Add(s, Pow(Sub(ll, yy), 2))

  }
  return s
}

func gradientDescent(x, y []float64, l line) line {

  var s ADReal

  for {

    l.A.Deriv = 1
    l.B.Deriv = 0
    s = sumOfSquares(x, y, l)
    l.A.Value = l.A.Value - epsilon*s.Deriv

    l.A.Deriv = 0
    l.B.Deriv = 1
    s = sumOfSquares(x, y, l)
    l.B.Value = l.B.Value - epsilon*s.Deriv

    l.A.Deriv = 1
    l.B.Deriv = 1
    s = sumOfSquares(x, y, l)
    fmt.Println(String(s))
    if (math.Abs(s.Deriv) < epsilon) {
      break
    }
  }
  return l
}

func regression() {

  x := []float64{1,2,3,4,5,6}
  y := []float64{1,2,3,4,5,6}

  l := line{
    ADReal{-1.23, 1},
    ADReal{0, 1}}

  l = gradientDescent(x, y, l)

  fmt.Println("a: ", l.A.Value, "b: ", l.B.Value)

}

func main() {

  regression()

}
