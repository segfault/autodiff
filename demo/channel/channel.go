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

import   "math"
import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/demo/blahut"

import   "github.com/gonum/plot"
import   "github.com/gonum/plot/plotter"
import   "github.com/gonum/plot/plotutil"
import   "github.com/gonum/plot/vg"

/* plot methods
 * -------------------------------------------------------------------------- */

type line struct {
  values []float64
  name   string
}

func plotGradientNorm(args ...line) {
  var list []interface{}

  for _, arg := range args {
    xy := make(plotter.XYs, len(arg.values))
    for i := 0; i < len(arg.values); i++ {
      xy[i].X = float64(i)+1
      xy[i].Y = arg.values[i]
    }
    list = append(list, arg.name)
    list = append(list, xy)
  }

  p, err := plot.New()
  if err != nil {
    panic(err)
  }
  p.Title.Text = ""
  p.X.Label.Text = "iteration"
  p.Y.Label.Text = "distance to optimum"
  p.X.Scale = plot.LogScale{}
  p.Y.Scale = plot.LogScale{}
  p.X.Tick.Marker = plot.LogTicks{}
  p.Y.Tick.Marker = plot.LogTicks{}
  p.Legend.Top = true

  err = plotutil.AddLines(p, list...)
  if err != nil {
    panic(err)
  }

  if err := p.Save(8*vg.Inch, 4*vg.Inch, "channel.png"); err != nil {
    panic(err)
  }

}

/* utility
 * -------------------------------------------------------------------------- */

func flatten(m [][]float64) []float64 {
  v := []float64{}
  for i, _ := range m {
    v = append(v, m[i]...)
  }
  return v
}

func distance(v1, v2 []float64) float64 {
  sum := 0.0
  for i, _ := range v1 {
    sum += math.Pow(v1[i]-v2[i], 2.0)
  }
  return math.Sqrt(sum)
}

/* hooks for keeping track of convergence speed
 * -------------------------------------------------------------------------- */

func hook_g(trace *[]float64, pxstar []float64, gradient []float64, variables Vector, s Scalar) bool {
  n := (len(variables) - 1)/2
  // split variables
  px := variables[0:n]
  // distance to optimum
  d := distance(px.Slice(), pxstar)
  // append result to trace
  *trace = append(*trace, d)

  return false
}

func hook_f(trace *[]float64, pxstar []float64, gradient Matrix, variables Vector, s Vector) bool {
  n := (len(variables) - 1)/2
  // split variables
  px := variables[0:n]
  // distance to optimum
  d := distance(px.Slice(), pxstar)
  // append result to trace
  *trace = append(*trace, d)

  return false
}

func hook_b(trace *[]float64, pxstar []float64, px []float64) bool {
  // distance to optimum
  d := distance(px, pxstar)
  // append result to trace
  *trace = append(*trace, d)

  return false
}

/* objective functions for gradient based maximization
 * -------------------------------------------------------------------------- */

func objective_f(active []bool, channel Matrix, variables Vector) Vector {
  n, m := channel.Dims()
  if len(variables) != 2*n+1 {
    panic("Input vector has invalid dimension!")
  }
  // split variables
  px     := variables[0:n]
  lambda := variables[n:2*n+1]
  // check if constraints need to be activated or deactivated
  for i := 0; i < n; i++ {
    // activate constraint
    if px[i].Value() < 0.0 {
      active[i] = true
    }
    // deactivate constraint
    if lambda[i].Value() > 0.0 {
      active[i] = false
    }
    if active[i] {
      px[i].SetValue(0.0)
    } else {
      lambda[i].SetValue(0.0)
    }
  }
  // compute p(y) from p(y|x)*p(x)
  py := NullVector(RealType, m)
  for j := 0; j < m; j++ {
    py[j] = NewReal(0.0)
    for i := 0; i < n; i++ {
      py[j] = Add(py[j], Mul(channel.At(i, j), px[i]))
    }
  }
  gradient := NullVector(RealType, 2*n+1)
  // derivative with respect to px[i]
  for i := 0; i < n; i++ {
    // -1
    gradient[i] = NewReal(-1.0)
    // -lambda_i
    if active[i] {
      gradient[i] = Sub(gradient[i], lambda[i])
    }
    // +lambda_n
    gradient[i] = Add(gradient[i], lambda[n])
    for j := 0; j < m; j++ {
      // t = p(y|x)
      t := channel.At(i, j)
      // p(x|y) log p(x|y)/p(y) - lambda
      if t.Value() > 0.0 {
        gradient[i] = Add(gradient[i], Mul(t, Log(Div(t, py[j]))))
      }
    }
  }
  // derivative with respect to lambda_i
  for i := 0; i < n; i++ {
    if active[i] {
      gradient[n+i] = Neg(px[i])
    } else {
      gradient[n+i] = NewReal(0.0)
    }
  }
  // derivative with respect to lambda_n
  gradient[2*n] = NewReal(-1.0)
  for i := 0; i < n; i++ {
    gradient[2*n] = Add(gradient[2*n], px[i])
  }
  return gradient
}

func objective_g(active []bool, channel Matrix, px Vector) Scalar {
  return Pow(VNorm(objective_f(active, channel, px)), 2.0)
}

/* main function
 * -------------------------------------------------------------------------- */

func channel_capacity(channel [][]float64, pxstar, px0 []float64) ([][]float64) {
  n := len(channel)
  m := len(channel[0])
  // precision
  const epsilon = 1e-12
  // initial gradient step size
  const step    = 0.001

  // copy variables for automatic differentation
  channelm := NewMatrix(RealType, n, m, flatten(channel))
  // add n+1 lagrange multipliers
  px0m     := NewVector(RealType, append(px0, make([]float64, n+1)...))

  // for Newton's method we need to specify the active set of constraints
  // (i.e. if a constraint is deactivated the respective rows and columns
  //  need to be removed from the Jacobian before computing the inverse)
  submatrix := make([]bool, 2*n+1)
  // initialize submatrix (all rows/columns that belong to inequality constraints
  // are excluded initially
  for i := 0; i < n; i++ {
    submatrix[i] = true
  }
  submatrix[2*n] = true

  // active constaints
  active1 := make([]bool, n)
  active2 := submatrix[n:2*n]

  // keep track of the path of an algorithm
  trace := make([][]float64, 4)
  trace[0] = []float64{distance(px0, pxstar)}
  trace[1] = []float64{distance(px0, pxstar)}
  trace[2] = []float64{distance(px0, pxstar)}
  trace[3] = []float64{distance(px0, pxstar)}

  // hooks
  hook1 := func(gradient []float64, variables Vector, s Scalar) bool {
    return hook_g(&trace[0], pxstar, gradient, variables, s)
  }
  hook2 := func(gradient Matrix, variables Vector, s Vector) bool {
    return hook_f(&trace[1], pxstar, gradient, variables, s)
  }
  hook3 := func(px []float64, J float64) bool {
    return hook_b(&trace[2], pxstar, px)
  }
  hook4 := func(px []float64, J float64) bool {
    return hook_b(&trace[3], pxstar, px)
  }

  // objective functions
  f := func(px Vector) Vector { return objective_f(active2, channelm, px) }
  g := func(px Vector) Scalar { return objective_g(active1, channelm, px) }

  // execute algorithms
  Rprop (g, px0m, epsilon, step, 0.01, hook1)
  Newton(f, px0m, epsilon, hook2, submatrix)
  BlahutNaive(channel, px0, 500, HookNaive{hook3}, Lambda{1.0})
  BlahutNaive(channel, px0,  40, HookNaive{hook4}, Lambda{8.0})

  return trace
}

func main() {

  // channel := [][]float64{
  //   {2.0/3.0, 1.0/3.0,     0.0},
  //   {1.0/3.0, 1.0/3.0, 1.0/3.0},
  //   {    0.0, 1.0/3.0, 2.0/3.0} }
  channel := [][]float64{
    {0.60, 0.30, 0.10},
    {0.70, 0.10, 0.20},
    {0.50, 0.05, 0.45} }
  pxstar  := []float64{0.501735, 0.0, 0.498265}

  // initial value
  px0 := []float64{1.0/3.0, 1.0/3.0, 1.0/3.0}

  trace := channel_capacity(channel, pxstar, px0)

  plotGradientNorm(
    line{trace[0], "Rprop"},
    line{trace[1], "Newton"},
    line{trace[2], "Blahut"},
    line{trace[3], "PP-Blahut"})
}
