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
import . "github.com/pbenner/autodiff"

import   "github.com/gonum/plot"
import   "github.com/gonum/plot/plotter"
import   "github.com/gonum/plot/plotutil"
import   "github.com/gonum/plot/vg"


/* -------------------------------------------------------------------------- */

func plotGradientNorm(gn1, gn2 []float64) {
  xy1 := make(plotter.XYs, len(gn1))
  xy2 := make(plotter.XYs, len(gn2))

  for i := 0; i < len(gn1); i++ {
    xy1[i].X = float64(i)
    xy1[i].Y = gn1[i]
  }
  for i := 0; i < len(gn2); i++ {
    xy2[i].X = float64(i)
    xy2[i].Y = gn2[i]
  }

  p, err := plot.New()
  if err != nil {
    panic(err)
  }
  p.Title.Text = "Squared norm of the gradient"
  p.X.Label.Text = "interation"
  p.Y.Label.Text = "||·||^2"
  p.Y.Scale = plot.LogScale{}
  p.Y.Tick.Marker = plot.LogTicks{}
  p.Legend.Top = true

  err = plotutil.AddLines(p,
    "vanilla", xy1,
    "rprop",   xy2)
  if err != nil {
    panic(err)
  }

  if err := p.Save(8*vg.Inch, 4*vg.Inch, "example1.png"); err != nil {
    panic(err)
  }

}


func main() {
  f := func(x Vector) Scalar {
    // x^4 - 3x^3 + 2
    return Add(Sub(Pow(x[0], 4), Mul(NewConstant(3), Pow(x[0], 3))), NewConstant(2))
  }
  x0 := NewVector([]float64{8})
  // vanilla gradient descent
  xn1, err1 := GradientDescent(f, x0, 0.0001, 1e-8)
  // resilient backpropagation
  xn2, err2 := Rprop(f, x0, 0.0001, 1e-8, 0.4)

  fmt.Println(xn1)
  fmt.Println(xn2)

  plotGradientNorm(err1, err2)
}
