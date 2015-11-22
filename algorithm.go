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

import "math"

/* -------------------------------------------------------------------------- */

func setConstant(variables Vector) {
    for i, _ := range variables {
      variables[i].Constant()
    }
}

func setVariable(variables Vector) {
    for i, _ := range variables {
      variables[i].Variable()
    }
}

/* -------------------------------------------------------------------------- */

func GradientDescent(f func(Vector) Scalar, x0 Vector, step, epsilon float64) (Vector, []float64) {

  var s   Scalar
  var err []float64
  // copy variables
  x := x0.Clone()

  for {
    // initialize all variables as constants
    setConstant(x)
    // compute partial derivatives and update variables
    for i, _ := range x {
      x[i].Variable()
      s = f(x)

      // update variable
      x[i] = Sub(x[i], NewScalar(step*s.Derivative()))
      // reset derivative
      x[i].Constant()
      if math.IsNaN(x[i].Value()) {
        panic("Gradient descent diverged!")
      }
    }
    // compute total derivative
    setVariable(x)
    s = f(x)
    // evaluate stop criterion
    err = append(err, math.Abs(s.Derivative()))
    if (err[len(err)-1] < epsilon) {
      setConstant(x)
      break;
    }
  }
  return x, err
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func Rprop(f func(Vector) Scalar, x0 Vector, step_init, epsilon, eta float64) (Vector, []float64) {

  var s   Scalar
  var err []float64
  // copy variables
  x := x0.Clone()
  // step size for each variable
  step := make([]float64, len(x))
  // gradients
  gradient_new := make([]float64, len(x))
  gradient_old := make([]float64, len(x))
  // initialize values
  for i, _ := range x {
    step[i]         = step_init
    gradient_new[i] = 1
    gradient_old[i] = 1
  }

  for {
    for i, _ := range x {
      gradient_old[i] = gradient_new[i]
    }
    // initialize all x as constants
    setConstant(x)
    // compute partial derivatives and update x
    for i, _ := range x {
      // differentiate with respect to the ith variable
      x[i].Variable()
      s = f(x)
      // save derivative
      gradient_new[i] = s.Derivative()
      // set variable back to constant
      x[i].Constant()
    }
    // update step size
    for i, _ := range x {
      if ((gradient_old[i] < 0 && gradient_new[i] < 0) ||
          (gradient_old[i] > 0 && gradient_new[i] > 0)) {
        step[i] *= 1.0 + eta
      } else {
        step[i] *= 1.0 - eta
      }
    }
    // update x
    for i, _ := range x {
      if gradient_new[i] > 0.0 {
        x[i] = Sub(x[i], NewScalar(step[i]))
      } else {
        x[i] = Add(x[i], NewScalar(step[i]))
      }
      if math.IsNaN(x[i].Value()) {
        panic("Gradient descent diverged!")
      }
    }
    // compute total derivative
    setVariable(x)
    s = f(x)
    // evaluate stop criterion
    err = append(err, math.Abs(s.Derivative()))
    if (err[len(err)-1] < epsilon) {
      setConstant(x)
      break;
    }
  }
  return x, err
}

/* -------------------------------------------------------------------------- */

func Newton(f func(Vector) Vector, x Vector, epsilon float64) (Vector, []float64) {
  x1  := x.Clone()
  x2  := x.Clone()
  err := []float64{}
  for {
    y  := f(x1)
    J  := Jacobian(f, x1)
    Q  := MInverse(J)
    x2  = VSub(x1, MxV(Q, y))
    err = append(err, VNorm(VSub(x1, x2)).Value())
    if err[len(err)-1] < epsilon {
      break
    }
    x1.CopyFrom(x2)
  }
  return x2, err
}
