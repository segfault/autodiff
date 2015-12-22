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

import   "math"
import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func([]float64, Vector, Scalar) bool
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func rprop(f func(Vector) Scalar, x0 Vector, step_init, eta, epsilon float64,
  hook func([]float64, Vector, Scalar) bool) Vector {

  var s Scalar
  t := x0.ElementType()
  // copy variables
  x := x0.Clone()
  // initialize all x as constants
  SetConstant(x)
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
    // compute partial derivatives and update x
    for i, _ := range x {
      // differentiate with respect to the ith variable
      x[i].Variable(1)
      s = f(x)
      // save derivative
      gradient_new[i] = s.Derivative(1)
      // set variable back to constant
      x[i].Constant()
    }
    // execute hook if available
    if hook != nil && hook(gradient_new, x, s) {
      break;
    }
    // evaluate stop criterion
    err := Norm(gradient_new)
    if (err < epsilon) {
      break;
    }
    // update step size
    for i, _ := range x {
      if gradient_new[i] != 0.0 {
        if ((gradient_old[i] < 0 && gradient_new[i] < 0) ||
            (gradient_old[i] > 0 && gradient_new[i] > 0)) {
          step[i] *= 1.0 + eta
        } else {
          step[i] *= 1.0 - eta
        }
      }
    }
    // update x
    for i, _ := range x {
      if gradient_new[i] != 0.0 {
        if gradient_new[i] > 0.0 {
          x[i] = Sub(x[i], NewScalar(t, step[i]))
        } else {
          x[i] = Add(x[i], NewScalar(t, step[i]))
        }
      }
      if math.IsNaN(x[i].Value()) {
        panic("Gradient descent diverged!")
      }
    }
  }
  return x
}

/* -------------------------------------------------------------------------- */

func Run(f func(Vector) Scalar, x0 Vector, step_init, eta float64, args ...interface{}) Vector {

  hook    := Hook   { nil}.Value
  epsilon := Epsilon{1e-8}.Value

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a.Value
    case Epsilon:
      epsilon = a.Value
    default:
      panic("Rprop(): Invalid optional argument!")
    }
  }
  return rprop(f, x0, step_init, eta, epsilon, hook)
}
