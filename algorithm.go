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
    variables.Constant(i)
  }
}

func setVariable(variables Vector, order int) {
  for i, _ := range variables {
    variables.Variable(order, i)
  }
}

func norm(v []float64) float64 {
  sum := 0.0
  for _, x := range v {
    sum += math.Pow(x, 2.0)
  }
  return math.Sqrt(sum)
}

/* -------------------------------------------------------------------------- */

func GradientDescent(f func(Vector) Scalar, x0 Vector, epsilon, step float64, args ...interface{}) Vector {

  var hook func([]float64, Vector, Scalar) bool = nil
  for _, arg := range args {
    switch a := arg.(type) {
    case func([]float64, Vector, Scalar) bool:
      hook = a
    default:
      panic("GradientDescent(): Invalid optional argument!")
    }
  }

  var s Scalar
  t := x0.ElementType()
  // copy variables
  x := x0.Clone()
  // initialize all variables as constants
  setConstant(x)
  // slice containing the gradient
  gradient := make([]float64, len(x))

  for {
    // compute partial derivatives and update variables
    for i, _ := range x {
      // differentiate once with respect to the ith variable
      x[i].Variable(1)
      s = f(x)
      // save partial derivative
      gradient[i] = s.Derivative(1)
      // set variables constant again
      x[i].Constant()
    }
    // execute hook if available
    if hook != nil && hook(gradient, x, s) {
      break;
    }
    // evaluate stop criterion
    err := norm(gradient)
    if (err < epsilon) {
      break;
    }
    // update variables
    for i, _ := range x {
      x[i] = Sub(x[i], NewScalar(t, step*s.Derivative(1)))
      if math.IsNaN(x[i].Value()) {
        panic("Gradient descent diverged!")
      }
    }
  }
  return x
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func Rprop(f func(Vector) Scalar, x0 Vector, epsilon, step_init, eta float64, args ...interface{}) Vector {

  var hook func([]float64, Vector, Scalar) bool = nil
  for _, arg := range args {
    switch a := arg.(type) {
    case func([]float64, Vector, Scalar) bool:
      hook = a
    default:
      panic("Rprop(): Invalid optional argument!")
    }
  }

  var s Scalar
  t := x0.ElementType()
  // copy variables
  x := x0.Clone()
  // initialize all x as constants
  setConstant(x)
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
    err := norm(gradient_new)
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

func newton(f func(Vector) Vector, x Vector, epsilon float64,
  hook func(Matrix, Vector, Vector) bool, submatrix []bool) Vector {
  x1  := x.Clone()
  x2  := x.Clone()
  for {
    y  := f(x1)
    J  := Jacobian(f, x1)
    Q  := MInverse(J, submatrix)
    x2  = VSub(x1, MxV(Q, y))
    // execute hook if available
    if hook != nil && hook(J, x2, y) {
      break;
    }
    // evaluate stop criterion
    err := VNorm(VSub(x1, x2)).Value()
    if err < epsilon {
      break
    }
    x1.Copy(x2)
  }
  return x2
}

func Newton(f func(Vector) Vector, x Vector, epsilon float64, args ...interface{}) Vector {

  var hook func(Matrix, Vector, Vector) bool = nil
  var submatrix []bool = nil

  for _, arg := range args {
    switch a := arg.(type) {
    case func(Matrix, Vector, Vector) bool:
      hook = a
    case []bool:
      submatrix = a
    default:
      panic("Newton(): Invalid optional argument!")
    }
  }

  return newton(f, x, epsilon, hook, submatrix)
}
