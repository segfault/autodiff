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

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

type Objective func(Vector) Scalar

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

func GradientDescent(f Objective, variables Vector, step, epsilon float64) {

  var s Scalar

  for {
    // initialize all variables as constants
    setConstant(variables)
    // compute partial derivatives and update variables
    for i, _ := range variables {
      variables[i].Variable()
      s = f(variables)

      // update variable
      variables[i] = Sub(variables[i], NewScalar(step*s.Derivative()))
      // reset derivative
      variables[i].Constant()
    }
    // compute total derivative
    setVariable(variables)
    s = f(variables)
    // evaluate stop criterion
    if (math.Abs(s.Derivative()) < epsilon) {
      break;
    }
  }
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func Rprop(f Objective, variables Vector, step_init, epsilon, eta float64) {

  var s Scalar
  // step size for each variable
  step := make([]float64, len(variables))
  // gradients
  gradient_new := make([]float64, len(variables))
  gradient_old := make([]float64, len(variables))
  // initialize values
  for i, _ := range variables {
    step[i]         = step_init
    gradient_new[i] = 1
    gradient_old[i] = 1
  }

  for {
    for i, _ := range variables {
      gradient_old[i] = gradient_new[i]
    }
    // initialize all variables as constants
    setConstant(variables)
    // compute partial derivatives and update variables
    for i, _ := range variables {
      // differentiate with respect to the ith variable
      variables[i].Variable()
      s = f(variables)
      // save derivative
      gradient_new[i] = s.Derivative()
      // set variable back to constant
      variables[i].Constant()
    }
    // update step size
    for i, _ := range variables {
      if ((gradient_old[i] < 0 && gradient_new[i] < 0) ||
          (gradient_old[i] > 0 && gradient_new[i] > 0)) {
        step[i] *= 1.0 + eta
      } else {
        step[i] *= 1.0 - eta
      }
    }
    // update variables
    for i, _ := range variables {
      if gradient_new[i] > 0.0 {
        variables[i] = Sub(variables[i], NewScalar(step[i]))
      } else {
        variables[i] = Add(variables[i], NewScalar(step[i]))
      }
    }
    // compute total derivative
    setVariable(variables)
    s = f(variables)
    // evaluate stop criterion
    if (math.Abs(s.Derivative()) < epsilon) {
      break;
    }
  }
}
