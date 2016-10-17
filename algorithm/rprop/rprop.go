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

import   "fmt"
import   "math"
import   "errors"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/algorithm"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func([]float64, []float64, Vector, Scalar) bool
}

/* -------------------------------------------------------------------------- */

/* Resilient Backpropagation:
 * M. Riedmiller und H. Braun: Rprop - A Fast Adaptive Learning Algorithm.
 * Proceedings of the International Symposium on Computer and Information Science VII, 1992
 */

func rprop(f func(Vector) (Scalar, error), x0 Vector, step_init float64 , eta []float64, epsilon float64,
  hook func([]float64, []float64, Vector, Scalar) bool) (Vector, error) {

  n := len(x0)
  t := x0.ElementType()
  // copy variables
  x1 := x0.Clone()
  x2 := x0.Clone()
  // step size for each variable
  step := make([]float64, n)
  // gradients
  gradient_new := make([]float64, n)
  gradient_old := make([]float64, n)
  // initialize values
  for i, _ := range x1 {
    step[i]         = step_init
    gradient_new[i] = 1
    gradient_old[i] = 1
  }
  x1.Variables(1)

  gradient_is_nan := func(s Scalar) bool {
    for i := 0; i < s.GetN(); i++ {
      if math.IsNaN(s.GetDerivative(1, i)) {
        return true
      }
    }
    return false
  }
  // evaluate objective function
  s, err := f(x1)
  if err != nil || gradient_is_nan(s) {
    return x1, fmt.Errorf("invalid initial value: %v", x1)
  }
  for {
    for i, _ := range x1 {
      gradient_old[i] = gradient_new[i]
    }
    // compute partial derivatives and update x
    for i, _ := range x1 {
      // save derivative
      gradient_new[i] = s.GetDerivative(1, i)
    }
    // execute hook if available
    if hook != nil && hook(gradient_new, step, x1, s) {
      break;
    }
    // evaluate stop criterion
    if (Norm(gradient_new) < epsilon) {
      break;
    }
    // update step size
    for i, _ := range x1 {
      if gradient_new[i] != 0.0 {
        if ((gradient_old[i] < 0 && gradient_new[i] < 0) ||
            (gradient_old[i] > 0 && gradient_new[i] > 0)) {
          step[i] *= eta[0]
        } else {
          step[i] *= eta[1]
        }
      }
    }
    for {
      // update x
      for i, _ := range x1 {
        if gradient_new[i] != 0.0 {
          if gradient_new[i] > 0.0 {
            x2[i].Sub(x1[i], NewScalar(t, step[i]))
          } else {
            x2[i].Add(x1[i], NewScalar(t, step[i]))
          }
        }
        if math.IsNaN(x2[i].GetValue()) {
          return x2, errors.New("Gradient descent diverged!")
        }
      }
      // evaluate objective function
      s, err = f(x2)
      if err != nil || gradient_is_nan(s) {
        // if the updated is invalid reduce step size
        for i, _ := range x1 {
          if gradient_new[i] != 0.0 {
            step[i] *= eta[1]
          }
        }
      } else {
        // new position is valid, exit loop
        break
      }
    }
    x1.Copy(x2)
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

func Run(f func(Vector) (Scalar, error), x0 Vector, step_init float64, eta []float64, args ...interface{}) (Vector, error) {

  hook    := Hook   { nil}.Value
  epsilon := Epsilon{1e-8}.Value

  if len(eta) != 2 {
    panic("Rprop(): Argument eta must have length two!")
  }

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
