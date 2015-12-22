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

package newton

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func(Matrix, Vector, Vector) bool
}

/* -------------------------------------------------------------------------- */

func newton(f func(Vector) Vector, x Vector, epsilon float64,
  hook func(Matrix, Vector, Vector) bool,
  options []interface{}) Vector {
  x1  := x.Clone()
  x2  := x.Clone()
  for {
    y  := f(x1)
    J  := Jacobian(f, x1)
    Q  := matrixInverse.Run(J, options...)
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

/* -------------------------------------------------------------------------- */

func Run(f func(Vector) Vector, x Vector, args ...interface{}) Vector {

  hook      := Hook     { nil}.Value
  epsilon   := Epsilon  {1e-8}.Value
  options   := make([]interface{}, 0)

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a.Value
    case Epsilon:
      epsilon = a.Value
    default:
      options = append(options, a)
    }
  }
  return newton(f, x, epsilon, hook, options)
}
