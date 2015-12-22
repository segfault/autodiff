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

package algorithm

/* -------------------------------------------------------------------------- */

import   "math"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func SetConstant(variables Vector) {
  for i, _ := range variables {
    variables.Constant(i)
  }
}

func SetVariable(variables Vector, order int) {
  for i, _ := range variables {
    variables.Variable(order, i)
  }
}

func Norm(v []float64) float64 {
  sum := 0.0
  for _, x := range v {
    sum += math.Pow(x, 2.0)
  }
  return math.Sqrt(sum)
}
