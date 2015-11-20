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

package scalar

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

type Objective func([]*Scalar) *Scalar

/* -------------------------------------------------------------------------- */

func GradientDescent(f Objective, variables []*Scalar, step, epsilon float64) {

  var s *Scalar

  for {
    // initialize all variables as constants
    for _, v := range variables {
      v.Constant()
    }
    // compute partial derivatives and update variables
    for _, v := range variables {
      v.Variable()
      s = f(variables)

      // update variable
      *v = *Sub(v, NewScalar(step*s.Derivative()))
      // reset derivative
      v.Constant()
    }
    // compute total derivative
    for _, v := range variables {
      v.Variable()
    }
    s = f(variables)
    // evaluate stop criterion
    if (math.Abs(s.Derivative()) < epsilon) {
      break;
    }
  }
}
