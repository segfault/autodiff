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

import   "testing"
import   "errors"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestNewton(t *testing.T) {

  f := func(x Vector) (Vector, error) {
    if len(x) != 2 {
      return nil, errors.New("Invalid input vector!")
    }
    y := NullVector(RealType, 2)
    // y1 = x1^2 + x2^2 - 6
    y[0] = Sub(Add(Pow(x[0], NewReal(2)), Pow(x[1], NewReal(2))), NewReal(6))
    // y2 = x1^3 - x2^2
    y[1] = Sub(Pow(x[0], NewReal(3)), Pow(x[1], NewReal(2)))

    return y, nil
  }
  v1 := NewVector(RealType, []float64{1,1})
  v2, _ := Run(f, v1, Epsilon{1e-8})
  v3 := NewVector(RealType, []float64{1.537656, 1.906728})

  if Vnorm(VsubV(v2, v3)).Value() > 1e-6  {
    t.Error("Newton method failed!")
  }
}
