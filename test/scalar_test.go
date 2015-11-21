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

package test

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "math"
import   "testing"
import . "github.com/pbenner/autodiff/scalar"

/* -------------------------------------------------------------------------- */

func TestScalar(t *testing.T) {

  a := NewScalar(1.0)

  if a.Value() != 1.0 {
    t.Error("a.Value() should be 1.0")
  }
}

func TestTan(t *testing.T) {

  a := NewVariable(4.321)
  s := Tan(a)

  if math.Abs(s.Derivative() - 6.87184) > 0.0001 {
    t.Error("Incorrect derivative for Tan()!", s.Derivative())
  }
}

func TestTanh(t *testing.T) {

  a := NewVariable(4.321)
  s := Tanh(a)

  if math.Abs(s.Derivative() - 0.00070588) > 0.0000001 {
    t.Error("Incorrect derivative for Tanh()!")
  }
}
