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

//import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
  m1 := NewMatrix(RealType, 2, 2, []float64{1,2,3,4})
  m2 := m1.Clone()
  m3 := NewMatrix(RealType, 2, 2, []float64{-2, 1, 1.5, -0.5})

  rows, cols := m1.Dims()
  if rows != cols {
    panic("MInverse(): Not a square matrix!")
  }
  I := IdentityMatrix(m1.ElementType(), rows)
  // objective function
  f := func(x Vector) Scalar {
    m2.SetValues(x)
    s := MNorm(MSub(MMul(m1, m2), I))
    return s
  }
  x := Run(f, m2.Values(), 0.01, 0.1)
  m2.SetValues(x)

  if MNorm(MSub(m2, m3)).Value() > 1e-8 {
    t.Error("Inverting matrix failed!")
  }
}