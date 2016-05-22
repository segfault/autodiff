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

package cholesky

/* -------------------------------------------------------------------------- */

//import   "fmt"

import   "testing"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestCholesky1(t *testing.T) {
  n := 4
  a := NewMatrix(RealType, n, n, []float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 })
  x, _ := Run(a)
  r    := NewMatrix(RealType, n, n, []float64{
     4.24264, 0.00000, 0.00000, 0.00000,
     5.18545, 6.56591, 0.00000, 0.00000,
    12.72792, 3.04604, 1.64974, 0.00000,
     9.89949, 1.62455, 1.84971, 1.39262 })

  if Mnorm(MsubM(x, r)).Value() > 1e-8 {
    t.Error("Cholesky failed!")
  }
}

func TestCholesky2(t *testing.T) {
  n := 4
  a := NewMatrix(RealType, n, n, []float64{
    18, 22,  54,  42,
    22, 70,  86,  62,
    54, 86, 174, 134,
    42, 62, 134, 106 })
  x, _ := Run(a, InSitu{true})
  r    := NewMatrix(RealType, n, n, []float64{
     4.24264, 0.00000, 0.00000, 0.00000,
     5.18545, 6.56591, 0.00000, 0.00000,
    12.72792, 3.04604, 1.64974, 0.00000,
     9.89949, 1.62455, 1.84971, 1.39262 })

  if Mnorm(MsubM(x, r)).Value() > 1e-8 {
    t.Error("Cholesky failed!")
  }
  if Mnorm(MsubM(x, a)).Value() > 1e-8 {
    t.Error("Cholesky failed!")
  }
}
