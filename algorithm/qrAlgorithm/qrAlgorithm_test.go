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

package qrAlgorithm

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "testing"

import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func TestRProp(t *testing.T) {
  a := NewDenseMatrix(RealType, 4, 4, []float64{
    1, 2,  3, 4,
    4, 4,  4, 4,
    0, 1, -1, 1,
    0, 0,  2, 3 })

  h, _ := Run(a)

  fmt.Println("H:",h)
  
  // q, r, _ := Run(a)

  // r1 := NewDenseMatrix(RealType, 2, 2, []float64{
  //   1, 0,
  //   0, 1})
  // r2 := NewDenseMatrix(RealType, 2, 2, []float64{
  //   4, -1,
  //   0,  1})

  // if Mnorm(MsubM(r1, q)).GetValue() > 1e-8 {
  //   t.Error("QR-Algorithm failed!")
  // }
  // if Mnorm(MsubM(r2, r)).GetValue() > 1e-8 {
  //   t.Error("QR-Algorithm failed!")
  // }
}
