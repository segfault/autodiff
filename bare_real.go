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

import "fmt"
import "reflect"
import "math"

/* -------------------------------------------------------------------------- */

type BareReal float64

/* register scalar type
 * -------------------------------------------------------------------------- */

var BareRealType ScalarType = NewBareReal(0.0).Type()

func init() {
  f := func(value float64) Scalar { return NewBareReal(value) }
  RegisterScalar(BareRealType, f)
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewBareReal(v float64) *BareReal {
  r := BareReal(v)
  return &r
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Clone() Scalar {
  return NewBareReal(float64(*a))
}

func (a *BareReal) Type() ScalarType {
  return reflect.TypeOf(a)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a *BareReal) String() string {
  return fmt.Sprintf("%e", a.GetValue())
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Copy(b Scalar) {
  *a = BareReal(b.GetValue())
}

func (a *BareReal) Alloc(n int) {
}

func (c *BareReal) AllocForOne(a Scalar) {
}

func (c *BareReal) AllocForTwo(a, b Scalar) {
}

/* read access
 * -------------------------------------------------------------------------- */

func (a *BareReal) GetOrder() int {
  return 0
}

func (a *BareReal) GetValue() float64 {
  return float64(*a)
}

func (a *BareReal) GetLogValue() float64 {
  return math.Log(a.GetValue())
}

func (a *BareReal) GetDerivative(i, j int) float64 {
  return 0.0
}

func (a *BareReal) GetN() int {
  return 0
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BareReal) Reset() {
  *a = 0.0
}

func (a *BareReal) Set(b Scalar) {
  *a = BareReal(b.GetValue())
}

func (a *BareReal) SetValue(v float64) {
  *a = BareReal(v)
}

func (a *BareReal) SetDerivative(i, j int, v float64) {
}

func (a *BareReal) SetVariable(i, n, order int) {
}
