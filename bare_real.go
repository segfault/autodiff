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
import "math"
import "reflect"

/* -------------------------------------------------------------------------- */

type BareReal struct {
  value float64
}

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
  return &BareReal{v}
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Clone() Scalar {
  return NewBareReal(a.value)
}

func (a *BareReal) Type() ScalarType {
  return reflect.TypeOf(a)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a *BareReal) String() string {
  return fmt.Sprintf("%e", a.Value())
}

/* -------------------------------------------------------------------------- */

func (a *BareReal) Copy(b Scalar) {
  a.value = b.Value()
}

/* read access
 * -------------------------------------------------------------------------- */

func (a *BareReal) Order() int {
  return 0
}

func (a *BareReal) Value() float64 {
  return a.value
}

func (a *BareReal) LogValue() float64 {
  return math.Log(a.Value())
}

func (a *BareReal) Derivative(i int) float64 {
  return 0.0
}

/* write access
 * -------------------------------------------------------------------------- */

func (a *BareReal) Set(b Scalar) {
  a.value = b.Value()
}

func (a *BareReal) SetValue(v float64) {
  a.value = v
}

func (a *BareReal) SetDerivative(i int, v float64) {
}

func (a *BareReal) Variable(order int) {
}

func (a *BareReal) Constant() {
}
