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

type Probability struct {
  BasicState
}

/* register scalar type
 * -------------------------------------------------------------------------- */

var ProbabilityType ScalarType = NewProbability(0.0).Type()

func init() {
  f := func(value float64) Scalar { return NewProbability(value) }
  RegisterScalar(ProbabilityType, f)
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewProbability(value float64) *Probability {
  if value < 0.0 {
    panic("NewProbability(): Value should be positive!")
  }
  return &Probability{*NewBasicState(math.Log(value))}
}

/* -------------------------------------------------------------------------- */

func (a *Probability) Clone() Scalar {
  r := NewProbability(0.0)
  r.Copy(a)
  return r
}

func (a *Probability) Value() float64 {
  return math.Exp(a.value)
}

func (a *Probability) LogValue() float64 {
  return a.value
}

func (a *Probability) Type() ScalarType {
  return reflect.TypeOf(a)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (a *Probability) Real() *Real {
  return NewReal(math.Exp(a.Value()))
}

func (a *Probability) String() string {
  switch a.order {
  case 0:
    return fmt.Sprintf("<exp(%e)>", math.Exp(a.LogValue()))
  case 1:
    return fmt.Sprintf("<exp(%e),%e>", math.Exp(a.LogValue()), a.Derivative(1))
  case 2:
    return fmt.Sprintf("<exp(%e),%e,%e>", math.Exp(a.LogValue()), a.Derivative(1), a.Derivative(2))
  default:
    panic("Invalid order!")
  }
}