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

/* -------------------------------------------------------------------------- */

// this allows to idenfity the type of a scalar
type ScalarType reflect.Type

type ScalarState interface {
  Copy          (Scalar)
  // read access
  Order         ()             int
  Value         ()             float64
  LogValue      ()             float64
  Derivative    (int, int)     float64
  N             ()             int
  // write access
  Set           (Scalar)
  SetValue      (float64)
  SetDerivative (int, int, float64)
  SetVariable   (int, int, int)
}

type Scalar interface {
  ScalarState
  Clone     ()        Scalar
  // type reflections
  Type      ()        ScalarType
  // some basic operations on scalars
  Equals    (Scalar)  bool
  Greater   (Scalar)  bool
  Smaller   (Scalar)  bool
  Neg       ()        Scalar
  Add       (Scalar)  Scalar
  Sub       (Scalar)  Scalar
  Mul       (Scalar)  Scalar
  Div       (Scalar)  Scalar
  Pow       (Scalar)  Scalar
  Sqrt      ()        Scalar
  Sin       ()        Scalar
  Sinh      ()        Scalar
  Cos       ()        Scalar
  Cosh      ()        Scalar
  Tan       ()        Scalar
  Tanh      ()        Scalar
  Exp       ()        Scalar
  Log       ()        Scalar
  // nice printing
  fmt.Stringer
}

/* keep a map of valid scalar implementations and a reference
 * to the constructors
 * -------------------------------------------------------------------------- */

type rtype map[ScalarType]func(float64) Scalar

// initialize empty registry
var registry rtype = make(rtype)

// scalar types can be registered so that the constructors below can be used for
// all types
func RegisterScalar(t ScalarType, constructor func(float64) Scalar) {
  registry[t] = constructor
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewScalar(t ScalarType, value float64) Scalar {
  f, ok := registry[t]
  if !ok {
    panic("NewScalar(): Invalid scalar type!")
  }
  return f(value)
}

func ZeroScalar(t ScalarType) Scalar {
  f, ok := registry[t]
  if !ok {
    panic("NewScalar(): Invalid scalar type!")
  }
  return f(0.0)
}

/* -------------------------------------------------------------------------- */

func Variables(order int, reals ...Scalar) {
  for i, _ := range reals {
    reals[i].SetVariable(i, len(reals), order)
  }
}
