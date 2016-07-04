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
  Copy            (Scalar)
  // allocate memory for derivatives of n variables
  Alloc           (int)
  // allocate enough memory for the derivatives of the given
  // variable(s) and copy the order
  AllocForOne     (Scalar)
  AllocForTwo     (Scalar, Scalar)
  // read access
  GetOrder        ()             int
  GetValue        ()             float64
  GetLogValue     ()             float64
  GetDerivative   (int, int)     float64
  GetN            ()             int
  // set value and derivatives to zero
  Reset           ()
  ResetDerivatives()
  // write access
  Set             (Scalar)
  SetValue        (float64)
  SetDerivative   (int, int, float64)
  SetVariable     (int, int, int)
}

type Scalar interface {
  ScalarState
  Clone     ()                Scalar
  // type reflections
  Type      ()                ScalarType
  // some basic operations on scalars
  Equals    (Scalar)          bool
  Greater   (Scalar)          bool
  Smaller   (Scalar)          bool
  Negative  ()                bool
  Neg       (Scalar)          Scalar
  Add       (Scalar, Scalar)  Scalar
  Sub       (Scalar, Scalar)  Scalar
  Mul       (Scalar, Scalar)  Scalar
  Div       (Scalar, Scalar)  Scalar
  Pow       (Scalar, Scalar)  Scalar
  Sqrt      (Scalar)          Scalar
  Sin       (Scalar)          Scalar
  Sinh      (Scalar)          Scalar
  Cos       (Scalar)          Scalar
  Cosh      (Scalar)          Scalar
  Tan       (Scalar)          Scalar
  Tanh      (Scalar)          Scalar
  Exp       (Scalar)          Scalar
  Log       (Scalar)          Scalar
  Erf       (Scalar)          Scalar
  Erfc      (Scalar)          Scalar
  LogErfc   (Scalar)          Scalar
  Gamma     (Scalar)          Scalar
  Lgamma    (Scalar)          Scalar
  Mlgamma   (Scalar, int)     Scalar // multivariate log gamma
  // vector operations
  VdotV     (a, b Vector)     Scalar
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

func ScalarConstructor(t ScalarType) func(float64) Scalar {
  f, ok := registry[t]
  if !ok {
    panic("invalid scalar type")
  }
  return f
}

func NewScalar(t ScalarType, value float64) Scalar {
  f, ok := registry[t]
  if !ok {
    panic("invalid scalar type")
  }
  return f(value)
}

func NullScalar(t ScalarType) Scalar {
  f, ok := registry[t]
  if !ok {
    panic("invalid scalar type")
  }
  return f(0.0)
}

/* -------------------------------------------------------------------------- */

func Variables(order int, reals ...Scalar) {
  for i, _ := range reals {
    reals[i].SetVariable(i, len(reals), order)
  }
}
