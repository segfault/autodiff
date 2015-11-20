
package scalar

/* -------------------------------------------------------------------------- */

import "fmt"

/* -------------------------------------------------------------------------- */

type Scalar struct {
  value      float64
  derivative float64
}

func NewScalar(v float64) *Scalar {
  s := new(Scalar)
  s.value      = v
  s.derivative = 0
  return s
}

func (a *Scalar) Value() float64 {
  return a.value
}

func (a *Scalar) Derivative() float64 {
  return a.derivative
}

func (a *Scalar) Differentiate() {
  a.derivative = 1
}

func (a *Scalar) Reset() {
  a.derivative = 0
}

func (a *Scalar) Assign(v float64) {
  a.value      = v
  a.derivative = 0
}

func (a Scalar) String() string {
  return fmt.Sprintf("<%f,%f>", a.Value(), a.Derivative())
}
