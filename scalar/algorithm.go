
package scalar

/* -------------------------------------------------------------------------- */

//import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

type Objective func([]*Scalar) *Scalar

/* -------------------------------------------------------------------------- */

func GradientDescent(f Objective, variables []*Scalar, step, epsilon float64) {

  var s *Scalar

  for {
    // compute partial derivatives and update variables
    for _, v := range variables {
      v.Differentiate()
      s = f(variables)

      // update variable
      *v = *Sub(v, NewScalar(step*s.Derivative()))
      // reset derivative
      v.Reset()
    }
    // compute total derivative
    for _, v := range variables {
      v.Differentiate()
    }
    s = f(variables)
    for _, v := range variables {
      v.Reset()
    }
    // evaluate stop criterion
    if (math.Abs(s.Derivative()) < epsilon) {
      break;
    }
  }
}
