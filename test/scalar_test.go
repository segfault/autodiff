package test

/* -------------------------------------------------------------------------- */

//import   "fmt"
import   "testing"
import . "github.com/pbenner/autodiff/scalar"

/* -------------------------------------------------------------------------- */

func TestScalar(t *testing.T) {

  a := NewScalar(1.0)

  if a.Value() != 1.0 {
    t.Error("a.Value() should be 1.0")
  }

}
