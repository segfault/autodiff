
package autodiff

/* -------------------------------------------------------------------------- */

import "fmt"
import "math"

/* -------------------------------------------------------------------------- */

type ADReal struct {
  Value float64
  Deriv float64
}

func String(a ADReal) string {
  return fmt.Sprintf("<%f,%f>", a.Value, a.Deriv)
}

/* -------------------------------------------------------------------------- */

func Add(a ADReal, b ADReal) ADReal {
  return ADReal{a.Value + b.Value, a.Deriv + b.Deriv}
}

func Sub(a ADReal, b ADReal) ADReal {
  return ADReal{a.Value - b.Value, a.Deriv - b.Deriv}
}

func Mul(a ADReal, b ADReal) ADReal {
  return ADReal{a.Value*b.Value, a.Value*b.Deriv + a.Deriv*b.Value}
}

func Div(a ADReal, b ADReal) ADReal {
  return ADReal{a.Value/b.Value, (a.Deriv*b.Value - a.Value*b.Deriv)/(b.Value*b.Value)}
}

/* -------------------------------------------------------------------------- */

func Sin(a ADReal) ADReal {
  return ADReal{math.Sin(a.Value), a.Deriv*math.Cos(a.Value)}
}

func Cos(a ADReal) ADReal {
  return ADReal{math.Cos(a.Value), -a.Deriv*math.Sin(a.Value)}
}

func Exp(a ADReal) ADReal {
  return ADReal{math.Exp(a.Value), a.Deriv*math.Exp(a.Value)}
}

func Log(a ADReal) ADReal {
  return ADReal{math.Log(a.Value), a.Deriv/a.Value}
}

func Pow(a ADReal, k float64) ADReal {
  return ADReal{math.Pow(a.Value, k), k*math.Pow(a.Value, k-1)*a.Deriv}
}
