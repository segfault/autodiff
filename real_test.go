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

//import "fmt"
import "math"
import "testing"

/* -------------------------------------------------------------------------- */

func TestReal(t *testing.T) {

  a := NewReal(1.0)

  if a.Value() != 1.0 {
    t.Error("a.Value() should be 1.0")
  }

  a.Add(a, NewReal(2.0))

  if a.Value() != 3.0 {
    t.Error("a.Value() should be 3.0")
  }
}

func TestDiff1(t *testing.T) {

  f := func(x Scalar) Scalar {
    return Add(Mul(NewReal(2), Pow(x, NewBareReal(3))), NewReal(4))
  }
  x := NewReal(9)

  Variables(2, x)

  y := f(x)

  if y.Derivative(1, 0) != 486 {
    t.Error("Differentiation failed!")
  }
  if y.Derivative(2, 0) != 108 {
    t.Error("Differentiation failed!")
  }
}

func TestDiff2(t *testing.T) {

  f := func(x Scalar) Scalar {
    y := x.Clone()
    y.Pow(y, NewBareReal(3))
    y.Mul(y, NewReal(2))
    y.Add(y, NewReal(4))
    return y
  }
  x := NewReal(9)

  Variables(2, x)

  y := f(x)

  if y.Derivative(1, 0) != 486 {
    t.Error("Differentiation failed!")
  }
  if y.Derivative(2, 0) != 108 {
    t.Error("Differentiation failed!")
  }
}

func TestMul(t *testing.T) {

  a := NewReal(13.123)
  b := NewReal( 4.321)

  Variables(2, a)

  a.Mul(a, a) // a^2
  a.Mul(a, a) // a^4
  a.Mul(a, b) // a^4 b

  if math.Abs(a.Value() - 128149.4603376) > 1e-4 {
    t.Error("Multiplication failed!")
  }
  if math.Abs(a.Derivative(1, 0) - 39061.025783) > 1e-4 {
    t.Error("Differentiation failed!")
  }
  if math.Abs(a.Derivative(2, 0) - 8929.5951649) > 1e-4 {
    t.Error("Differentiation failed!")
  }
}

func TestPow1(t *testing.T) {
  x := NewReal(3.4)
  k := NewReal(4.1)

  Variables(2, x, k)

  r := Pow(x, k)

  if math.Abs(r.Derivative(1, 0) - 182.124553) > 1e-4  ||
    (math.Abs(r.Derivative(1, 1) - 184.826947) > 1e-4) {
    t.Error("Pow failed!")
  }
  if math.Abs(r.Derivative(2, 0) - 166.054739) > 1e-4  ||
    (math.Abs(r.Derivative(2, 1) - 226.186676) > 1e-4) {
    t.Error("Pow failed!")
  }
}

func TestPow2(t *testing.T) {
  x := NewReal(-3.4)
  k := NewReal( 4.0)

  Variables(2, x, k)

  r := Pow(x, k)

  if math.Abs(r.Derivative(1, 0) - -157.216) > 1e-4  ||
    (math.Abs(r.Derivative(2, 0) -  138.720) > 1e-4) {
    t.Error("Pow failed!")
  }
  if !math.IsNaN(r.Derivative(1, 1))  ||
    (!math.IsNaN(r.Derivative(2, 1))) {
    t.Error("Pow failed!")
  }
}

func TestTan(t *testing.T) {

  a := NewReal(4.321)
  Variables(1, a)

  s := Tan(a)

  if math.Abs(s.Derivative(1, 0) - 6.87184) > 0.0001 {
    t.Error("Incorrect derivative for Tan()!", s.Derivative(1, 0))
  }
}

func TestTanh1(t *testing.T) {

  a := NewReal(4.321)
  Variables(2, a)

  s := Tanh(a)

  if math.Abs(s.Derivative(1, 0) -  0.00070588) > 0.0000001 {
    t.Error("Incorrect derivative for Tanh()!")
  }
  if math.Abs(s.Derivative(2, 0) - -0.00141127) > 0.0000001 {
    t.Error("Incorrect derivative for Tanh()!")
  }
}

func TestTanh2(t *testing.T) {

  a := NewReal(4.321)
  Variables(2, a)

  a.Tanh(a)

  if math.Abs(a.Derivative(1, 0) -  0.00070588) > 0.0000001 {
    t.Error("Incorrect derivative for Tanh()!")
  }
  if math.Abs(a.Derivative(2, 0) - -0.00141127) > 0.0000001 {
    t.Error("Incorrect derivative for Tanh()!")
  }
}

func TestErf(t *testing.T) {

  a := NewReal(0.23)
  Variables(2, a)

  s := Erf(a)

  if math.Abs(s.Derivative(1, 0) -  1.07023926) > 1e-6 ||
    (math.Abs(s.Derivative(2, 0) - -0.49231006) > 1e-6) {
    t.Error("Incorrect derivative for Erf()!")
  }
}

func TestErfc(t *testing.T) {

  a := NewReal(0.23)
  Variables(2, a)

  s := Erfc(a)

  if math.Abs(s.Derivative(1, 0) - -1.07023926) > 1e-6 ||
    (math.Abs(s.Derivative(2, 0) -  0.49231006) > 1e-6) {
    t.Error("Incorrect derivative for Erfc()!")
  }
}

func TestLogErfc1(t *testing.T) {

  a := NewReal(0.23)
  Variables(2, a)

  s := LogErfc(a)

  if math.Abs(s.Derivative(1, 0) - -1.436606354) > 1e-6 {
    t.Error("Incorrect derivative for LogErfc()!")
  }
  if math.Abs(s.Derivative(2, 0) - -1.402998894) > 1e-6 {
    t.Error("Incorrect derivative for LogErfc()!")
  }
}

func TestLogErfc2(t *testing.T) {

  a := NewReal(0.23)
  Variables(2, a)

  a.LogErfc(a)

  if math.Abs(a.Derivative(1, 0) - -1.436606354) > 1e-6 {
    t.Error("Incorrect derivative for LogErfc()!")
  }
  if math.Abs(a.Derivative(2, 0) - -1.402998894) > 1e-6 {
    t.Error("Incorrect derivative for LogErfc()!")
  }
}

func TestGamma(t *testing.T) {

  a := NewReal(4.321)
  Variables(2, a)

  s := Gamma(a)

  if math.Abs(s.Derivative(1, 0) - 12.2353264) > 1e-6 ||
    (math.Abs(s.Derivative(2, 0) - 18.8065398) > 1e-6) {
    t.Error("Incorrect derivative for Gamma()!")
  }
}
