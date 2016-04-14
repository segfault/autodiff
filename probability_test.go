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

func TestProbability(t *testing.T) {

  a := NewProbability(2)

  if math.Abs(a.LogValue() - math.Log(2.0)) > 1e-8 {
    t.Error("a.Value() should be 1.0")
  }
}

func TestProbabilityNeg(t *testing.T) {

  p := NewProbability(2)
  r := Neg(p)

  if !Equal(r, NewReal(-2)) {
    t.Error("Negation test failed")
  }
}

func TestProbabilityAdd(t *testing.T) {

  var p1, p2 Scalar

  v1 := []float64{2,  2,  0,  2, -1, -1}
  v2 := []float64{2,  0,  2, -3,  1,  2}
  v3 := []float64{4,  2,  2, -1,  0,  1}

  for i, _ := range v1 {
    if v1[i] > 0.0 {
      p1 = NewProbability(v1[i])
    } else {
      p1 = NewReal(v1[i])
    }
    if v2[i] > 0.0 {
      p2 = NewProbability(v2[i])
    } else {
      p2 = NewReal(v2[i])
    }
    r := Add(p1, p2)

    if !Equal(r, NewReal(v3[i])) {
      t.Error("Addition test", i, " (1) failed")
    }
    if  Smaller(r, NewReal(0.0)) && r.Type() != RealType {
      t.Error("Addition test", i, " (2) failed")
    }
  }
}

func TestProbabilityMul(t *testing.T) {

  v1 := []float64{2,   2}
  v2 := []float64{2, 0.5}
  v3 := []float64{4,   1}

  for i, _ := range v1 {
    p1 := NewProbability(v1[i])
    p2 := NewProbability(v2[i])

    r := Mul(p1, p2).(*Probability)

    if !Equal(r, NewProbability(v3[i])) {
      t.Error("Multiplication test", i, "failed")
    }
  }
}

func TestProbabilityDiv(t *testing.T) {

  v1 := []float64{4, 0}
  v2 := []float64{2, 2}
  v3 := []float64{2, 0}

  for i, _ := range v1 {
    p1 := NewProbability(v1[i])
    p2 := NewProbability(v2[i])

    r := Div(p1, p2)

    if !Equal(r, NewProbability(v3[i])) {
      t.Error("Division test", i, "failed")
    }
  }
}

func TestProbabilityLog(t *testing.T) {

  v1 := []float64{1.0, 2.0, 3.0}

  for i, _ := range v1 {

    p := NewProbability(v1[i])
    r := Log(p)

    if !Equal(r, NewReal(math.Log(v1[i]))) {
      t.Error("Log test", i, "failed")
    }
  }
}

func TestProbabilityExp(t *testing.T) {

  v1 := []float64{1.0, 2.0, 0.2}

  for i, _ := range v1 {

    p := NewProbability(v1[i])
    r := Exp(p)

    if !Equal(r, NewProbability(math.Exp(v1[i]))) {
      t.Error("Exp test", i, "failed")
    }
  }
}

func TestProbabilityDiff1(t *testing.T) {

  {
    a := NewProbability(2.0)
    b := NewProbability(3.0)
    Variables(1, a)

    c := Add(a, b)

    if math.Abs(c.Derivative(1, 0) - 1.0) > 1e-8 {
      t.Error("Probability differentiation test 1 failed")
    }
  }
  {
    a := NewProbability(2.0)
    b := NewProbability(3.0)
    Variables(1, a)

    c := Mul(a, b)

    if math.Abs(c.Derivative(1, 0) - 3.0) > 1e-8 {
      t.Error("Probability differentiation test 2 failed")
    }
  }
  {
    a := NewProbability(2.0)
    Variables(1, a)

    c := Neg(Mul(a, a))

    if math.Abs(c.Derivative(1, 0) + 4.0) > 1e-8 {
      t.Error("Probability differentiation test 3 failed")
    }
  }
  {

    a := NewProbability(2.0)
    Variables(1, a)

    c := Pow(a, NewBareReal(13))

    if math.Abs(c.Derivative(1, 0) - 13*math.Pow(2.0, 12)) > 1e-8 {
      t.Error("Probability differentiation test 4 failed")
    }
  }
  {
    a := NewProbability(2.0)
    b := NewProbability(3.0)
    Variables(1, b)

    c := Div(a, b)

    if math.Abs(c.Derivative(1, 0) + 2.0/math.Pow(3.0, 2)) > 1e-8 {
      t.Error("Probability differentiation test 5 failed")
    }
  }
}

func TestTypes(t *testing.T) {

  a := NewProbability(2)
  b := NewReal(3)

  c1 := Add(a,b)
  c2 := Add(b,a)

  if c1.Type() != ProbabilityType {
    t.Error("TestTypes() failed!")
  }
  if c2.Type() != RealType {
    t.Error("TestTypes() failed!")
  }
}

func TestMultinomialLikelihood(t *testing.T) {
  likelihood := func(theta Scalar, c1, c2 float64) Scalar {
    theta1 := theta
    theta2 := Sub(NewProbability(1), theta)
    return Mul(Pow(theta1, NewBareReal(c1)), Pow(theta2, NewBareReal(c2)))
  }
  // evaluate the likelihood at the mode
  theta := NewProbability(13.0/(13.0+17.0))
  l := likelihood(theta, 13, 17)

  Variables(2, theta)

  // first derivative at the mode should be zero
  if math.Abs(l.Derivative(1, 0)) > 1e-12 {
    t.Error("l.Derivative(1) should be 0.0")
  }
  // second derivative at the mode should be negative
  if l.Derivative(2, 0) > 0.0 {
    t.Error("l.Derivative(2) should be negative")
  }
}

func TestEntropy(t *testing.T) {
  entropy := func(theta Scalar) Scalar {
    theta1 := theta
    theta2 := Sub(NewProbability(1), theta)
    c1 := Mul(theta1, Log(theta1))
    c2 := Mul(theta2, Log(theta2))
    return Neg(Add(c1, c2))
  }
  // evaluate the likelihood at the mode
  theta := NewProbability(1.0/2.0)
  e := entropy(theta)

  Variables(2, theta)

  // first derivative at the mode should be zero
  if math.Abs(e.Derivative(1, 0)) > 1e-12 {
    t.Error("l.Derivative(1) should be 0.0")
  }
  // second derivative at the mode should be negative
  if e.Derivative(2, 0) > 0.0 {
    t.Error("l.Derivative(2) should be negative")
  }
}
