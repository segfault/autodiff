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

package bfgs

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"

import . "github.com/pbenner/autodiff"
//import . "github.com/pbenner/autodiff/algorithm"
import   "github.com/pbenner/autodiff/algorithm/matrixInverse"

/* -------------------------------------------------------------------------- */

type Objective func(Vector) (Scalar, error)

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func(gradient, x Vector, y Scalar) bool
}

/* -------------------------------------------------------------------------- */

type ObjectiveInSitu struct {
  Eval func(x Vector, y Scalar) error
}

func newObjectiveInSitu(f Objective) ObjectiveInSitu {
  g := func(x Vector, y Scalar) error {
    z, err := f(x)
    if err != nil {
      return err
    }
    y.Copy(z)
    return nil
  }
  return ObjectiveInSitu{g}
}

func (f ObjectiveInSitu) Differentiate(x Vector, y Scalar, g Vector) error {
  x.Variables(1)  
  if err := f.Eval(x, y); err != nil {
    return err
  }
  copyGradient(y, g)
  x.ResetDerivatives()
  return nil
}

/* -------------------------------------------------------------------------- */

/* Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm:
 */

func copyGradient(y Scalar, g Vector) {
  for i := 0; i < y.GetN(); i++ {
    g[i].SetValue(y.GetDerivative(1, i))
  }
}

func bgfs_computeDirection(x Vector, y Scalar, g Vector, B Matrix, p Vector) {
  p.MdotV(B, g)
  for i := 0; i < len(x); i++ {
    p[i].Neg(p[i])
  }
}

func bgfs_backtrackingLineSearch(f ObjectiveInSitu, x1, x2 Vector, y1, y2 Scalar, g1, g2, p1, p2 Vector, a1 Vector) bool {
  c1  := 1e-3
  rho := NewReal(0.9)
  t1  := NullReal()
  // always begin with a = 1
  a1[0].SetValue(1.0)
  for {
    // compute x2 = f(x1 + a1 p1)
    p2.VmulS(p1, a1[0])
    x2.VaddV(x1, p2)
    // evaluate function at x2
    f.Differentiate(x2, y2, g2)
    // check NaN
    if math.IsNaN(y2.GetValue()) {
      return false
    }
    // check Wolfe conditions
    t1.VdotV(p1, g1)
    if y2.GetValue() <= y1.GetValue() + c1*a1[0].GetValue()*t1.GetValue() {
      break
    }
    a1[0].Mul(rho, a1[0])
  }
  a1.ResetDerivatives()
  x1.ResetDerivatives()
  x2.ResetDerivatives()
  y1.ResetDerivatives()
  y2.ResetDerivatives()
  g1.ResetDerivatives()
  g2.ResetDerivatives()
  p1.ResetDerivatives()
  p2.ResetDerivatives()
  return true
}

func bfgs_updateB(g1, g2, p2 Vector, B1, B2 Matrix, t1, t2 Vector, t3, t4 Matrix) {
  s := p2
  y := t1
  // y = Df(x2) - Df(x1)
  y.VsubV(g2, g1)
  // s.y
  t6 := VdotV(s, y)
  // check if value is zero
  if math.Abs(t6.GetValue()) < 1e-16 {
    B2.Copy(B1)
    return
  }
  // y s^T
  t3.Outer(y, s)
  // B y s^T
  t3.MdotM(B1, t3)
  // s y^T B
  t4.Outer(s, y)
  // s y^T B
  t4.MdotM(t4, B1)
  // B y s^T + s y^T B
  t3.MaddM(t3, t4)
  // (B y s^T + s y^T B) / (s.y)
  t3.MdivS(t3, t6)
  // save result
  B2.MsubM(B1, t3)
  // y.B
  t2.VdotM(y, B1)
  // y.B.y
  t5 := VdotV(t2, y)
  // s.y + y.B.y
  t5.Add(t5, t6)
  // (s.y)^2
  t6.Mul(t6, t6)
  // (s.y + y.B.y)/(s.y)^2
  t5.Div(t5, t6)
  // s s^T
  t3.Outer(s, s)
  // (s.y + y.B.y)/(s.y)^2 (s s^T)
  t3.MmulS(t3, t5)
  // save result
  B2.MaddM(B2, t3)
}

func bfgs(f ObjectiveInSitu, x0 Vector, B0 Matrix, epsilon Epsilon, hook Hook) (Vector, error) {

  n := len(x0)
  t := x0.ElementType()

  a1 := NewVector(t, []float64{1e-8})
  p1 := NullVector(t, n)
  p2 := NullVector(t, n)
  x1 := x0.Clone()
  x2 := NullVector(t, n)
  y1 := NullScalar(t)
  y2 := NullScalar(t)
  g1 := NullVector(t, n)
  g2 := NullVector(t, n)
  B1 := B0.Clone()
  B2 := NullDenseMatrix(t, n, n)
  // some temporary variables
  t1 := NullVector(t, n)
  t2 := NullVector(t, n)
  t3 := NullDenseMatrix(t, n, n)
  t4 := NullDenseMatrix(t, n, n)

  // evaluate objective function
  if err := f.Differentiate(x1, y1, g1); err != nil {
    return x1, fmt.Errorf("invalid initial value: %s", err)
  }
  for {
    // execute hook if available
    fmt.Println("x1      :", x1)
    fmt.Println("x2      :", x2)
    fmt.Println("g1      :", g1)
    if hook.Value != nil && hook.Value(g1, x1, y1) {
      break
    }
    // evaluate stop criterion
    if Vnorm(g1).GetValue() < epsilon.Value {
      break
    }
    bgfs_computeDirection(x1, y1, g1, B1, p1)
    fmt.Println("line search...")
    if ok := bgfs_backtrackingLineSearch(f, x1, x2, y1, y2, g1, g2, p1, p2, a1); !ok {
      return x1, fmt.Errorf("line search failed")
    }
    // evaluate objective at new position
    if err := f.Differentiate(x2, y2, g2); err != nil {
      return x1, fmt.Errorf("invalid value: %s", err)
    }
    if Vnorm(VsubV(x1, x2)).GetValue() < 1e-20 {
      return x2, nil
    }
    
    fmt.Println("x2:", x2)
    fmt.Println("y2:", y2)
    fmt.Println("g2:", g2)
    fmt.Println("a1:", a1)
    bfgs_updateB(g1, g2, p2, B1, B2, t1, t2, t3, t4)
    fmt.Println("B1:", B1)
    fmt.Println("B2:", B2)
    
    g1.Copy(g2)
    x1.Copy(x2)
    y1.Copy(y2)
    p1.Copy(p2)
    B1.Copy(B2)
    fmt.Println("iteration done...")
    fmt.Println()
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

func Run(f Objective, x0 Vector, B0 Matrix, args ...interface{}) (Vector, error) {

  hook    := Hook   { nil}
  epsilon := Epsilon{1e-8}

  n := len(x0)
  r, c := B0.Dims()
  if n != r || n != c {
    return nil, fmt.Errorf("argument dimensions do not match, i.e. x0 has length %s and B0 %dx%d\n", n, r, c)
  }

  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a
    case Epsilon:
      epsilon = a
    default:
      panic("Bfgs(): Invalid optional argument!")
    }
  }
  B, err := matrixInverse.Run(B0)
  if err != nil {
    return nil, err
  }
  return bfgs(newObjectiveInSitu(f), x0, B, epsilon, hook)
}
