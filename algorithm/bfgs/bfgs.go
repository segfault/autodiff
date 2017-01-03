/* Copyright (C) 2016, 2017 Philipp Benner
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

type Hessian struct {
  Value Matrix
}

type Epsilon struct {
  Value float64
}

type Hook struct {
  Value func(gradient, x Vector, y Scalar) bool
}

type Constraints struct {
  Value func(x Vector) bool
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

func bgfs_backtrackingLineSearch(f ObjectiveInSitu, x1, x2 Vector, y1, y2 Scalar, g1, g2, p1, p2 Vector, a1 Vector, t1, t2 Scalar, constraints Constraints) bool {
  c1  := 1e-3
  rho := t2
  rho.Reset()
  rho.SetValue(0.5)
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
    // check if new value satisfies constraints
    if constraints.Value == nil || (constraints.Value != nil && constraints.Value(x2)) {
      // check Wolfe conditions
      t1.VdotV(p1, g1)
      if y2.GetValue() <= y1.GetValue() + c1*a1[0].GetValue()*t1.GetValue() {
        break
      }
    }
    a1[0].Mul(rho, a1[0])
}
  return true
}

// update approximation of the Hessian matrix
func bfgs_updateB(g1, g2, p2 Vector, B1, B2 Matrix, t1, t2 Scalar, t3, t4 Vector, t5, t6 Matrix) {
  s := p2
  y := t3
  // y = Df(x2) - Df(x1)
  y.VsubV(g2, g1)
  // s.y
  t2.VdotV(s, y)
  // check if value is zero
  if math.Abs(t2.GetValue()) < 1e-16 {
    B2.Copy(B1)
    return
  }
  // y s^T
  t5.Outer(y, s)
  // B y s^T
  t5.MdotM(B1, t5)
  // s y^T B
  t6.Outer(s, y)
  // s y^T B
  t6.MdotM(t6, B1)
  // B y s^T + s y^T B
  t5.MaddM(t5, t6)
  // (B y s^T + s y^T B) / (s.y)
  t5.MdivS(t5, t2)
  // save result
  B2.MsubM(B1, t5)
  // y.B
  t4.VdotM(y, B1)
  // y.B.y
  t1.VdotV(t4, y)
  // s.y + y.B.y
  t1.Add(t1, t2)
  // (s.y)^2
  t2.Mul(t2, t2)
  // (s.y + y.B.y)/(s.y)^2
  t1.Div(t1, t2)
  // s s^T
  t5.Outer(s, s)
  // (s.y + y.B.y)/(s.y)^2 (s s^T)
  t5.MmulS(t5, t1)
  // save result
  B2.MaddM(B2, t5)
}

// update approximation of the inverse Hessian matrix
func bfgs_updateH(g1, g2, p2 Vector, H1, H2, I Matrix, t1, t2 Scalar, t3, t4 Vector, t5, t6 Matrix) bool {
  s := p2
  y := t3
  // y = Df(x2) - Df(x1)
  y.VsubV(g2, g1)
  // y^T s
  t1.VdotV(s, y)
  // check if value is zero
  if math.Abs(t1.GetValue()) == 0.0 {
    return false
  }
  // s y^T
  t5.Outer(s, y)
  // s y^T / (y^T s)
  t5.MdivS(t5, t1)
  // I - s y^T / (y^T s)
  t5.MsubM(I, t5)
  // [I - s y^T / (y^T s)] H1 [I - s y^T / (y^T s)]
  H2.MdotM(t5, H1)
  H2.MdotM(H2, t5)
  // s s^T
  t5.Outer(s, s)
  // s s^T / (y^T s)
  t5.MdivS(t5, t1)
  // [I - s y^T / (y^T s)] H1 [I - s y^T / (y^T s)] + s s^T / (y^T s)
  H2.MaddM(H2, t5)
  return true
}

func bfgs(f ObjectiveInSitu, x0 Vector, H0 Matrix, epsilon Epsilon, hook Hook, constraints Constraints) (Vector, error) {

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
  H1 := H0.Clone()
  H2 := NullDenseMatrix(t, n, n)
  // some temporary variables
  t1 := NullScalar(t)
  t2 := NullScalar(t)
  t3 := NullVector(t, n)
  t4 := NullVector(t, n)
  t5 := NullDenseMatrix(t, n, n)
  t6 := NullDenseMatrix(t, n, n)
  I  := IdentityMatrix(t, n)

  // evaluate objective function
  if err := f.Differentiate(x1, y1, g1); err != nil {
    return x1, fmt.Errorf("invalid initial value: %s", err)
  }
  // execute hook if available
  if hook.Value != nil && hook.Value(g1, x1, y1) {
    return x1, nil
  }
  for {
    bgfs_computeDirection(x1, y1, g1, H1, p1)

    if ok := bgfs_backtrackingLineSearch(f, x1, x2, y1, y2, g1, g2, p1, p2, a1, t1, t2, constraints); !ok {
      return x1, fmt.Errorf("line search failed")
    }
    // execute hook if available
    if hook.Value != nil && hook.Value(g2, x2, y2) {
      break
    }
    // evaluate stop criterion
    if Vnorm(g2).GetValue() < epsilon.Value {
      break
    }
    // evaluate objective at new position
    if err := f.Differentiate(x2, y2, g2); err != nil {
      return x1, fmt.Errorf("invalid value: %s", err)
    }
    if ok := bfgs_updateH(g1, g2, p2, H1, H2, I, t1, t2, t3, t4, t5, t6); !ok {
      // reset H to find a new direction
      H2.Copy(H0)
    }

    g1.Copy(g2)
    x1.Copy(x2)
    y1.Copy(y2)
    p1.Copy(p2)
    H1.Copy(H2)
  }
  return x1, nil
}

/* -------------------------------------------------------------------------- */

// x0: starting point
// B0: initial approximation to the Hessian matrix

func Run(f Objective, x0 Vector, args ...interface{}) (Vector, error) {

  hessian     := Hessian{ nil}
  hook        := Hook   { nil}
  epsilon     := Epsilon{1e-8}
  constraints := Constraints{ nil}

  n := len(x0)

  for _, arg := range args {
    switch a := arg.(type) {
    case Hessian:
      hessian = a
    case Hook:
      hook = a
    case Epsilon:
      epsilon = a
    case Constraints:
      constraints = a
    default:
      panic("Bfgs(): Invalid optional argument!")
    }
  }
  if hessian.Value == nil {
    hessian.Value = IdentityMatrix(x0.ElementType(), n)
  } else {
    r, c := hessian.Value.Dims()
    if n != r || n != c {
      return nil, fmt.Errorf("argument dimensions do not match, i.e. x0 has length %s and B0 has dimension %dx%d\n", n, r, c)
    }
  }
  H, err := matrixInverse.Run(hessian.Value)
  if err != nil {
    return nil, err
  }
  return bfgs(newObjectiveInSitu(f), x0, H, epsilon, hook, constraints)
}
