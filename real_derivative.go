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

/* derivatives of monadic functions
 * -------------------------------------------------------------------------- */

func (c *Real) monadic(a Scalar, v0, v1, v2 float64) Scalar {
  c.AllocForOne(a)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v2 +
            a.GetDerivative(2, i)*v1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v1)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) monadicLazy(a Scalar, v0 float64, f1, f2 func () float64) Scalar {
  c.AllocForOne(a)
  if c.Order >= 1 {
    v1 := f1()
    if c.Order >= 2 {
      v2 := f2()
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v2 +
            a.GetDerivative(2, i)*v1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v1)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) realMonadic(a *Real, v0, v1, v2 float64) *Real {
  c.AllocForOne(a)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v2 +
            a.GetDerivative(2, i)*v1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v1)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) realMonadicLazy(a *Real, v0 float64, f1, f2 func() float64) *Real {
  c.AllocForOne(a)
  if c.Order >= 1 {
    v1 := f1()
    if c.Order >= 2 {
      v2 := f2()
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v2 +
            a.GetDerivative(2, i)*v1)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v1)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

/* derivatives of dyadic functions
 * -------------------------------------------------------------------------- */

func (c *Real) dyadic(a, b Scalar, v0, v10, v01, v11, v20, v02 float64) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*v10 +
            b.GetDerivative(2, i)*v01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*v02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*v11*2)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v10 + b.GetDerivative(1, i)*v01)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) dyadicLazy(a, b Scalar, v0 float64, f1 func() (float64, float64), f2 func() (float64, float64, float64)) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    v10, v01 := f1()
    if c.Order >= 2 {
      v11, v20, v02 := f2()
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*v10 +
            b.GetDerivative(2, i)*v01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*v02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*v11*2)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v10 + b.GetDerivative(1, i)*v01)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) realDyadic(a, b *Real, v0, v10, v01, v11, v20, v02 float64) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    if c.Order >= 2 {
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*v10 +
            b.GetDerivative(2, i)*v01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*v02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*v11*2)
      }
    }
    // compute first derivatives
    for i := 0; i < c.GetN(); i++ {
      c.SetDerivative(1, i, a.GetDerivative(1, i)*v10 + b.GetDerivative(1, i)*v01)
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}

func (c *Real) realDyadicLazy(a, b Scalar, v0 float64, f1 func() (float64, float64), f2 func() (float64, float64, float64)) *Real {
  c.AllocForTwo(a, b)
  if c.Order >= 1 {
    v10, v01 := f1()
    if c.Order >= 2 {
      v11, v20, v02 := f2()
      // compute second derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(2, i,
            a.GetDerivative(2, i)*v10 +
            b.GetDerivative(2, i)*v01 +
            a.GetDerivative(1, i)*a.GetDerivative(1, i)*v20 +
            b.GetDerivative(1, i)*b.GetDerivative(1, i)*v02 +
            a.GetDerivative(1, i)*b.GetDerivative(1, i)*v11*2)
      }
      // compute first derivatives
      for i := 0; i < c.GetN(); i++ {
        c.SetDerivative(1, i, a.GetDerivative(1, i)*v10 + b.GetDerivative(1, i)*v01)
      }
    }
  }
  // compute new value
  c.SetValue(v0)
  return c
}
