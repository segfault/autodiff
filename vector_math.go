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

func VEqual(a, b Vector) bool {
  if len(a) != len(b) {
    panic("VEqual(): Vector dimensions do not match!")
  }
  for i, _ := range (a) {
    if !Equal(a[i], b[i]) {
      return false
    }
  }
  return true
}

func VaddV(a, b Vector) Vector {
  if len(a) != len(b) {
    panic("VAdd(): Vector dimensions do not match!")
  }
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Add(a[i], b[i])
  }
  return r
}

func VaddS(a Vector, b Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Add(a[i], b)
  }
  return r
}

func VsubV(a, b Vector) Vector {
  if len(a) != len(b) {
    panic("VSub(): Vector dimensions do not match!")
  }
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Sub(a[i], b[i])
  }
  return r
}

func VsubS(a Vector, b Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Sub(a[i], b)
  }
  return r
}

func Vnorm(a Vector) Scalar {
  r := Pow(a[0], NewBareReal(2))
  for i := 1; i < len(a); i++ {
    r = Add(r, Pow(a[i], NewBareReal(2)))
  }
  return Sqrt(r)
}

func VmulV(a, b Vector) Scalar {
  r := ZeroScalar(a.ElementType())
  for i := 0; i < len(a); i++ {
    r = Add(r, Mul(a[i], b[i]))
  }
  return r
}

func VmulS(a Vector, s Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Mul(a[i], s)
  }
  return r
}

func VdivS(a Vector, s Scalar) Vector {
  r := NullVector(a.ElementType(), len(a))
  for i := 0; i < len(a); i++ {
    r[i] = Div(a[i], s)
  }
  return r
}
