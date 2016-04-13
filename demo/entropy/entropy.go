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

/* Objective:
 * ----------
 * maximize H(x) = - Sum_i p(x_i) log p(x_i)
 * subject to Sum_i p(x_i) = 1
 * =>
 * Find critical points of L(p) = H(x) + lambda (Sum_i p(x_i) - 1)
 * which is equivalent to
 *  i) Finding the roots of the gradient of L(p), i.e. with Newton's method
 * ii) Minimizing the norm of the gradient of L(p)
 */

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import . "github.com/pbenner/autodiff"
import   "github.com/pbenner/autodiff/algorithm/newton"
import   "github.com/pbenner/autodiff/algorithm/rprop"

/* gradient based optimization
 * -------------------------------------------------------------------------- */

func hook_g(gradient []float64, px Vector, s Scalar) bool {
  fmt.Println("px: ", px)
  return false
}

func hook_f(gradient Matrix, px Vector, s Vector) bool {
  fmt.Println("px: ", px)
  return false
}

/* Gradient of L(p) */
func objective_f(px Vector) Vector {
  n := len(px) - 1
  if len(px) != n+1 {
    panic("Input vector has invalid dimension!")
  }
  gradient := NullVector(RealType, n+1)
  // derivative with respect to px[i]
  for i := 0; i < n; i++ {
    gradient[i] = Sub(NewReal(-1), Log(px[i]))
    gradient[i] = Sub(gradient[i], px[n])
  }
  // derivative with respect to lambda
  gradient[n] = NewReal(-1.0)
  for i := 0; i < n; i++ {
    gradient[n] = Add(gradient[n], px[i])
  }
  return gradient
}

/* Norm of the gradient of L(p) */
func objective_g(px Vector) Scalar {
  return Pow(VNorm(objective_f(px)), NewBareReal(2.0))
}

func main() {
  // precision
  const epsilon = 1e-8
  // initial gradient step size
  const step    = 0.001

  // initial value for px
  px0v := []float64{0.5, 0.2, 0.3}
  // append initial value for lambda
  px0m := NewVector(RealType, append(px0v, 1))

  fmt.Println("Rprop optimization:")
  pxn1 := rprop.Run (objective_g, px0m, step, 0.5,
    rprop.Hook{hook_g},
    rprop.Epsilon{epsilon})
  fmt.Println("Newton optimization:")
  pxn2 := newton.Run(objective_f, px0m,
    newton.Hook{hook_f},
    newton.Epsilon{epsilon})

  fmt.Println("Rprop  p(x): ", pxn1)
  fmt.Println("Newton p(x): ", pxn2)
}
