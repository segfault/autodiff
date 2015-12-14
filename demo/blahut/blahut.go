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

package blahut

/* -------------------------------------------------------------------------- */

import   "math"
import . "github.com/pbenner/autodiff"

/* initialization of data structures
 * -------------------------------------------------------------------------- */

func blahut_init_q(n, m int) Matrix {
  return NullMatrix(ProbabilityType, m, n)
}

func blahut_init_r(n int) Vector {
  return NullVector(ProbabilityType, n)
}

/* naive Blahut implementation
 * -------------------------------------------------------------------------- */

func blahut_compute_q(channel Matrix, p Vector, q Matrix) {
  n, m := channel.Dims()
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      q.Set(Mul(channel.At(i, j), p.At(i)), j, i)
    }
    normalizeVector(q.Row(j))
  }
}

func blahut_compute_r(channel, q Matrix, r Vector) {
  n, m := channel.Dims()
  for i := 0; i < n; i++ {
    r[i].SetValue(0.0)
    for j := 0; j < m; j++ {
      if !math.IsInf(channel.At(i, j).LogValue(), -1) && // 0 log q = 0
         !math.IsInf(r[i].LogValue(), 1) {               // Inf + x = Inf
        r.Set(Sub(r.At(i), Mul(channel.At(i, j), Log(q.At(j, i)))), i)
      }
    }
    r.Set(Exp(Neg(r.At(i))), i)
  }
}

func blahut_compute_J(r Vector, J Scalar) {
  sum := NewScalar(r.ElementType(), 0.0)
  for i, _ := range r {
    sum = Add(sum, r[i])
  }
  J.Set(Div(Log(sum), NewScalar(r.ElementType(), math.Log(2.0))))
}

func blahut_compute_p(r Vector, lambda float64, p Vector) {
  for i, _ := range p {
    if math.IsInf(p.At(i).LogValue(), -1) {
      // p[i] = r[i]
      p.Set(r.At(i),
        i)
    } else {
      // p[i] = p[i]^(1-lambda) * r[i]^lambda
      p.Set(Mul(Pow(p.At(i), 1.0 - lambda), Pow(r.At(i), lambda)),
        i)
    }
  }
  normalizeVector(p)
}

func blahut(channel Matrix, p_init Vector, steps int,
  hook func(Vector, Scalar) bool,
  lambda float64) Vector {

  n, m := channel.Dims()
  p := p_init.Clone()
  q := blahut_init_q(n, m)
  r := blahut_init_r(n)
  J := NewProbability(0.0)

  for k := 0; k < steps; k++ {
    blahut_compute_q(channel, p, q)
    blahut_compute_r(channel, q, r)
    blahut_compute_J(r, J)
    blahut_compute_p(r, lambda, p)

    if hook != nil && hook(p, J) {
      break
    }
  }
  return p
}

/* main
 * -------------------------------------------------------------------------- */

func Blahut(channel Matrix, p_init Vector, steps int, args ...interface{}) Vector {
  // default values for optional parameters
  hook   := Hook  {nil}.Value
  lambda := Lambda{1.0}.Value

  // parse optional arguments
  for _, arg := range args {
    switch a := arg.(type) {
    case Hook:
      hook = a.Value
    case Lambda:
      lambda = a.Value
    default:
      panic("blahut(): Invalid optional argument!")
    }
  }
  return blahut(channel, p_init, steps, hook, lambda)
}
