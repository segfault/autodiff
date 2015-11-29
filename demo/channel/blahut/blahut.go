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

/* naive Blahut implementation
 * -------------------------------------------------------------------------- */

func normalize(p []float64) {
  sum := 0.0
  for _, v := range p {
    sum += v
  }
  for i, _ := range p {
    p[i] /= sum
  }
}

func MI(channel [][]float64, px []float64) float64 {
  n := len(channel)
  m := len(channel[0])
  if len(px) != n {
    panic("Input vector has invalid dimension!")
  }
  // compute p(y) from p(y|x)*p(x)
  py := make([]float64, m)
  for j := 0; j < m; j++ {
    for i := 0; i < n; i++ {
      py[j] += channel[i][j]*px[i]
    }
  }
  // compute mutual information
  mi := 0.0
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      if channel[i][j] > 0.0 {
        mi += channel[i][j]*px[i]*math.Log2(channel[i][j]/py[j])
      }
    }
  }
  return mi
}

func Blahut(channel [][]float64, px_init []float64, steps int, args ...interface{}) []float64 {
  // parse optional arguments
  var hook func([]float64) bool = nil
  for _, arg := range args {
    switch a := arg.(type) {
    case func([]float64) bool:
      hook = a
    default:
      panic("blahut(): Invalid optional argument!")
    }
  }
  n  := len(channel)
  m  := len(channel[0])
  // init px
  px := make([]float64, n)
  copy(px, px_init)
  // init qx
  qx := make([][]float64, m)
  for j := 0; j < m; j++ {
    qx[j] = make([]float64, n)
  }
  for k := 0; k < steps; k++ {
    // update qx
    for j := 0; j < m; j++ {
      for i := 0; i < n; i++ {
        qx[j][i] = channel[i][j]*px[i]
      }
      normalize(qx[j])
    }
    // update px
    for i := 0; i < n; i++ {
      sum := 0.0
      for j := 0; j < m; j++ {
        if channel[i][j] > 0.0 {
          sum += channel[i][j]*math.Log(qx[j][i])
        }
      }
      px[i] = math.Exp(sum)
    }
    normalize(px)
    // execute hook if available
    if hook != nil && hook(px) {
      break;
    }
  }
  return px
}

