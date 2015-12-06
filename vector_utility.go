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

func (v Vector) Permute(_p []int) {
  if len(_p) != len(v) {
    panic("Permute(): permutation vector has invalid length!")
  }
  // make a copy of _p
  p := make([]int, len(_p))
  copy(p, _p)
  // permute vector
  for i := 0; i < len(v); i++ {
    if i != p[i] {
      // permute elements
      v[p[i]], v[i] = v[i], v[p[i]]
      // save permutation
      for k := 0; k < len(p); k++ {
        if p[k] == i {
          p[k], p[i] = p[i], i
          break
        }
      }
    }
  }
}
