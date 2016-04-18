/* Copyright (C) 2016 Philipp Benner
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

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

var MaxLogFloat64       float64
var EpsilonFloat64      float64
var PrecisionFloat64    int
var SeriesIterationsMax int

/* -------------------------------------------------------------------------- */

func init() {
  MaxLogFloat64       = math.Floor(math.Log(math.MaxFloat64))
  EpsilonFloat64      = math.Nextafter(1.0,2.0)-1.0
  PrecisionFloat64    = 53
  SeriesIterationsMax = 1000000
}
