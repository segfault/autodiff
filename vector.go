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

import "bytes"
import "bufio"
import "compress/gzip"
import "errors"
import "reflect"
import "strconv"
import "strings"
import "os"

/* vector type declaration
 * -------------------------------------------------------------------------- */

type Vector []Scalar

/* constructors
 * -------------------------------------------------------------------------- */

// Allocate a vector for scalars of type t (i.e. RealType, or ProbabilityType).
func NewVector(t ScalarType, values []float64) Vector {
  v := NilVector(len(values))

  for i, _ := range values {
    v[i] = NewScalar(t, values[i])
  }
  return v
}

// Allocate an empty vector of type t. All values are initialized to zero.
func NullVector(t ScalarType, length int) Vector {
  v := NilVector(length)

  for i := 0; i < length; i++ {
    v[i] = NewScalar(t, 0.0)
  }
  return v
}

// Create a vector without allocating memory for the scalar variables.
func NilVector(length int) Vector {
  return make(Vector, length)
}

/* -------------------------------------------------------------------------- */

// Create a deep copy of the vector.
func (v Vector) Clone() Vector {
  result := make(Vector, len(v))

  for i, _ := range v {
    result[i] = v[i].Clone()
  }
  return result
}

// Copy scalars from w into this vector. The lengths of both vectors must
// match.
func (v Vector) Copy(w Vector) {
  if len(v) != len(w) {
    panic("CopyFrom(): Vector dimensions do not match!")
  }
  for i := 0; i < len(w); i++ {
    v[i].Copy(w[i])
  }
}

/* imlement ScalarContainer
 * -------------------------------------------------------------------------- */

// Return the scalar at the given position. For vectors only a single
// argument should be passed.
func (v Vector) At(args ...int) Scalar {
  return v[args[0]].Clone()
}

func (v Vector) Set(value Scalar, args ...int) {
  v[args[0]].Copy(value)
}

func (v Vector) Map(f func(Scalar) Scalar) ScalarContainer {
  for i := 0; i < len(v); i++ {
    v[i] = f(v[i])
  }
  return v
}

func (v Vector) Reduce(f func(Scalar, Scalar) Scalar) Scalar {
  r := v[0]
  for i := 1; i < len(v); i++ {
    r = f(r, v[i])
  }
  return r
}

func (v Vector) ElementType() ScalarType {
  if len(v) > 0 {
    return reflect.TypeOf(v[0])
  }
  return nil
}

func (v Vector) Variables(order int) {
  Variables(order, v...)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (v Vector) Matrix(n, m int) Matrix {
  if n*m != len(v) {
    panic("Matrix dimension does not fit input vector!")
  }
  return Matrix{v, n, m, false}

}

func (v Vector) Slice() []float64 {
  s := make([]float64, len(v))
  for i, _ := range v {
    s[i] = v[i].Value()
  }
  return s
}

func (v Vector) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(", ")
    }
    buffer.WriteString(v[i].String())
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (v Vector) ToTable() string {
  var buffer bytes.Buffer

  for i, _ := range v {
    if i != 0 {
      buffer.WriteString(" ")
    }
    buffer.WriteString(v[i].String())
  }

  return buffer.String()
}

func ReadVector(t ScalarType, filename string) (Vector, error) {
  result := NewVector(t, []float64{})

  var scanner *bufio.Scanner
  // open file
  f, err := os.Open(filename)
  if err != nil {
    return result, err
  }
  defer f.Close()
  isgzip, err := isGzip(filename)
  if err != nil {
    return result, err
  }
  // check if file is gzipped
  if isgzip {
    g, err := gzip.NewReader(f)
    if err != nil {
      return result, err
    }
    defer g.Close()
    scanner = bufio.NewScanner(g)
  } else {
    scanner = bufio.NewScanner(f)
  }

  for scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    if len(fields) == 0 {
      continue
    }
    if len(result) != 0 {
      return result, errors.New("invalid table")
    }
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return result, errors.New("invalid table")
      }
      result = append(result, NewScalar(t, value))
    }
  }
  return result, nil
}
