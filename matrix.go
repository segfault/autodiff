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

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type Matrix struct {
  values Vector
  rows   int
  cols   int
  t      bool
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewMatrix(t ScalarType, rows, cols int, values []float64) Matrix {
  m := NilMatrix(rows, cols)
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      m.values[i] = NewScalar(t, values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      m.values[i] = NewScalar(t, values[i])
    }
  } else {
    panic("NewMatrix(): Matrix dimension does not fit input values!")
  }
  return m
}

func NullMatrix(t ScalarType, rows, cols int) Matrix {
  return Matrix{NullVector(t, rows*cols), rows, cols, false}
}

func NilMatrix(rows, cols int) Matrix {
  return Matrix{NilVector(rows*cols), rows, cols, false}
}

/* copy and cloning
 * -------------------------------------------------------------------------- */

func (matrix Matrix) Clone() Matrix {
  return Matrix{
    values: matrix.values.Clone(),
    rows  : matrix.rows,
    cols  : matrix.cols,
    t     : matrix.t}
}

func (m1 Matrix) Copy(m2 Matrix) {
  if m1.rows != m2.rows || m1.cols != m2.cols {
    panic("Copy(): Matrix dimension does not match!")
  }
  m1.values.Copy(m2.values)
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func IdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.Set(NewScalar(t, 1), i, i)
  }
  return matrix
}

/* field access
 * -------------------------------------------------------------------------- */

func (matrix Matrix) index(i, j int) int {
  var k int
  if matrix.t {
    k = j*matrix.rows + i
  } else {
    k = i*matrix.cols + j
  }
  return k
}

func (matrix Matrix) Dims() (int, int) {
  return matrix.rows, matrix.cols
}

func (matrix Matrix) Values() Vector {
  return matrix.values
}

func (matrix *Matrix) SetValues(v Vector) {
  matrix.values = v
}

func (matrix *Matrix) Row(i int) Vector {
  n := matrix.index(i, 0)
  m := matrix.index(i, matrix.cols)
  return matrix.values[n:m]
}

func (matrix *Matrix) Col(j int) Vector {
  v := NilVector(matrix.rows)
  for i := 0; i < matrix.rows; i++ {
    v[i] = matrix.values[matrix.index(i, j)]
  }
  return v
}

func (matrix *Matrix) Diag() Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := NilVector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.values[matrix.index(i, i)]
  }
  return v
}

func (matrix *Matrix) Submatrix(rfrom, rto, cfrom, cto int) Matrix {
  t := matrix.ElementType()
  n := rto-rfrom+1
  m := cto-cfrom+1
  r := NullMatrix(t, n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.Set(matrix.At(rfrom+i, cfrom+j), i, j)
    }
  }
  return r
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */

func (matrix Matrix) At(args ...int) Scalar {
  i := args[0]
  j := args[1]
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("At(): Index out of bounds!")
  }
  // to avoid confusion we clone the value before
  // returning a reference to it
  //
  // take for instane:
  // c := m.At(0, 0)
  // m.Set(1, 0, 0)
  // which would alter the value of c!
  return matrix.values[k].Clone()
}

func (matrix Matrix) Set(s Scalar, args ...int) {
  i := args[0]
  j := args[1]
  k := matrix.index(i, j)
  if k >= len(matrix.values) {
    panic("Set(): Index out of bounds!")
  }
  matrix.values[k].Copy(s)
}

func (matrix Matrix) ElementType() ScalarType {
  if matrix.rows > 0 && matrix.cols > 0 {
    return reflect.TypeOf(matrix.values[0])
  }
  return nil
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (m Matrix) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i := 0; i < m.rows; i++ {
    if i != 0 {
      buffer.WriteString(",\n ")
    }
    buffer.WriteString("[")
    for j := 0; j < m.cols; j++ {
      if j != 0 {
        buffer.WriteString(", ")
      }
      buffer.WriteString(m.At(i,j).String())
    }
    buffer.WriteString("]")
  }
  buffer.WriteString("]")

  return buffer.String()
}

func (m Matrix) ToTable() string {
  var buffer bytes.Buffer

  for i := 0; i < m.rows; i++ {
    if i != 0 {
      buffer.WriteString("\n")
    }
    for j := 0; j < m.cols; j++ {
      if j != 0 {
        buffer.WriteString(" ")
      }
      buffer.WriteString(m.At(i,j).String())
    }
  }

  return buffer.String()
}

func ReadMatrix(t ScalarType, filename string) (Matrix, error) {
  result := NewMatrix(t, 0, 0, []float64{})
  data   := []float64{}
  rows   := 0
  cols   := 0

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
    if cols == 0 {
      cols = len(fields)
    }
    if cols != len(fields) {
      return result, errors.New("invalid table")
    }
    for i := 0; i < len(fields); i++ {
      value, err := strconv.ParseFloat(fields[i], 64)
      if err != nil {
        return result, errors.New("invalid table")
      }
      data = append(data, value)
    }
    rows++
  }
  result = NewMatrix(t, rows, cols, data)

  return result, nil
}
