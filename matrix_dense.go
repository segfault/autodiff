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
import "fmt"
import "reflect"
import "strconv"
import "strings"
import "os"

/* matrix type declaration
 * -------------------------------------------------------------------------- */

type DenseMatrix struct {
  Values     Vector
  Rows       int
  Cols       int
  Transposed bool
  Tmp1       Vector
  Tmp2       Vector
}

/* constructors
 * -------------------------------------------------------------------------- */

func NewDenseMatrix(t ScalarType, rows, cols int, values []float64) *DenseMatrix {
  m := NilDenseMatrix(rows, cols)
  v := m.GetValues()
  f := ScalarConstructor(t)
  if len(values) == 1 {
    for i := 0; i < rows*cols; i++ {
      v[i] = f(values[0])
    }
  } else if len(values) == rows*cols {
    for i := 0; i < rows*cols; i++ {
      v[i] = f(values[i])
    }
  } else {
    panic("NewDenseMatrix(): Matrix dimension does not fit input values!")
  }
  m.initTmp()

  return m
}

func NullDenseMatrix(t ScalarType, rows, cols int) *DenseMatrix {
  m := DenseMatrix{}
  m.Values = NullVector(t, rows*cols)
  m.Rows   = rows
  m.Cols   = cols
  m.initTmp()
  return &m
}

func NilDenseMatrix(rows, cols int) *DenseMatrix {
  m := DenseMatrix{}
  m.Values = NilVector(rows*cols)
  m.Rows   = rows
  m.Cols   = cols
  return &m
}

func (matrix *DenseMatrix) initTmp() {
  t := matrix.ElementType()
  if len(matrix.Tmp1) == 0 {
    matrix.Tmp1 = NullVector(t, matrix.Rows)
  }
  if len(matrix.Tmp2) == 0 {
    matrix.Tmp2 = NullVector(t, matrix.Cols)
  }
}

/* copy and cloning
 * -------------------------------------------------------------------------- */

// Clone matrix including data.
func (matrix *DenseMatrix) Clone() Matrix {
  return &DenseMatrix{
    Values    : matrix.Values.Clone(),
    Rows      : matrix.Rows,
    Cols      : matrix.Cols,
    Transposed: matrix.Transposed}
}

// Clone matrix without duplicating data.
func (matrix *DenseMatrix) CloneShallow() Matrix {
  return &DenseMatrix{
    Values    : matrix.Values,
    Rows      : matrix.Rows,
    Cols      : matrix.Cols,
    Transposed: matrix.Transposed}
}

func (a *DenseMatrix) Copy(b Matrix) {
  n1, m1 := a.Dims()
  n2, m2 := b.Dims()
  if n1 != n2 || m1 != m2 {
    panic("Copy(): Matrix dimension does not match!")
  }
  a.Values.Copy(b.GetValues())
}

/* constructors for special types of matrices
 * -------------------------------------------------------------------------- */

func IdentityMatrix(t ScalarType, dim int) Matrix {
  matrix := NullDenseMatrix(t, dim, dim)
  for i := 0; i < dim; i++ {
    matrix.Set(NewScalar(t, 1), i, i)
  }
  return matrix
}

/* field access
 * -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) index(i, j int) int {
  if matrix.Transposed {
    return j*matrix.Rows + i
  } else {
    return i*matrix.Cols + j
  }
}

func (matrix *DenseMatrix) Dims() (int, int) {
  return matrix.Rows, matrix.Cols
}

func (matrix *DenseMatrix) GetValues() Vector {
  return matrix.Values
}

func (matrix *DenseMatrix) SetValues(v Vector) {
  matrix.Values = v
}

func (matrix *DenseMatrix) Row(i int) Vector {
  v := NilVector(matrix.Cols)
  for j := 0; j < matrix.Cols; j++ {
    v[j] = matrix.Values[matrix.index(i, j)]
  }
  return v
}

func (matrix *DenseMatrix) Col(j int) Vector {
  v := NilVector(matrix.Rows)
  for i := 0; i < matrix.Rows; i++ {
    v[i] = matrix.Values[matrix.index(i, j)]
  }
  return v
}

func (matrix *DenseMatrix) Diag() Vector {
  n, m := matrix.Dims()
  if n != m {
    panic("Diag(): not a square matrix!")
  }
  v := NilVector(n)
  for i := 0; i < n; i++ {
    v[i] = matrix.Values[matrix.index(i, i)]
  }
  return v
}

func (matrix *DenseMatrix) Submatrix(rfrom, rto, cfrom, cto int) Matrix {
  t := matrix.ElementType()
  n := rto-rfrom+1
  m := cto-cfrom+1
  r := NullDenseMatrix(t, n, m)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      r.Set(matrix.At(rfrom+i, cfrom+j), i, j)
    }
  }
  return r
}

func (matrix *DenseMatrix) Reshape(rows, cols int) error {
  if len(matrix.Values) != rows*cols {
    return errors.New("Reshape(): invalid parameters")
  }
  matrix.Rows = rows
  matrix.Cols = cols
  return nil
}

/* -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) At(i, j int) Scalar {
  return matrix.Values[matrix.index(i, j)].Clone()
}

func (matrix *DenseMatrix) ReferenceAt(i, j int) Scalar {
  return matrix.Values[matrix.index(i, j)]
}

func (matrix *DenseMatrix) RealReferenceAt(i, j int) *Real {
  return matrix.Values[matrix.index(i, j)].(*Real)
}

func (matrix *DenseMatrix) BareRealReferenceAt(i, j int) *BareReal {
  return matrix.Values[matrix.index(i, j)].(*BareReal)
}

func (matrix *DenseMatrix) Set(s Scalar, i, j int) {
  matrix.Values[matrix.index(i, j)].Copy(s)
}

func (matrix *DenseMatrix) SetReference(s Scalar, i, j int) {
  k := matrix.index(i, j)
  matrix.Values[k] = s
}

func (matrix *DenseMatrix) Reset() {
  for i := 0; i < len(matrix.Values); i++ {
    matrix.Values[i].Reset()
  }
}

func (matrix *DenseMatrix) ResetDerivatives() {
  for i := 0; i < len(matrix.Values); i++ {
    matrix.Values[i].ResetDerivatives()
  }
}

func (matrix *DenseMatrix) SetIdentity() {
  n, m := matrix.Dims()
  c := NewScalar(matrix.ElementType(), 1.0)
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      if i == j {
        matrix.ReferenceAt(i, j).Set(c)
      } else {
        matrix.ReferenceAt(i, j).Reset()
      }
    }
  }
}

/* implement ScalarContainer
 * -------------------------------------------------------------------------- */

func (matrix *DenseMatrix) Map(f func(Scalar) Scalar) {
  n, m := matrix.Dims()
  for i := 0; i < n; i++ {
    for j := 0; j < m; j++ {
      // set reference here to have identical behaviour to
      // vectors
      matrix.SetReference(f(matrix.At(i, j)), i, j)
    }
  }
}

func (matrix *DenseMatrix) Reduce(f func(Scalar, Scalar) Scalar) Scalar {
  n, m := matrix.Dims()
  r := matrix.At(0, 0)
  // first row
  for j := 1; j < m; j++ {
    r = f(r, matrix.ReferenceAt(0, j))
  }
  // all other rows
  for i := 1; i < n; i++ {
    for j := 0; j < m; j++ {
      r = f(r, matrix.ReferenceAt(i, j))
    }
  }
  return r
}

func (matrix *DenseMatrix) ElementType() ScalarType {
  if matrix.Rows > 0 && matrix.Cols > 0 {
    return reflect.TypeOf(matrix.Values[0])
  }
  return nil
}

func (matrix *DenseMatrix) ConvertElementType(t ScalarType) {
  matrix.Map(func(x Scalar) Scalar {
    return NewScalar(t, x.GetValue())
  })
}

func (matrix *DenseMatrix) Variables(order int) {
  Variables(order, matrix.Values...)
}

/* type conversion
 * -------------------------------------------------------------------------- */

func (m *DenseMatrix) String() string {
  var buffer bytes.Buffer

  buffer.WriteString("[")
  for i := 0; i < m.Rows; i++ {
    if i != 0 {
      buffer.WriteString(",\n ")
    }
    buffer.WriteString("[")
    for j := 0; j < m.Cols; j++ {
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

func (a *DenseMatrix) Table() string {
  var buffer bytes.Buffer

  n, m := a.Dims()

  for i := 0; i < n; i++ {
    if i != 0 {
      buffer.WriteString("\n")
    }
    for j := 0; j < m; j++ {
      if j != 0 {
        buffer.WriteString(" ")
      }
      buffer.WriteString(a.ReferenceAt(i,j).String())
    }
  }

  return buffer.String()
}

func (m *DenseMatrix) WriteMatrix(filename string) error {
  f, err := os.Create(filename)
  if err != nil {
    return err
  }
  defer f.Close()

  w := bufio.NewWriter(f)
  defer w.Flush()

  fmt.Fprintf(w, "%s\n", m.Table())

  return nil
}

func ReadMatrix(t ScalarType, filename string) (Matrix, error) {
  result := NewDenseMatrix(t, 0, 0, []float64{})
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
  result = NewDenseMatrix(t, rows, cols, data)

  return result, nil
}
