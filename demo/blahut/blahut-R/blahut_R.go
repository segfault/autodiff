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

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "os"
import   "regexp"
import   "strconv"
import   "code.google.com/p/getopt"

import . "github.com/pbenner/autodiff"
import . "github.com/pbenner/autodiff/demo/blahut"

/* -------------------------------------------------------------------------- */

func parseRawVector(str string) []float64 {
  r := regexp.MustCompile("[-0-9.]+")
  m := r.FindAllStringSubmatch(str, -1)
  if m == nil {
    panic("parseRawVector(): Invalid vector!")
  }
  v := make([]float64, len(m))
  for i := 0; i < len(m); i++ {
    v[i], _ = strconv.ParseFloat(m[i][0], 64)
  }
  return v
}

func parseVector(str string) Vector {
  r := regexp.MustCompile("^[ \t]*num \\[([0-9]+):([0-9]+)\\] (.+)$")
  m := r.FindStringSubmatch(str)
  if m == nil {
    panic("parseVector(): Invalid vector!")
  }
  from, _ := strconv.Atoi(m[1])
  to,   _ := strconv.Atoi(m[2])
  v       := parseRawVector(m[3])
  if from != 1 || to != len(v){
    panic("parseVector(): Invalid vector!")
  }
  return NewVector(ProbabilityType, v)
}

func parseMatrix(str string) Matrix {
  r := regexp.MustCompile("^[ \t]*num \\[([0-9]+):([0-9]+), ([0-9]+):([0-9]+)\\] (.+)$")
  m := r.FindStringSubmatch(str)
  if m == nil {
    panic("parseMatrix(): Invalid matrix!")
  }
  rfrom, _ := strconv.Atoi(m[1])
  rto,   _ := strconv.Atoi(m[2])
  cfrom, _ := strconv.Atoi(m[3])
  cto,   _ := strconv.Atoi(m[4])
  if rfrom != 1 || cfrom != 1 {
    panic("parseMatrix(): Invalid matrix!")
  }
  v := parseRawVector(m[5])
  return NewMatrix(ProbabilityType, rto, cto, v)
}

func printVector(v Vector) {
  fmt.Print("c(")
  for i, _ := range(v) {
    if i != 0 {
      fmt.Print(", ")
    }
    fmt.Print(v[i].Value())
  }
  fmt.Print(")")
}

func hook(px Vector, J Scalar) bool {
  fmt.Printf("%s (J = %f)\n", px.String(), J.Value())
  return false
}

func main() {

  optLambda     := getopt.StringLong("lambda",     'l',   "", "proximal point step size [default: 1.0]")
  optIterations := getopt.IntLong   ("iterations", 'i', 1000, "number of iterations     [default: 1000]")
  optHelp       := getopt.BoolLong  ("help",       'h',       "print help")
  optVerbose    := getopt.BoolLong  ("verbose", 'v', "print verbose output")

  getopt.SetParameters("channel p_init")
  getopt.Parse()

  if *optHelp {
    getopt.Usage()
    os.Exit(0)
  }
  if len(getopt.Args()) != 2 {
    getopt.Usage()
    os.Exit(1)
  }

  // parse channel
  channel := parseMatrix(getopt.Args()[0])
  // parse initial value
  px_init := parseVector(getopt.Args()[1])
  // check dimensions
  if n, _ := channel.Dims(); len(px_init) != n {
    panic("Channel dimension does not match length of p_init!")
  }
  // convert lambda to float
  lambda, err := strconv.ParseFloat(*optLambda, 64)
  if err != nil {
    panic("Invalid lambda!")
  }
  // get number of iterations
  n := *optIterations
  // get hook
  h := hook
  if !*optVerbose {
    h = nil
  }

  pxn := Blahut(channel, px_init, n, Lambda{lambda}, Hook{h})

  printVector(pxn)
}
