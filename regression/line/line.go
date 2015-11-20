
package line

/* -------------------------------------------------------------------------- */

import . "github.com/pbenner/autodiff/scalar"

/* -------------------------------------------------------------------------- */

type Line struct {
  slope     Scalar
  intercept Scalar
}

func NewLine(slope, intercept *Scalar) *Line {

  l := new(Line)
  l.slope     = *slope
  l.intercept = *intercept

  return l
}

func (l *Line) Slope() *Scalar {
  return &l.slope
}

func (l *Line) Intercept() *Scalar {
  return &l.intercept
}

func (l *Line) SetSlope(s *Scalar) {
  l.slope = *s
}

func (l *Line) SetIntercept(i *Scalar) {
  l.intercept = *i
}

func (l *Line) Eval(x *Scalar) *Scalar {

  return Add(Mul(&l.slope, x), &l.intercept)
}
