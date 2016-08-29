/* Copyright (C) 2016 Philipp Benner
 *
 * Code ported from boost (boost.org).
 * boost/math/special_functions/digamma.hpp
 */

//  Copyright John Maddock 2006-7, 2013-14.
//  Copyright Paul A. Bristow 2007, 2013-14.
//  Copyright Nikhar Agrawal 2013-14
//  Copyright Christopher Kormanyos 2013-14

package special

/* -------------------------------------------------------------------------- */

import "math"

/* -------------------------------------------------------------------------- */

func finite_gamma_q(a, x float64) float64 {
  //
  // Calculates normalised Q when a is an integer:
  //
  e := math.Exp(-x)
  sum := e
  if(sum != 0) {
    term := sum
    for n := 1.0; n < a; n += 1.0 {
      term /= n
      term *= x
      sum  += term
    }
  }
  return sum
}

func finite_half_gamma_q(a, x float64) float64 {
  //
  // Calculates normalised Q when a is a half-integer:
  //
  e := math.Erfc(math.Sqrt(x))
  if e != 0.0 && a > 1.0 {
    term := math.Exp(-x) / math.Sqrt(math.Pi*x)
    term *= x
    half := 1.0/2.0
    term /= half
    sum  := term
    for n := 2.0; n < a; n += 1.0 {
      term /= n - half
      term *= x
      sum  += term
    }
    e += sum
  }
  return e;
}

func regularised_gamma_prefix(a, z float64) float64 {
  limit := math.Max(10.0, a)
  sum   := lower_gamma_series  (a, limit)/a
  sum   += upper_gamma_fraction(a, limit, 2.22045e-16)

  if a < 10.0 {
    // special case for small a:
    prefix := math.Pow(z/10.0, a)
    prefix *= math.Exp(10.0 - z)
    if 0.0 == prefix {
      prefix = math.Pow((z*math.Exp((10.0-z)/a))/10.0, a)
    }
    prefix /= sum
    return prefix
  }

  zoa    := z/a
  amz    := a - z
  alzoa  := a*math.Log(zoa)
  prefix := 0.0
  if math.Min(alzoa, amz) <= MinLogFloat64 || math.Max(alzoa, amz) >= MaxLogFloat64 {
    amza := amz / a
    if amza <= MinLogFloat64 || amza >= MaxLogFloat64 {
      prefix = math.Exp(alzoa + amz)
    } else {
      prefix = math.Pow(zoa*math.Exp(amza), a)
    }
  } else {
    prefix = math.Pow(zoa, a)*math.Exp(amz)
  }
  prefix /= sum
  return prefix
}

func full_igamma_prefix(a, z float64) float64 {
  prefix := 0.0
  alz    := a*math.Log(z)

  if z >= 1 {
    if alz < MaxLogFloat64 && -z > MinLogFloat64 {
      prefix = math.Pow(z, a)*math.Exp(-z)
    } else
    if a >= 1.0 {
      prefix = math.Pow(z/math.Exp(z/a), a)
    } else {
      prefix = math.Exp(alz - z)
    }
  } else {
    if alz > MinLogFloat64 {
      prefix = math.Pow(z, a)*math.Exp(-z)
    } else
    if z/a < MaxLogFloat64 {
      prefix = math.Pow(z/math.Exp(z/a), a)
    } else {
      prefix = math.Exp(alz - z)
    }
   }

   return prefix
}

func gamma_incomplete_imp(a, x float64, normalised, invert bool) float64 {

  result := 0.0

  is_int      := false
  is_half_int := false
  is_small_a  := a < 30 && a <= x + 1.0 && x < MaxLogFloat64

  if is_small_a {
    fa := math.Floor(a)
    if fa == a {
      is_int = true
    } else
    if math.Abs(fa - a) == 0.5 {
      is_half_int = true
    }
  }

  var eval_method int
   
  if is_int && x > 0.6 {
    // calculate Q via finite sum:
    invert = !invert
    eval_method = 0
  } else
  if is_half_int && x > 0.2  {
    // calculate Q via finite sum for half integer a:
    invert = !invert
    eval_method = 1
  } else
  if x < EpsilonFloat64 && a > 1 {
    eval_method = 6
  } else
  if x < 0.5 {
    //
    // Changeover criterion chosen to give a changeover at Q ~ 0.33
    //
    if -0.4/math.Log(x) < a {
      eval_method = 2
    } else {
      eval_method = 3
    }
  } else
  if x < 1.1 {
    //
    // Changover here occurs when P ~ 0.75 or Q ~ 0.25:
    //
    if x * 0.75 < a {
      eval_method = 2
    } else {
      eval_method = 3
    }
  } else {
    //
    // Begin by testing whether we're in the "bad" zone
    // where the result will be near 0.5 and the usual
     // series and continued fractions are slow to converge:
    //
    use_temme := false
    if normalised &&  a > 20 {
      sigma := math.Abs((x-a)/a)
      if a > 200 && PrecisionFloat64 <= 113 {
        //
        // This limit is chosen so that we use Temme's expansion
        // only if the result would be larger than about 10^-6.
        // Below that the regular series and continued fractions
        // converge OK, and if we use Temme's method we get increasing
        // errors from the dominant erfc term as it's (inexact) argument
        // increases in magnitude.
        //
        if 20/a > sigma*sigma {
          use_temme = true
        }
      } else
      if PrecisionFloat64 <= 64 {
        // Note in this zone we can't use Temme's expansion for 
        // types longer than an 80-bit real:
        // it would require too many terms in the polynomials.
        if sigma < 0.4 {
          use_temme = true
        }
      }
    }
    if use_temme {
      eval_method = 5
    } else {
      //
      // Regular case where the result will not be too close to 0.5.
      //
      // Changeover here occurs at P ~ Q ~ 0.5
      // Note that series computation of P is about x2 faster than continued fraction
      // calculation of Q, so try and use the CF only when really necessary, especially
      // for small x.
      //
      if x - 1.0/(3.0*x) < a {
        eval_method = 2
      } else {
        eval_method = 4
        invert = !invert
      }
    }
  }

  switch eval_method {
  case 0:
    result = finite_gamma_q(a, x)
    if normalised == false {
      result *= math.Gamma(a)
    }
  case 1:
    result = finite_half_gamma_q(a, x)
    if normalised == false {
      result *= math.Gamma(a)
    }
  case 2:
    // Compute P:
    if normalised {
      result = regularised_gamma_prefix(a, x)
    } else {
      result = full_igamma_prefix(a, x)
    }
    if result != 0 {
      //
      // If we're going to be inverting the result then we can
      // reduce the number of series evaluations by quite
      // a few iterations if we set an initial value for the
      // series sum based on what we'll end up subtracting it from
      // at the end.
      // Have to be careful though that this optimization doesn't 
      // lead to spurious numberic overflow.  Note that the
      // scary/expensive overflow checks below are more often
      // than not bypassed in practice for "sensible" input
      // values:
      //
      init_value := 0.0
      optimised_invert := false
      if invert {
        if normalised {
          init_value = 1
        } else {
          init_value = math.Gamma(a)
        }
        if normalised || result >= 1 || math.MaxFloat64*result > init_value {
          init_value /= result
          if normalised || a < 1 || math.MaxFloat64/a > init_value {
            init_value *= -a
            optimised_invert = true
          } else {
            init_value = 0
          }
        } else {
          init_value = 0
        }
      }
      result *= lower_gamma_series(a, x, init_value) / a
      if optimised_invert {
        invert = false
        result = -result
      }
    }
  case 3:
    // Compute Q:
    invert = !invert
    g := 0.0
    result = tgamma_small_upper_part(a, x, &g, invert)
    invert = false
    if normalised {
      result /= g
    }
  case 4:
    // Compute Q:
    if normalised {
      result = regularised_gamma_prefix(a, x)
    } else {
      result = full_igamma_prefix(a, x)
    }
    if result != 0 {
      result *= upper_gamma_fraction(a, x, 2.22045e-16)
    }
  case 5:
    result = igamma_temme_large(a, x, 0)
    if x >= a {
      invert = !invert
    }
  case 6:
    // x is so small that P is necessarily very small too,
    // use http://functions.wolfram.com/GammaBetaErf/GammaRegularized/06/01/05/01/01/
    if normalised {
      result = math.Pow(x, a)/a
    } else {
      result = math.Pow(x, a) / math.Gamma(a + 1.0)
    }
    result *= 1.0 - a*x/(a + 1.0)
  }
  if normalised && result > 1.0 {
    result = 1.0
  }
  if invert {
    gam := 1.0
    if !normalised {
      gam = math.Gamma(a)
    }
    result = gam - result
  }

  return result
}
