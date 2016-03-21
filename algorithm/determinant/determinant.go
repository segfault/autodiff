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

package determinant

/* -------------------------------------------------------------------------- */

//import   "math"
import . "github.com/pbenner/autodiff"

/* -------------------------------------------------------------------------- */

func Determinant(a Matrix) Scalar {

  n, _ := a.Dims()
  det  := ZeroScalar(a.ElementType())

  if (n < 1) {
    /* nothing to do */
  } else if n == 1 {
    det = a.At(0, 0)
  } else if n == 2 {
    det = Sub(
      Mul(a.At(0, 0), a.At(1, 1)),
      Mul(a.At(1, 0), a.At(0, 1)))
  } else {
    m := NullMatrix(a.ElementType(), n-1, n-1)
    for j1 := 0; j1 < n; j1++ {
      for i := 1; i < n; i++ {
        j2 := 0
        for j := 0; j < n; j++ {
          if j == j1 {
            continue
          }
          m.Set(a.At(i, j), i-1, j2);
          j2++;
        }
      }
      if (j1+1.0) % 2 == 0 {
        det = Add(det,  Mul(a.At(0, j1), Determinant(m)))
      } else {
        det = Sub(det,  Mul(a.At(0, j1), Determinant(m)))
      }
    }
  }

  return det
/*
   double det = 0;
   double **m = NULL;

   if (n < 1) {

   } else if (n == 1) {
      det = a[0][0];
   } else if (n == 2) {
      det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
   } else {
      det = 0;
      for (j1=0;j1<n;j1++) {
         m = malloc((n-1)*sizeof(double *));
         for (i=0;i<n-1;i++)
            m[i] = malloc((n-1)*sizeof(double));
         for (i=1;i<n;i++) {
            j2 = 0;
            for (j=0;j<n;j++) {
               if (j == j1)
                  continue;
               m[i-1][j2] = a[i][j];
               j2++;
            }
         }
         det += pow(-1.0,1.0+j1+1.0) * a[0][j1] * Determinant(m,n-1);
         for (i=0;i<n-1;i++)
            free(m[i]);
         free(m);
      }
   }
   return(det);
*/
}
