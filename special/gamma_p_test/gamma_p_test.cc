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

#include <iostream>
#include <iomanip>

#include <boost/math/special_functions/gamma.hpp>

/* -------------------------------------------------------------------------- */

using namespace std;
using namespace boost::math;

/* -------------------------------------------------------------------------- */

void test_lower_incomplete_gamma() {
        for (double  a = 1.0; a <= 4.0; a += 0.4) {
                for (double z = 0.05; z <= 4; z += 0.05) {
                        cout << "{"
                             << setw(4)
                             << setprecision( 1) << fixed << a << ", "
                             << setw(8)
                             << setprecision( 6) << fixed << z << ", "
                             << setprecision(20) << fixed << scientific
                             << boost::math::gamma_p(a, z)
                             << "},"
                             << endl;
                }
        }
}

int main() {
        test_lower_incomplete_gamma();
}
