/*
 *  Copyright 2011-2015 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "tiling_factor.h"

#include "neural_network_exception.h"
#include <boost/format.hpp>

namespace nnforge
{
	tiling_factor::tiling_factor()
		: f(1)
		, b_multiply(true)
	{
	}

	tiling_factor::tiling_factor(
		unsigned int f,
		bool b_multiply)
		: f(f)
		, b_multiply(b_multiply || (f == 1))
	{
	}

	tiling_factor::operator unsigned int() const
	{
		if (b_multiply)
			return f;
		else
			throw neural_network_exception((boost::format("Cannot convert tiling_factor %1% to unsigned int") % str()).str());
	}

	std::string tiling_factor::str() const
	{
		if (b_multiply)
			return (boost::format("%1%") % f).str();
		else
			return (boost::format("1/%1%") % f).str();
	}

	tiling_factor& tiling_factor::operator *=(const tiling_factor& other)
	{
		if (b_multiply == other.b_multiply)
			f *= other.f;
		else
		{
			if (f >= other.f)
			{
				if (f % other.f == 0)
					f /= other.f;
				else
					throw neural_network_exception((boost::format("Cannot multiply tiling factor %1% and %2% ") % str() % other.str()).str());
			}
			else
			{
				if (other.f % f == 0)
				{
					f = other.f / f;
					b_multiply = !b_multiply;
				}
				else
					throw neural_network_exception((boost::format("Cannot multiply tiling factor %1% and %2% ") % str() % other.str()).str());
			}
		}

		if (f == 1)
			b_multiply = true;

		return *this;
	}

	tiling_factor operator *(const tiling_factor& t1, const tiling_factor& t2)
	{
		tiling_factor res(t1);
		res *= t2;
		return res;
	}

	tiling_factor tiling_factor::get_inverse() const
	{
		return tiling_factor(f, !b_multiply);
	}

	bool operator ==(const tiling_factor& t1, const tiling_factor& t2)
	{
		return (t1.b_multiply == t2.b_multiply) && (t1.f == t2.f);
	}

	bool operator <(const tiling_factor& t1, const tiling_factor& t2)
	{
		if (t1.b_multiply)
		{
			if (t2.b_multiply)
				return t1.f < t2.f;
			else
				return false;
		}
		else
		{
			if (t2.b_multiply)
				return true;
			else
				return t1.f > t2.f;
		}
	}

	bool operator <=(const tiling_factor& t1, const tiling_factor& t2)
	{
		return (t1 < t2) || (t1 == t2);
	}

	bool operator !=(const tiling_factor& t1, const tiling_factor& t2)
	{
		return !(t1 == t2);
	}

	bool operator >=(const tiling_factor& t1, const tiling_factor& t2)
	{
		return !(t1 < t2);
	}

	bool operator >(const tiling_factor& t1, const tiling_factor& t2)
	{
		return !(t1 <= t2);
	}
}
