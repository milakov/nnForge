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

#pragma once

#include <string>

namespace nnforge
{
	class tiling_factor
	{
	public:
		tiling_factor();

		tiling_factor(
			unsigned int f,
			bool b_multiply = true);

		operator unsigned int() const;

		std::string str() const;

		tiling_factor get_inverse() const;

		tiling_factor& operator *=(const tiling_factor& other);

	private:
		unsigned int f;
		bool b_multiply;

		friend tiling_factor operator *(const tiling_factor& t1, const tiling_factor& t2);
		friend bool operator ==(const tiling_factor& t1, const tiling_factor& t2);
		friend bool operator <(const tiling_factor& t1, const tiling_factor& t2);
	};

	tiling_factor operator *(const tiling_factor& t1, const tiling_factor& t2);
	bool operator ==(const tiling_factor& t1, const tiling_factor& t2);
	bool operator <(const tiling_factor& t1, const tiling_factor& t2);
	bool operator <=(const tiling_factor& t1, const tiling_factor& t2);
	bool operator !=(const tiling_factor& t1, const tiling_factor& t2);
	bool operator >=(const tiling_factor& t1, const tiling_factor& t2);
	bool operator >(const tiling_factor& t1, const tiling_factor& t2);
}
