/*
 *  Copyright 2011-2016 Maxim Milakov
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
#include <vector>

namespace nnforge
{
	struct rgba_color
	{
		rgba_color()
		{
		}

		rgba_color(unsigned int val)
		{
			c.val = val;
		}

		union
		{
			unsigned char rgba[4];
			unsigned int val;
		} c;
	};

	unsigned int get_distance_squared(const rgba_color& x, const rgba_color& y);

	class color_palette
	{
	public:
		std::string get_color_name(unsigned int logical_color_id) const;

		static color_palette& get_singleton();

	private:
		color_palette();

		std::vector<rgba_color> colors;
	};
}
