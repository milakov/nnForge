/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "debug_util.h"

#include <fstream>

namespace nnforge
{
	void debug_util::dump_list(
		const float * buffer,
		size_t elem_count,
		const char * filepath,
		unsigned int elem_count_per_line)
	{
		std::ofstream out(filepath);
		for(size_t i = 0; i < elem_count; ++i)
		{
			if ((i % elem_count_per_line) != 0)
				out << '\t';
			out << buffer[i];
			if (((i + 1) % elem_count_per_line) == 0)
				out << std::endl;
		}
		if ((elem_count % elem_count_per_line) != 0)
			out << std::endl;
	}

	void debug_util::dump_list(
		const int * buffer,
		size_t elem_count,
		const char * filepath,
		unsigned int elem_count_per_line)
	{
		std::ofstream out(filepath);
		for(size_t i = 0; i < elem_count; ++i)
		{
			if ((i % elem_count_per_line) != 0)
				out << '\t';
			out << buffer[i];
			if (((i + 1) % elem_count_per_line) == 0)
				out << std::endl;
		}
		if ((elem_count % elem_count_per_line) != 0)
			out << std::endl;
	}
}
