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

#include "training_stat.h"

#include <boost/format.hpp>

namespace nnforge
{
	training_stat::training_stat()
	{
	}

	std::ostream& operator<< (std::ostream& out, const training_stat& val)
	{
		out << "Weight updates";

		for(unsigned int layer_id = 0; layer_id < val.absolute_updates.size(); ++layer_id)
		{
			if (!val.absolute_updates[layer_id].empty())
			{
				out << " #" << layer_id << " (" << (boost::format("%|1$.5e|") % val.absolute_updates[layer_id][0]);
				for(int i = 1; i < val.absolute_updates[layer_id].size(); ++i)
					out << (boost::format(", %|1$.5e|") % val.absolute_updates[layer_id][i]);
				out << ")";
			}
		}

		return out;
	}
}
