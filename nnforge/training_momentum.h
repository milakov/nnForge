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

namespace nnforge
{
	class training_momentum
	{
	public:
		enum momentum_type
		{
			no_momentum,
			vanilla_momentum,
			nesterov_momentum,
			adam_momentum
		};

		training_momentum();

		training_momentum(
			const std::string& momentum_type_str,
			float momentum_val,
			float momentum_val2);

		training_momentum(
			momentum_type type,
			float momentum_val = 0.0F,
			float momentum_val2 = 0.0F);

		bool is_momentum_data() const;

		bool is_momentum_data2() const;

		momentum_type type;
		float momentum_val;
		float momentum_val2;
	};
}
