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

#include "learning_rate_decay_policy.h"

namespace nnforge
{
	class exponential_learning_rate_decay_policy : public learning_rate_decay_policy
	{
	public:
		exponential_learning_rate_decay_policy(
			float learning_rate_decay_rate = 1.0F,
			unsigned int start_decay_epoch = 0);

		virtual ~exponential_learning_rate_decay_policy();

		virtual float get_learning_rate_decay(unsigned int epoch) const;

	private:
		float learning_rate_decay_rate;
		unsigned int start_decay_epoch;
	};
}
