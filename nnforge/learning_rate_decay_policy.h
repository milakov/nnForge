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

#include <memory>

namespace nnforge
{
	class learning_rate_decay_policy
	{
	public:
		typedef std::shared_ptr<learning_rate_decay_policy> ptr;
		typedef std::shared_ptr<const learning_rate_decay_policy> const_ptr;

		virtual ~learning_rate_decay_policy() = default;

		virtual float get_learning_rate_decay(unsigned int epoch) const = 0;

	protected:
		learning_rate_decay_policy() = default;
		
	private:
		learning_rate_decay_policy(const learning_rate_decay_policy&) = delete;
		learning_rate_decay_policy& operator =(const learning_rate_decay_policy&) = delete;
	};
}

