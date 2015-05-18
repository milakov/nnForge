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

#include "testing_complete_result_set_visualizer.h"

#include <set>

namespace nnforge
{
	class testing_complete_result_set_roc_visualizer : public testing_complete_result_set_visualizer
	{
	public:
		testing_complete_result_set_roc_visualizer(
			float threshold = 0.5F, // Used for accuracy and [optionally] for F-score
			float beta = 1.0F,
			const std::set<unsigned int>& neuron_id_valid_for_roc_set = std::set<unsigned int>()); // Used for F-score

		~testing_complete_result_set_roc_visualizer();

		virtual void dump(
			std::ostream& out,
			const testing_complete_result_set& val) const;

	public:
		float threshold;
		float beta;
		std::set<unsigned int> neuron_id_valid_for_roc_set;
	};
}

