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

#pragma once

#include <nnforge/testing_complete_result_set_visualizer.h>
#include <nnforge/normalize_data_transformer.h>

#include <memory>
#include <ostream>

class galaxy_zoo_testing_complete_result_set_visualizer : public nnforge::testing_complete_result_set_visualizer
{
public:
	galaxy_zoo_testing_complete_result_set_visualizer(nnforge::normalize_data_transformer_smart_ptr nds);

	~galaxy_zoo_testing_complete_result_set_visualizer();

	virtual void dump(
		std::ostream& out,
		const nnforge::testing_complete_result_set& val) const;

private:
	nnforge::normalize_data_transformer_smart_ptr nds;
};
