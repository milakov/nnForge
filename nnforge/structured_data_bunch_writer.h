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

#include "layer_configuration_specific.h"
#include "nn_types.h"

#include <map>

namespace nnforge
{
	class structured_data_bunch_writer
	{
	public:
		typedef nnforge_shared_ptr<structured_data_bunch_writer> ptr;

		~structured_data_bunch_writer();

		virtual void set_config_map(const std::map<std::string, layer_configuration_specific> config_map) = 0;

		virtual void write(const std::map<std::string, const float *>& data_map) = 0;

	protected:
		structured_data_bunch_writer();

	private:
		structured_data_bunch_writer(const structured_data_bunch_writer&);
		structured_data_bunch_writer& operator =(const structured_data_bunch_writer&);
	};
}
