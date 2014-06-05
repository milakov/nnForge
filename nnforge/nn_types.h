/*
 *  Copyright 2011-2013 Maxim Milakov
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
#include <random>
#include <array>
#include <regex>

#ifdef NNFORGE_CPP11COMPILER
#define nnforge_shared_ptr std::shared_ptr
#define nnforge_normal_distribution std::normal_distribution
#define nnforge_uniform_real_distribution std::uniform_real_distribution
#define nnforge_uniform_int_distribution std::uniform_int_distribution
#define nnforge_mt19937 std::mt19937
#define nnforge_regex std::regex
#define nnforge_cmatch std::cmatch
#define nnforge_regex_search std::regex_search
#define nnforge_regex_match std::regex_match
#define nnforge_dynamic_pointer_cast std::dynamic_pointer_cast
#define nnforge_array std::array
#else
#define nnforge_shared_ptr std::tr1::shared_ptr
#define nnforge_normal_distribution std::tr1::normal_distribution
#define nnforge_uniform_real_distribution std::tr1::uniform_real
#define nnforge_uniform_int_distribution std::tr1::uniform_int
#define nnforge_mt19937 std::tr1::mt19937
#define nnforge_regex std::tr1::regex
#define nnforge_cmatch std::tr1::cmatch
#define nnforge_regex_search std::tr1::regex_search
#define nnforge_regex_match std::tr1::regex_match
#define nnforge_dynamic_pointer_cast std::tr1::dynamic_pointer_cast
#define nnforge_array std::tr1::array
#endif
