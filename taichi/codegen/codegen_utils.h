#pragma once
#include "taichi/program/program.h"

namespace taichi::lang {

inline bool codegen_vector_type(CompileConfig *config) {
  if (config->real_matrix) {
    return true;
  }

  return false;
}

}  // namespace taichi::lang