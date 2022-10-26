#pragma once

struct RootMeta : public StructMeta {
  int tag;
};

STRUCT_FIELD(RootMeta, tag);

#ifdef ARCH_amdgpu

#endif
void Root_activate(Ptr meta, Ptr node, int i) {
}

#ifdef ARCH_amdgpu

#endif
i32 Root_is_active(Ptr meta, Ptr node, int i) {
  return 1;
}

#ifdef ARCH_amdgpu

#endif
Ptr Root_lookup_element(Ptr meta, Ptr node, int i) {
  // only one element
  return node;
}

#ifdef ARCH_amdgpu

#endif
i32 Root_get_num_elements(Ptr meta, Ptr node) {
  return 1;
}
