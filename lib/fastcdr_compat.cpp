#include <cstdint>

// Compatibility shim for fastcdr 2.2.5.
// rtabmap_msgs was compiled against fastcdr 2.2.7+ which exports
// Cdr::serialize(uint32_t) as a library symbol. 2.2.5 has it inline
// in the header only. This shim provides the missing out-of-line symbol.
//
// Build:
//   g++ -shared -fPIC -O2 -o libfastcdr_compat.so fastcdr_compat.cpp \
//       -L/opt/ros/jazzy/lib -lfastcdr -Wl,-rpath,/opt/ros/jazzy/lib

namespace eprosima {
namespace fastcdr {

class Cdr {
public:
    Cdr& serialize(const int32_t);    // exported from libfastcdr.so.2.2.5
    Cdr& serialize(const uint32_t);   // missing, provide here
};

Cdr& Cdr::serialize(const uint32_t value) {
    return serialize(static_cast<int32_t>(value));
}

} // namespace fastcdr
} // namespace eprosima
