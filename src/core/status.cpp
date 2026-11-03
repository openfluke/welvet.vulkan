#include "welvet/status.hpp"

// Header-only Status helpers; this TU keeps the core/ layout aligned with siblings.
namespace welvet {
namespace {
const char kStatusAnchor = 0;
}
const char* status_tu_anchor() { return &kStatusAnchor; }
} // namespace welvet
