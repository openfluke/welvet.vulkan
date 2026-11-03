#include "welvet/gaps.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static void usage() {
  std::fprintf(stderr,
               "welvet_dense_fwd (M0) — identity / gaps only until Dense lands\n"
               "  --version | --identity | --gaps | --help\n");
}

int main(int argc, char** argv) {
  std::string a = (argc >= 2) ? argv[1] : "--help";
  if (a == "--version" || a == "-v") {
    std::printf("%s\n", welvet_version());
    return 0;
  }
  if (a == "--identity") {
    auto st = welvet_vulkan_init();
    if (st.code != 0) {
      std::fprintf(stderr, "init: %s\n", st.message);
      return 1;
    }
    std::printf("%s\n", welvet_runtime_identity());
    return 0;
  }
  if (a == "--gaps") {
    welvet::gaps::print_census(std::cout);
    return 0;
  }
  usage();
  return (a == "--help" || a == "-h") ? 0 : 1;
}
