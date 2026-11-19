# SPDX-License-Identifier: MIT
#
# A 4diac FORTE module with HANDWIRED_RUN, a function block that runs handwired-rs program files. Copy this
# directory into FORTE's modules/, add `add_subdirectory(handwired)` to modules/CMakeLists.txt, build the C
# interface (`cargo build --release -p handwired-rs-run-c` in handwired-rs) and configure FORTE with
# -DFORTE_MODULE_HANDWIRED=ON -DHANDWIRED_RS_DIR=<the handwired-rs checkout>. plc/4diac/build.sh does all of it.

option(FORTE_MODULE_HANDWIRED "handwired-rs program files as function blocks" OFF)

if (NOT FORTE_MODULE_HANDWIRED)
    return()
endif ()

set(HANDWIRED_RS_DIR "" CACHE PATH "The handwired-rs checkout whose run-c crate is built")
set(HANDWIRED_RUN_C_LIBRARY "${HANDWIRED_RS_DIR}/target/release/libhandwired_rs_run_c.a" CACHE FILEPATH
    "The C interface's static library, from cargo build --release -p handwired-rs-run-c")
if (NOT EXISTS "${HANDWIRED_RUN_C_LIBRARY}")
    message(FATAL_ERROR "No ${HANDWIRED_RUN_C_LIBRARY}: set HANDWIRED_RS_DIR and build the run-c crate first")
endif ()

add_library(forte-handwired)
target_include_directories(forte-handwired PUBLIC $<BUILD_INTERFACE:${HANDWIRED_RS_DIR}/run-c/include>)
# The Rust standard library inside the static library needs the C library's threads, dynamic loading and maths.
find_package(Threads REQUIRED)
target_link_libraries(forte-handwired PUBLIC forte-core "${HANDWIRED_RUN_C_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS} m)
target_link_libraries(forte PUBLIC $<IF:$<BOOL:${BUILD_SHARED_LIBS}>,forte-handwired,$<LINK_LIBRARY:WHOLE_ARCHIVE,forte-handwired>>)

target_sources(forte-handwired PUBLIC
        FILE_SET HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES include/forte/handwired/HANDWIRED_RUN_fbt.h
)
target_sources(forte-handwired PRIVATE src/handwired/HANDWIRED_RUN_fbt.cpp)

# Like every FORTE module, part of its export set, since the forte target links it.
install(TARGETS forte-handwired EXPORT forte-export FILE_SET HEADERS)
