# Compile all sources in lib/ into one static library and attach it to CoreLib.
file(GLOB LIBRARY_SOURCES "*.cpp")
add_library(mmto STATIC ${LIBRARY_SOURCES})
target_include_directories(mmto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mmto PRIVATE mfem)
target_link_libraries(CoreLib INTERFACE mmto)
