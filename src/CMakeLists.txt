# Each source in src/ is one executable.
file(GLOB EXAMPLE_SOURCES *.cpp)
foreach(src ${EXAMPLE_SOURCES})
  get_filename_component(prog ${src} NAME_WE)
  add_executable(${prog} ${src})
  target_link_libraries(${prog} PRIVATE CoreLib)
endforeach()
