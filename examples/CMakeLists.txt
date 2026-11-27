set(ATTENTION_EXAMPLE_OPERATORS kda gdn gla dsa msa qsa)

foreach(operator IN LISTS ATTENTION_EXAMPLE_OPERATORS)
    add_executable(${operator}_example ${operator}/main.cpp)
    target_link_libraries(${operator}_example PRIVATE attention_primitives::attention_primitives)
    if(BUILD_TESTING)
        add_test(NAME ${operator}_example COMMAND ${operator}_example)
    endif()
endforeach()
