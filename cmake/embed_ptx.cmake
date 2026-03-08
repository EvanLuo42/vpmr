function(embed_ptx)
    cmake_parse_arguments(PARSE_ARGV 0 PTX "" "OUTPUT_TARGET" "SOURCES")

    if(NOT PTX_OUTPUT_TARGET)
        message(FATAL_ERROR "embed_ptx: OUTPUT_TARGET required")
    endif()
    if(NOT PTX_SOURCES)
        message(FATAL_ERROR "embed_ptx: SOURCES required")
    endif()

    set(PTX_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx_gen")
    file(MAKE_DIRECTORY "${PTX_GEN_DIR}")

    set(PTX_C_SOURCES "")
    set(PTX_HEADER_CONTENT "#pragma once\n")

    foreach(CU_SRC ${PTX_SOURCES})
        get_filename_component(CU_NAME "${CU_SRC}" NAME_WE)
        set(PTX_FILE "${PTX_GEN_DIR}/${CU_NAME}.ptx")
        set(PTX_C_FILE "${PTX_GEN_DIR}/${CU_NAME}_ptx.c")
        set(PTX_EMBED_VAR "${CU_NAME}_ptx")

        get_filename_component(CU_ABS "${CU_SRC}" ABSOLUTE)

        # Build -I flags for each CUDA toolkit include directory
        set(CUDA_INCLUDE_FLAGS "")
        foreach(CUDA_INC_DIR ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
            list(APPEND CUDA_INCLUDE_FLAGS -I "${CUDA_INC_DIR}")
        endforeach()

        add_custom_command(
            OUTPUT "${PTX_FILE}"
            COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx
                -O3
                --use_fast_math
                -I "${OPTIX_INCLUDE_DIR}"
                ${CUDA_INCLUDE_FLAGS}
                -o "${PTX_FILE}"
                "${CU_ABS}"
            DEPENDS "${CU_ABS}"
            COMMENT "Compiling ${CU_NAME}.cu to PTX"
        )

        add_custom_command(
            OUTPUT "${PTX_C_FILE}"
            COMMAND ${CMAKE_COMMAND}
                -DPTX_FILE="${PTX_FILE}"
                -DPTX_C_FILE="${PTX_C_FILE}"
                -DPTX_VAR_NAME="${PTX_EMBED_VAR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ptx_to_c.cmake"
            DEPENDS "${PTX_FILE}"
            COMMENT "Embedding ${CU_NAME}.ptx as C string"
        )

        list(APPEND PTX_C_SOURCES "${PTX_C_FILE}")
        string(APPEND PTX_HEADER_CONTENT "extern const char ${PTX_EMBED_VAR}[];\n")
    endforeach()

    set(PTX_HEADER_FILE "${PTX_GEN_DIR}/embedded_ptx.h")
    file(WRITE "${PTX_HEADER_FILE}" "${PTX_HEADER_CONTENT}")

    add_library(${PTX_OUTPUT_TARGET} STATIC ${PTX_C_SOURCES})
    target_include_directories(${PTX_OUTPUT_TARGET} PUBLIC "${PTX_GEN_DIR}")
endfunction()
