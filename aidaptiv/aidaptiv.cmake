set(AIDAPTIV_DISPATCH_DIR ${CMAKE_CURRENT_LIST_DIR}/dispatch)
set(AIDAPTIV_DISPATCH_SRC ${AIDAPTIV_DISPATCH_DIR}/aidaptiv-dispatch.cpp)

add_library(aidaptiv SHARED IMPORTED)
set_target_properties(aidaptiv PROPERTIES
    IMPORTED_LOCATION             "${CMAKE_CURRENT_LIST_DIR}/bin/aidaptiv.dll"
    IMPORTED_IMPLIB               "${CMAKE_CURRENT_LIST_DIR}/lib/aidaptiv.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
)
