set(S2D_SOURCE_FILES
	aabb.c
	allocate.c
	allocate.h
	array.c
	array.h
	block_allocator.c
	block_allocator.h
	body.c
	body.h
	broad_phase.c
	broad_phase.h
	contact.c
	contact.h
	core.h
	distance.c
	dynamic_tree.c
	geometry.c
	hull.c
	joint.c
	joint.h
	manifold.c
	math.c
	mouse_joint.c
	pool.c
	pool.h
	revolute_joint.c
	shape.c
	shape.h
	solve_avbd.c
	solve_common.c
	solve_pgs.c
	solve_pgs_ngs.c
	solve_pgs_ngs_block.c
	solve_pgs_soft.c
	solve_soft_step.c
	solve_tgs_ngs.c
	solve_tgs_soft.c
	solve_tgs_sticky.c
	solve_xpbd.c
	solvers.h
	stack_allocator.c
	stack_allocator.h
	table.c
	table.h
	world.c
	world.h
)

set(S2D_API_FILES
	../include/solver2d/aabb.h
	../include/solver2d/solver2d.h
	../include/solver2d/callbacks.h
	../include/solver2d/color.h
	../include/solver2d/constants.h
	../include/solver2d/debug_draw.h
	../include/solver2d/distance.h
	../include/solver2d/dynamic_tree.h
	../include/solver2d/geometry.h
	../include/solver2d/hull.h
	../include/solver2d/id.h
	../include/solver2d/joint_types.h
	../include/solver2d/manifold.h
	../include/solver2d/math.h
	../include/solver2d/timer.h
	../include/solver2d/types.h
)

add_library(solver2d ${S2D_SOURCE_FILES} ${S2D_API_FILES})
target_include_directories(solver2d
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(solver2d PROPERTIES
	C_STANDARD 17
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)

message(STATUS "CMake C compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "CMake C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake system name: ${CMAKE_SYSTEM_NAME}")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src" FILES ${S2D_SOURCE_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../include" PREFIX "include" FILES ${S2D_API_FILES})
