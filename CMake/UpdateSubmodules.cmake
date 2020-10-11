find_package(Git)
if (GIT_FOUND)
	option(UPDATE_SUBMODULES "Download missing submodules during build" ON)
	if (UPDATE_SUBMODULES )
		message("Updating submodules.")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
			RESULT_VARIABLE GIT_RESULT)
		if(NOT GIT_RESULT EQUAL "0")
			message(FATAL_ERROR "Failed to run git submodule with error ${GIT_SUBMOD_RESULT}.")
		endif()
	endif()
endif()
message("Finishing updated submodules.")
