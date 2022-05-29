ifeq ($(OS),Windows_NT)
	RM=del
	export_files=string_matching.exe string_matching.lib string_matching.exp
endif

ifeq ($(OS),Linux)
	RM=rm -f
	export_files=string_matching string_matching.lib string_matching.exp
endif

string_matching: string_matching.cu
	nvcc -o string_matching string_matching.cu

clean:
	$(RM) $(export_files)