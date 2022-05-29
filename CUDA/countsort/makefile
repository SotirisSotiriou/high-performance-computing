ifeq ($(OS),Windows_NT)
	RM=del
	export_files=countsort.lib countsort.exp countsort.exe
endif

ifeq ($(OS),Linux)
	RM=rm -f
	export_files=countsort countsort.lib countsort.exp
endif

countsort:
	nvcc -o countsort countsort.cu

clean:
	$(RM) $(export_files)