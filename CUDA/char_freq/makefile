ifeq ($(OS),Windows_NT)
	RM=del
	export_files=char_freq.exe char_freq.lib char_freq.exp
endif

ifeq ($(OS),Linux)
	RM=rm -f
	export_files=char_freq char_freq.lib char_freq.exp
endif

char_freq: char_freq.cu
	nvcc -o char_freq char_freq.cu

clean:
	$(RM) $(export_files)